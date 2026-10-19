//! The v0 layer two-level preconditioner (DECISIONS D17 first rung).
//!
//! One global near-pattern matrix, built exactly like the cubel-global
//! assembler, but its entries are split by copper layer at assembly
//! time: rows (tagged by their first branch's layer) are permuted into
//! contiguous layer groups, each group's entries form a block CSC that
//! is factored numerically per frequency, and the entries between two
//! different layers ride a separate cross-layer CSR applied in a
//! multiplicative block-Gauss-Seidel style (forward ascending sweep,
//! then a recomputing descending sweep - `2 x layers` block solves per
//! apply). Cross-layer couplings therefore participate in every
//! preconditioner apply, which is what the cubel-global pattern could
//! not do within its single factor: the structural cure for the
//! cross-layer coupling blindness behind the MPPT 365-iteration row.
//!
//! This module COPIES the CUBEL global column-accumulator assembler
//! verbatim in structure (dense accumulator per column over the
//! branch-pair row expansion, rayon chunked flushes); the only change
//! is the per-entry layer routing. It never touches the working
//! `cubel-global` path.

use crate::cubel::{hwm_mib, near_pair_data, FieldParts};
use crate::filament::Filament;
use crate::mesh::MeshMatrix;
use faer::linalg::solvers::SolveCore;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::traits::ComplexField as FaerComplexField;
use rayon::prelude::*;

/// One-shot flag so a fallback print appears once per process.
static FALLBACK_SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One layer's CSC block over the block-local (perm-group) indices,
/// plus the symbolic factorization built once at pattern time.
struct LayerCsc {
    /// Block dimension (= internal rows tagged with this layer).
    n: usize,
    /// CSC over local indices 0..n (column-major, ascending rows).
    col_ptr: Vec<u32>,
    row_idx: Vec<u32>,
    re_part: Vec<f64>,
    im_part: Vec<f64>,
    /// COLAMD symbolic LU, `None` for empty layers or failed builds
    /// (the apply then treats the block as an identity, one-shot
    /// warned).
    symbolic: Option<SymbolicLu<u32>>,
}

/// The omega-independent pattern of the two-level split.
pub(crate) struct TwoLevel {
    /// Dimension: the internal mesh rows.
    n: usize,
    layer_count: usize,
    /// Permuted-index ranges `(start, end)` per layer, contiguous.
    layer_bounds: Vec<(usize, usize)>,
    /// Permuted position -> internal row index.
    perm: Vec<u32>,
    /// Per-layer blocks, layers ascending in z.
    layers: Vec<LayerCsc>,
    /// Cross-layer entries as a CSR over permuted rows 0..=n. Each
    /// entry couples two rows of DIFFERENT layers; the matrix is
    /// symmetric, so one CSR serves both GS directions.
    cross_col_ptr: Vec<u32>,
    cross_row_idx: Vec<u32>,
    cross_re: Vec<f64>,
    cross_im: Vec<f64>,
}

/// The numeric factors for one frequency: per-layer LUs plus the
/// omega-scaled cross values (aligned with the cross CSR entries).
pub(crate) struct TwoLevelFactors<T> {
    lus: Vec<Option<Lu<u32, T>>>,
    cross_values: Vec<T>,
}

impl TwoLevel {
    /// Builds the pattern. Layer tagging, permutation and routing are
    /// all local to this function; the near-pair kernel list comes from
    /// the shared `near_pair_data` (computed once, upper side only).
    pub(crate) fn build(
        mesh: &MeshMatrix,
        filaments: &[Filament],
        resistances: &[f64],
        near_radius: f64,
    ) -> Result<Self, crate::inductance::KernelError> {
        let internal = mesh.internal_count();
        if internal == 0 {
            return Ok(Self {
                n: 0,
                layer_count: 1,
                layer_bounds: vec![(0, 0)],
                perm: Vec::new(),
                layers: vec![empty_layer()],
                cross_col_ptr: vec![0],
                cross_row_idx: Vec::new(),
                cross_re: Vec::new(),
                cross_im: Vec::new(),
            });
        }
        let wall = std::time::Instant::now();
        let pattern = near_pair_data(mesh, filaments, internal, near_radius)?;
        let partner_total: usize = pattern.partner_all.iter().map(|p| p.len()).sum();
        eprintln!(
            "two-level: near pairs {partner_total} directed (avg {} per branch, wall {:.3} s)",
            partner_total / filaments.len().max(1),
            wall.elapsed().as_secs_f64(),
        );

        // Layer tagging: rows ride their FIRST branch's layer (the
        // recorded v0 tagging rule); spanning rows are counted as their
        // own population for the log.
        let branch_layer = crate::census::branch_layers(filaments);
        let layer_count = branch_layer.iter().max().map_or(1, |&m| m + 1);
        let mut row_layer = vec![0usize; internal];
        let mut spanning_rows = 0usize;
        for (row, layer) in row_layer.iter_mut().enumerate() {
            let branches = mesh.row(row);
            let first = branches.first().map_or(0, |&(b, _)| branch_layer[b]);
            let mixed =
                branches.len() > 1 && branches.iter().any(|&(b, _)| branch_layer[b] != first);
            if mixed {
                spanning_rows += 1;
            }
            *layer = first;
        }

        // Permutation into contiguous layer groups, stable within a
        // layer (ascending original row), so local index arithmetic
        // stays the original-order arithmetic of the swept rows that
        // share a layer.
        let mut perm: Vec<u32> = (0..internal as u32).collect();
        perm.sort_by_key(|&r| row_layer[r as usize]);
        let mut perm_pos = vec![0u32; internal];
        for (p, &r) in perm.iter().enumerate() {
            perm_pos[r as usize] = p as u32;
        }
        let mut layer_sizes = vec![0usize; layer_count];
        for &l in &row_layer {
            layer_sizes[l] += 1;
        }
        let mut layer_bounds: Vec<(usize, usize)> = Vec::with_capacity(layer_count);
        let mut start = 0usize;
        for &count in &layer_sizes {
            layer_bounds.push((start, start + count));
            start += count;
        }
        eprintln!(
            "two-level: {layer_count} layers, rows per layer {layer_sizes:?}, spanning rows {spanning_rows}"
        );

        // Layer-aligned rayon chunks over PERMUTED columns: a chunk
        // never crosses a layer boundary, so its block-local col indices
        // are contiguous and its cross edges share one permutation
        // convention. Copy of the cubel-global column-accumulator
        // structure (duplicates merge inside the sweep; peak memory is
        // CSC + O(n) scratch per worker).
        #[derive(Default)]
        struct ChunkOut {
            col_ptr: Vec<u32>,
            row_idx: Vec<u32>,
            re_part: Vec<f64>,
            im_part: Vec<f64>,
            cross: Vec<(u32, u32, f64, f64)>,
            peak_col: usize,
        }
        let threads = rayon::current_num_threads().max(1);
        let chunk_len = internal.div_ceil((threads * 4).max(1)).max(1);
        let mut chunk_specs: Vec<(usize, usize, usize)> = Vec::new();
        for (layer, &(lo, hi)) in layer_bounds.iter().enumerate() {
            let mut a = lo;
            while a < hi {
                let b = (a + chunk_len).min(hi);
                chunk_specs.push((a, b, layer));
                a = b;
            }
        }
        let chunk_count = chunk_specs.len().max(1);
        eprintln!("two-level: assembling {chunk_count} chunks over {threads} threads");
        let progress = std::sync::atomic::AtomicUsize::new(0);
        let printed = std::sync::atomic::AtomicUsize::new(1);
        let chunks: Vec<ChunkOut> = chunk_specs
            .clone()
            .into_par_iter()
            .map(|(lo, hi, layer)| {
                let mut out = ChunkOut::default();
                let mut col_ptr = vec![0u32; hi - lo + 1];
                let mut acc_re = vec![0.0f64; internal];
                let mut acc_im = vec![0.0f64; internal];
                let mut touched = vec![false; internal];
                let mut visited: Vec<u32> = Vec::new();
                for p in lo..hi {
                    let col = perm[p] as usize;
                    let block_start = layer_bounds[layer].0;
                    for &(a, sign_a) in mesh.row(col) {
                        let re_a = *resistances.get(a).unwrap_or(&0.0);
                        let l_self = pattern.self_l[a];
                        for inc in &pattern.branch_rows[a] {
                            let i = inc.row as usize;
                            let prod = sign_a * inc.sign;
                            if !touched[i] {
                                touched[i] = true;
                                visited.push(i as u32);
                            }
                            acc_re[i] += prod * re_a;
                            acc_im[i] += prod * l_self;
                        }
                        for &(other, l) in &pattern.partner_all[a] {
                            for inc in &pattern.branch_rows[other as usize] {
                                let i = inc.row as usize;
                                let prod = sign_a * inc.sign;
                                if !touched[i] {
                                    touched[i] = true;
                                    visited.push(i as u32);
                                }
                                acc_im[i] += prod * l;
                            }
                        }
                    }
                    if visited.is_empty() {
                        col_ptr[p - lo + 1] = out.row_idx.len() as u32;
                        continue;
                    }
                    visited.sort_unstable();
                    for &i in &visited {
                        let i = i as usize;
                        if row_layer[i] == layer {
                            out.row_idx.push(perm_pos[i] - block_start as u32);
                            out.re_part.push(acc_re[i]);
                            out.im_part.push(acc_im[i]);
                        } else {
                            out.cross
                                .push((perm_pos[i], p as u32, acc_re[i], acc_im[i]));
                        }
                    }
                    col_ptr[p - lo + 1] = out.row_idx.len() as u32;
                    out.peak_col = out.peak_col.max(visited.len());
                    for &i in &visited {
                        let i = i as usize;
                        acc_re[i] = 0.0;
                        acc_im[i] = 0.0;
                        touched[i] = false;
                    }
                    visited.clear();
                }
                out.col_ptr = col_ptr;
                let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let percent = 100 * done / chunk_count;
                let prev = printed.fetch_max(percent, std::sync::atomic::Ordering::Relaxed);
                if prev < percent {
                    eprintln!("two-level: assemble {}% ({} chunks done)", percent, done);
                }
                out
            })
            .collect();

        // Stitch per-layer segments (chunk_specs flip to runs in layer
        // order, since each layer's chunks were pushed contiguously) and
        // assemble the blocks.
        let mut layers: Vec<LayerCsc> = (0..layer_count)
            .map(|_| LayerCsc {
                n: 0,
                col_ptr: vec![0],
                row_idx: Vec::new(),
                re_part: Vec::new(),
                im_part: Vec::new(),
                symbolic: None,
            })
            .collect();
        for (chunk_index, chunk) in chunks.iter().enumerate() {
            let layer = chunk_specs[chunk_index].2;
            let base = layers[layer].row_idx.len() as u32;
            for &offset in chunk.col_ptr.iter().skip(1) {
                layers[layer].col_ptr.push(base + offset);
            }
            layers[layer].row_idx.extend_from_slice(&chunk.row_idx);
            layers[layer].re_part.extend_from_slice(&chunk.re_part);
            layers[layer].im_part.extend_from_slice(&chunk.im_part);
        }
        for (layer, bounds) in layer_bounds.iter().enumerate() {
            layers[layer].n = bounds.1 - bounds.0;
        }

        // Empty blocks: normalize to the empty-layer shape so the
        // factor/apply path can treat them uniformly.
        for (layer, block) in layers.iter_mut().enumerate() {
            if block.n > 0 && block.row_idx.is_empty() {
                eprintln!(
                    "two-level: warning: layer {layer} has {0} pattern entries; block never fills",
                    block.n
                );
            }
        }

        // Cross CSR: merge duplicate (row, col) pairs (the branch-pair
        // expansion produces them in both chunks' accumulators), then
        // lay out a row-major CSR over permuted rows.
        let mut cross: Vec<(u32, u32, f64, f64)> = Vec::new();
        for chunk in &chunks {
            cross.extend_from_slice(&chunk.cross);
        }
        cross.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut merged: Vec<(u32, u32, f64, f64)> = Vec::new();
        for &(row, col, re, im) in &cross {
            match merged.last_mut() {
                Some(last) if last.0 == row && last.1 == col => {
                    last.2 += re;
                    last.3 += im;
                }
                _ => merged.push((row, col, re, im)),
            }
        }
        // `merged` is sorted by (row, col), so the CSR arrays are the
        // merged tuples in order plus a prefix-summed per-row count.
        let mut cross_col_ptr = vec![0u32; internal + 1];
        for &(row, _, _, _) in &merged {
            cross_col_ptr[row as usize + 1] += 1;
        }
        for i in 1..=internal {
            cross_col_ptr[i] += cross_col_ptr[i - 1];
        }
        let cross_row_idx: Vec<u32> = merged.iter().map(|&(_, col, _, _)| col).collect();
        let cross_re: Vec<f64> = merged.iter().map(|&(_, _, re, _)| re).collect();
        let cross_im: Vec<f64> = merged.iter().map(|&(_, _, _, im)| im).collect();
        debug_assert_eq!(cross_col_ptr.len(), internal + 1);

        let block_nnz: Vec<usize> = layers.iter().map(|l| l.row_idx.len()).collect();
        eprintln!(
            "two-level: CSC complete: block nnz {block_nnz:?}, cross nnz {} (widest column {}, wall {:.3} s, hwm {} MiB)",
            cross_row_idx.len(),
            chunks.iter().map(|c| c.peak_col).max().unwrap_or(0),
            wall.elapsed().as_secs_f64(),
            hwm_mib(),
        );

        // Symbolic LU per layer (COLAMD inside faer).
        for block in layers.iter_mut() {
            if block.n == 0 || block.row_idx.is_empty() {
                if block.n > 0 {
                    eprintln!("two-level: warning: empty block skipped (identity)");
                }
                continue;
            }
            let sym = faer::sparse::SymbolicSparseColMat::new_checked(
                block.n,
                block.n,
                block.col_ptr.clone(),
                None,
                block.row_idx.clone(),
            );
            block.symbolic = SymbolicLu::try_new(sym.as_ref()).ok();
            if block.symbolic.is_none() {
                eprintln!(
                    "two-level: warning: symbolic LU build failed for a {}-row block; identity fallback",
                    block.n
                );
            }
        }
        Ok(Self {
            n: internal,
            layer_count,
            layer_bounds,
            perm,
            layers,
            cross_col_ptr,
            cross_row_idx,
            cross_re,
            cross_im,
        })
    }

    /// Factors `S(omega)` per layer; failures degrade that block to the
    /// identity (one-shot warned at apply, matching the global path).
    pub(crate) fn factor<T>(&self, omega: f64) -> TwoLevelFactors<T>
    where
        T: FieldParts + FaerComplexField + Copy,
    {
        let wall = std::time::Instant::now();
        let mut lus: Vec<Option<Lu<u32, T>>> = Vec::with_capacity(self.layers.len());
        for block in &self.layers {
            let factor = match (&block.symbolic, block.n > 0) {
                (Some(symbolic), true) => {
                    let values: Vec<T> = block
                        .re_part
                        .iter()
                        .zip(&block.im_part)
                        .map(|(&re, &im)| T::of(re, omega * im))
                        .collect();
                    let sym = faer::sparse::SymbolicSparseColMat::new_checked(
                        block.n,
                        block.n,
                        block.col_ptr.clone(),
                        None,
                        block.row_idx.clone(),
                    );
                    let mat = faer::sparse::SparseColMatRef::<u32, T>::new(sym.as_ref(), &values);
                    Lu::try_new_with_symbolic(symbolic.clone(), mat).ok()
                }
                _ => None,
            };
            lus.push(factor);
        }
        let cross_values: Vec<T> = self
            .cross_re
            .iter()
            .zip(&self.cross_im)
            .map(|(&re, &im)| T::of(re, omega * im))
            .collect();
        let failed = lus.iter().filter(|f| f.is_none()).count();
        eprintln!(
            "two-level: factored {} layers ({} identity fallbacks), perm n = {}, wall {:.3} s, hwm {} MiB",
            self.layer_count,
            failed,
            self.n,
            wall.elapsed().as_secs_f64(),
            hwm_mib(),
        );
        TwoLevelFactors { lus, cross_values }
    }

    /// Per-solve apply: permute in, `2 x layer_count` block solves with
    /// cross corrections (forward ascending, then recomputing
    /// descending), permute out.
    pub(crate) fn apply<T>(&self, factors: &TwoLevelFactors<T>, v: &[T]) -> Vec<T>
    where
        T: FaerComplexField + Copy,
    {
        if self.n == 0 {
            return v.to_vec();
        }
        // Permute in: position p holds row perm[p]'s value.
        let mut x: Vec<T> = self.perm.iter().map(|&r| v[r as usize]).collect();
        for layer in 0..self.layer_count {
            self.solve_layer(layer, factors, &mut x);
        }
        for layer in (0..self.layer_count).rev() {
            self.solve_layer(layer, factors, &mut x);
        }
        // Permute out.
        let mut out = v.to_vec();
        for (p, &r) in self.perm.iter().enumerate() {
            out[r as usize] = x[p].clone();
        }
        out
    }

    /// One block solve: cross corrections against the CURRENT x, then
    /// the layer's LU (or the identity fallback when its factor is
    /// `None`). Used by both the forward and the backward sweep.
    fn solve_layer<T>(&self, layer: usize, factors: &TwoLevelFactors<T>, x: &mut [T])
    where
        T: FaerComplexField + Copy,
    {
        let (start, end) = self.layer_bounds[layer];
        if start == end {
            return;
        }
        let mut rhs: Vec<T> = x[start..end].to_vec();
        // Cross correction, row by row: rows [start, end) hold CSR
        // ranges [col_ptr[r], col_ptr[r+1]).
        for r in start..end {
            let local = r - start;
            let lo = self.cross_col_ptr[r] as usize;
            let hi = self.cross_col_ptr[r + 1] as usize;
            for k in lo..hi {
                let col = self.cross_row_idx[k] as usize;
                let correction = factors.cross_values[k].clone() * x[col].clone();
                rhs[local] = rhs[local].clone() - correction;
            }
        }
        match factors.lus.get(layer).and_then(|f| f.as_ref()) {
            Some(lu) => {
                let n = rhs.len();
                let mut mat =
                    faer::MatMut::<T, usize, usize>::from_column_major_slice_mut(&mut rhs, n, 1);
                lu.solve_in_place_with_conj(faer::Conj::No, mat.as_mut());
            }
            None => {
                if !FALLBACK_SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "two-level: warning: failed block acting as identity rows (fallback)"
                    );
                }
            }
        }
        x[start..end].copy_from_slice(&rhs);
    }
}

fn empty_layer() -> LayerCsc {
    LayerCsc {
        n: 0,
        col_ptr: vec![0],
        row_idx: Vec::new(),
        re_part: Vec::new(),
        im_part: Vec::new(),
        symbolic: None,
    }
}
