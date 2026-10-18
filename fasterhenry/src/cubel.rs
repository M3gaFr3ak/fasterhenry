//! CUBEL-class mesh-block preconditioner prototype.
//!
//! The FastHenry2 `newPrecond_cube` pattern, ported to the fork's loop-basis
//! system: a spatial cube partition of the filaments induces a partition of
//! the *mesh rows*, and the near-field restriction
//! `S = M (R + iω L_near) Mᵀ` of the exact branch impedance to pairs inside
//! the pFFT near-field radius is assembled in mesh-row space. The
//! preconditioner step is a triangular solve against `S`'s block structure.
//!
//! This module is the *scout* shape: per-cube dense factorisations with a
//! block-diagonal apply (one solve per cube, no cross-cube pivots), built
//! fresh at every frequency from a ω-independent pattern. It answers the
//! semantic question — does near-block structure kill the √n GMRES crawl at
//! the fork's tol 1e-10 — before the global one-matrix sparse factorisation
//! (the D-4 letter, `faer` `SupernodalLu`) is wired behind the same knob.
//!
//! Shared machinery (near-pair enumeration, exact kernel, loop basis) is
//! the operator's own: the preconditioner only ever changes the iteration
//! count, never the answer.
//!
//! Knob: `FASTERHENRY_PRECOND=cubel` selects this over the default Jacobi
//! diagonal; `=none` is the identity (the unpreconditioned floor).

use faer::linalg::solvers::SolveCore;
use nalgebra::{ComplexField, Dyn, OMatrix, OVector, LU};
use num_complex::Complex;
use rayon::prelude::*;

use crate::filament::Filament;
use crate::inductance::{mutual_inductance, self_inductance, KernelError};
use crate::mesh::MeshMatrix;
use crate::pfft::near::near_pairs;

/// Matches `precorrect`'s orthogonality rule so the preconditioner's pair
/// values agree with the operator's near field (never the answer, just the
/// subtraction of two almost-equal quantities — but agreement costs nothing
/// and keeps the diff story clean).
const ORTHOGONAL_COS: f64 = 1e-14;

/// FH2's `AVER_MAT_MAX` pressure constant (induct.h): the mesh-row block
/// size the cube partition targets.
pub(crate) const AVER_ROWS: usize = 110;

/// One-shot flag so a fallback print appears once per process, not once
/// per application.
static FALLBACK_SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// (re, im) constructor over the two instantiations the schur solve uses:/// `f64` (only ever combined with ω = 0, so the imaginary part is kept but
/// unused) and `Complex<f64>` (ω > 0). `nalgebra`'s `ComplexField` has no
/// generic complex-from-parts constructor, so the list is explicit.
pub(crate) trait FieldParts: Copy {
    fn of(re: f64, im: f64) -> Self;
}

impl FieldParts for f64 {
    fn of(re: f64, _im: f64) -> f64 {
        re
    }
}

impl FieldParts for Complex<f64> {
    fn of(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }
}

/// ω-independent part of the preconditioner: the cube partition of mesh
/// rows, the per-branch exact near-pair inductances, and the branch→row
/// incidence the block assembly gathers through.
#[derive(Debug)]
pub(crate) struct CubelPattern {
    /// Per internal mesh row: its cube id.
    row_cube: Vec<u32>,
    /// Per cube: the internal rows belonging to it (ascending).
    cube_rows: Vec<Vec<u32>>,
    /// Per branch: `(partner, L_ab)` for the near-field partners with
    /// `partner > branch` (upper triangle; the block fill mirrors every
    /// entry to its transpose, so a bidirectional list would double-count).
    partners: Vec<Vec<(u32, f64)>>,
    /// Per branch: every internal row that contains it.
    branch_rows: Vec<Vec<u32>>,
    /// Per branch: its own exact partial inductance `L_bb`.
    self_l: Vec<f64>,
}

/// Tuning of the cube partition.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CubelParams {
    /// Target mesh rows per cube.
    pub(crate) max_block_rows: usize,
    /// Near-field pairing radius in metres (the operator's own `k·h`).
    pub(crate) near_radius: f64,
}

impl CubelPattern {
    /// Builds the ω-independent pattern from the assembled system pieces.
    pub(crate) fn build(
        mesh: &MeshMatrix,
        filaments: &[Filament],
        params: CubelParams,
    ) -> Result<Self, KernelError> {
        let internal = mesh.internal_count();
        let branches = filaments.len();
        if internal == 0 {
            return Ok(Self {
                row_cube: Vec::new(),
                cube_rows: Vec::new(),
                partners: Vec::new(),
                branch_rows: Vec::new(),
                self_l: Vec::new(),
            });
        }
        eprintln!(
            "cubel: pattern build: {internal} mesh rows, {branches} branches, radius {:.3e} m",
            params.near_radius,
        );

        // Near-field partner lists, upper triangle, self excluded (the
        // shared-branch gather below handles the self pair, with its
        // resistance). Same pair enumeration as `precorrect`, minus the
        // grid part.
        let pairs = near_pairs(filaments, params.near_radius);
        let mut partners: Vec<Vec<(u32, f64)>> = vec![Vec::new(); branches];
        let self_l: Vec<f64> = filaments.iter().map(self_inductance).collect();
        for a in 0..branches {
            let direction = filaments[a].direction();
            for &j in &pairs[a] {
                let j = j as usize;
                if j <= a {
                    continue;
                }
                if direction.dot(&filaments[j].direction()).abs() < ORTHOGONAL_COS {
                    continue;
                }
                let m = mutual_inductance(&filaments[a], &filaments[j])?;
                partners[a].push((j as u32, m));
            }
        }
        for list in &mut partners {
            list.sort_by_key(|&(p, _)| p);
        }

        // Branch → internal rows incidence.
        let mut branch_rows: Vec<Vec<u32>> = vec![Vec::new(); branches];
        for row in 0..internal {
            for &(b, _) in mesh.row(row) {
                branch_rows[b].push(row as u32);
            }
        }

        // The row's cube is the cube of its longest filament: a mesh row is
        // the reduced node of a parallel bundle, and the longest member
        // carries the row's geometric identity.
        let longest_branch: Vec<u32> = (0..internal)
            .map(|row| {
                mesh.row(row)
                    .iter()
                    .copied()
                    .max_by(|&(a, _), &(b, _)| {
                        let (la, lb) = (filaments[a].length(), filaments[b].length());
                        la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(b, _)| b as u32)
                    .unwrap_or(0)
            })
            .collect();

        // Cube partition: two passes of one sizing rule - partition,
        // measure the worst row block, rescale once, re-partition.
        let centres: Vec<_> = filaments.iter().map(|f| f.center()).collect();
        let rows_per_branch = (internal as f64 / branches as f64).max(1.0 / 1024.0);
        let target_fils = (params.max_block_rows as f64 / rows_per_branch).max(1.0);
        let mut edge = initial_edge(&centres, target_fils);
        let mut state = partition_state(&centres, edge, &longest_branch, internal);
        if state.worst > params.max_block_rows {
            edge *= (state.worst as f64 / params.max_block_rows as f64).sqrt();
            state = partition_state(&centres, edge, &longest_branch, internal);
        }
        let n_cubes = state
            .row_cube
            .iter()
            .copied()
            .max()
            .map_or(0, |c| c as usize + 1);
        let mut cube_rows: Vec<Vec<u32>> = vec![Vec::new(); n_cubes];
        for row in 0..internal {
            cube_rows[state.row_cube[row] as usize].push(row as u32);
        }
        eprintln!(
            "cubel: {} cubes, edge {:.4e} m, worst rows/cube {}",
            n_cubes,
            edge,
            cube_rows.iter().map(Vec::len).max().unwrap_or(0),
        );

        Ok(Self {
            row_cube: state.row_cube,
            cube_rows,
            partners,
            branch_rows,
            self_l,
        })
    }

    /// Assembles and factors the per-cube blocks of
    /// `M (R + iω L_near) Mᵀ` at this frequency. Call with `ω = 0` under
    /// `f64` only (the real instantiation cannot carry the imaginary part).
    pub(crate) fn factor_blocks<T>(
        &self,
        mesh: &MeshMatrix,
        resistances: &[f64],
        omega: f64,
    ) -> Vec<LU<T, Dyn, Dyn>>
    where
        T: FieldParts + ComplexField<RealField = f64> + Copy + Send + Sync,
    {
        let wall = std::time::Instant::now();
        let dim_sum = self
            .cube_rows
            .iter()
            .map(|r| r.len() * r.len())
            .sum::<usize>();
        let factors: Vec<LU<T, Dyn, Dyn>> = self
            .assemble_blocks::<T>(mesh, resistances, omega)
            .into_par_iter()
            .map(|mut block| {
                // Tiny ridge so a locally disconnected Laplacian piece does
                // not produce a structurally zero pivot; failure here can
                // only be as bad as the Jacobi row it stands in for.
                let big = block
                    .diagonal()
                    .iter()
                    .map(|&v| v.abs())
                    .fold(0.0, f64::max);
                for i in 0..block.nrows() {
                    *block.index_mut((i, i)) += T::from_real(big * 1e-9);
                }
                LU::new(block)
            })
            .collect();
        eprintln!(
            "cubel: factored {} blocks (worst dim {}, pattern dim-sum {}), wall {:.3} s",
            factors.len(),
            self.cube_rows.iter().map(Vec::len).max().unwrap_or(0),
            dim_sum,
            wall.elapsed().as_secs_f64(),
        );
        factors
    }

    /// Per-cube block-diagonal apply: gather, cube-local triangular solve,
    /// scatter. A cube that does not solve falls back to the identity on
    /// its rows (never worse than the 1.0 diagonal the `none` knob gives).
    pub(crate) fn apply_blocks<T>(&self, factors: &[LU<T, Dyn, Dyn>], v: &[T]) -> Vec<T>
    where
        T: ComplexField<RealField = f64> + Copy + Send + Sync,
    {
        let solved: Vec<Option<Vec<T>>> = self
            .cube_rows
            .par_iter()
            .zip(factors)
            .map(|(rows, factor)| {
                if rows.is_empty() {
                    return None;
                }
                let b = OVector::<T, Dyn>::from_iterator(
                    rows.len(),
                    rows.iter().map(|&r| v[r as usize]),
                );
                factor.solve(&b).map(|x| x.iter().copied().collect())
            })
            .collect();
        // A failed cube solve silently degrades the measurement: say so
        // once per process (the apply itself runs every GMRES iteration,
        // so a per-apply print would drown the residual stream).
        if solved.iter().any(Option::is_none) {
            if !FALLBACK_SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("cubel: warning: block fallback engaged (identity rows on failed cubes)");
            }
        }
        let mut out = v.to_vec(); // fallback: identity rows succeed.
        for (rows, values) in self.cube_rows.iter().zip(solved) {
            if let Some(values) = values {
                for (k, &r) in rows.iter().enumerate() {
                    out[r as usize] = values[k];
                }
            }
        }
        out
    }

    /// The dense per-cube blocks of `M (R + iω L_near) Mᵀ`. Cross-cube
    /// couplings are dropped by construction (block-diagonal scout); the
    /// shared-branch and near-partner gathers below carry every coupling
    /// between rows of one cube, including long chains sharing a branch
    /// metres apart.
    fn assemble_blocks<T>(
        &self,
        mesh: &MeshMatrix,
        resistances: &[f64],
        omega: f64,
    ) -> Vec<OMatrix<T, Dyn, Dyn>>
    where
        T: FieldParts + ComplexField<RealField = f64> + Copy + Send + Sync,
    {
        self.cube_rows
            .par_iter()
            .enumerate()
            .map(|(_cube, rows)| {
                let dim = rows.len();
                let mut block = OMatrix::<T, Dyn, Dyn>::zeros(dim, dim);
                // Local index of every row in this cube.
                let mut index = vec![u32::MAX; self.row_cube.len()];
                for (k, &r) in rows.iter().enumerate() {
                    index[r as usize] = k as u32;
                }
                for (li, &row) in rows.iter().enumerate() {
                    for &(a, sign_a) in mesh.row(row as usize) {
                        // Shared branch: `R_a + iω L_aa` couples every
                        // internal row containing `a`, including this one
                        // (that is the block diagonal, sign_a² = 1).
                        let self_l = self.self_l[a];
                        let za = T::of(*resistances.get(a).unwrap_or(&0.0), omega * self_l);
                        for &j in &self.branch_rows[a] {
                            let lj = index[j as usize];
                            // One-sided add (j >= row); the transpose comes
                            // from row j's own pass through the loop.
                            if lj != u32::MAX && j >= row as u32 {
                                let sign_j = sign_of(mesh.row(j as usize), a);
                                let w = T::from_real(sign_a * sign_j) * za;
                                block[(li, lj as usize)] += w;
                                if li != lj as usize {
                                    block[(lj as usize, li)] += w;
                                }
                            }
                        }
                        // Near partners: exact `iω L_ab`, resistance-free
                        // (only a shared branch carries one).
                        for &(b, l_ab) in &self.partners[a] {
                            let zb = T::of(0.0, omega * l_ab);
                            for &j in &self.branch_rows[b as usize] {
                                let lj = index[j as usize];
                                if lj == u32::MAX {
                                    continue;
                                }
                                let sign_j = sign_of(mesh.row(j as usize), b as usize);
                                let w = T::from_real(sign_a * sign_j) * zb;
                                block[(li, lj as usize)] += w;
                                if li != lj as usize {
                                    block[(lj as usize, li)] += w;
                                }
                            }
                        }
                    }
                }
                block
            })
            .collect()
    }
}

/// Sign the mesh row carries for branch `b` (`0` when absent).
fn sign_of(row: &[(usize, f64)], b: usize) -> f64 {
    row.iter()
        .find(|&&(bb, _)| bb == b)
        .map(|&(_, s)| s)
        .unwrap_or(0.0)
}

/// Metre edge of the initial cube partition: area of the two dominant
/// bounding-box axes times the per-cube filament budget.
fn initial_edge(centres: &[nalgebra::Vector3<f64>], target_fils: f64) -> f64 {
    let lo = centres.iter().fold([f64::MAX; 3], |acc, c| {
        [acc[0].min(c.x), acc[1].min(c.y), acc[2].min(c.z)]
    });
    let hi = centres.iter().fold([f64::MIN; 3], |acc, c| {
        [acc[0].max(c.x), acc[1].max(c.y), acc[2].max(c.z)]
    });
    let mut ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    ext.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let area = ext[0] * ext[1];
    if area <= 0.0 {
        return 1.0;
    }
    (target_fils * area / centres.len() as f64).sqrt()
}

/// One sizing pass: branch cubes from the centres, rows via the longest
/// member, plus the measured worst row block.
struct PartitionState {
    row_cube: Vec<u32>,
    worst: usize,
}

fn partition_state(
    centres: &[nalgebra::Vector3<f64>],
    edge: f64,
    longest_branch: &[u32],
    internal: usize,
) -> PartitionState {
    let lo = centres.iter().fold([f64::MAX; 3], |acc, c| {
        [acc[0].min(c.x), acc[1].min(c.y), acc[2].min(c.z)]
    });
    let hi = centres.iter().fold([f64::MIN; 3], |acc, c| {
        [acc[0].max(c.x), acc[1].max(c.y), acc[2].max(c.z)]
    });
    let dims: [i64; 3] = [0, 1, 2].map(|k| ((hi[k] - lo[k]) / edge).ceil().max(1.0) as i64);
    let nx = dims[0] as u32;
    let ny = dims[1] as u32;
    let cube_of: Vec<u32> = centres
        .iter()
        .map(|c| {
            let ix = (((c.x - lo[0]) / edge).floor() as i64).clamp(0, dims[0] - 1) as u32;
            let iy = (((c.y - lo[1]) / edge).floor() as i64).clamp(0, dims[1] - 1) as u32;
            let iz = (((c.z - lo[2]) / edge).floor() as i64).clamp(0, dims[2] - 1) as u32;
            ix + iy * nx + iz * nx * ny
        })
        .collect();
    let mut row_cube = vec![0u32; internal];
    let mut counts: Vec<usize> = Vec::new();
    for (row, &branch) in longest_branch.iter().enumerate() {
        let cube = cube_of[branch as usize];
        row_cube[row] = cube;
        let c = cube as usize;
        if c >= counts.len() {
            counts.resize(c + 1, 0);
        }
        counts[c] += 1;
    }
    PartitionState {
        row_cube,
        worst: counts.iter().copied().max().unwrap_or(0),
    }
}

/// D-4's letter, built on the scout's gather: *every* near pair
/// (cross-cube included) plus every shared-branch term as ONE sparse
/// mesh-space matrix `S(omega) = M (R + i omega L_near) M^T`, factorised
/// by faer's supernodal LU with its built-in COLAMD fill-reducing
/// ordering. The symbolic factorization is built once from the
/// omega-independent pattern; each frequency refactors the same pattern
/// with scaled values.
///
/// Driver for the code below: the AC band showed the block-diagonal
/// scout stall (grid-20k AC stuck at 3.9e-2 residual), because the
/// far-field pFFT inductance the operator carries is absent from any
/// block; the global near-pattern restores the operator's dominant part
/// to the preconditioner itself.
pub(crate) struct CubelGlobal {
    /// Dimension: the internal mesh rows.
    n: usize,
    /// CSC pattern (column-major, ascending rows per column, no
    /// duplicates) and the two omega-halves of the values.
    col_ptr: Vec<u32>,
    row_idx: Vec<u32>,
    re_part: Vec<f64>,
    im_part: Vec<f64>,
    /// Symbolic LU (COLAMD order included), built once. `None` after a
    /// failed build: the apply then hands back the identity, one-shot
    /// warned.
    symbolic: Option<faer::sparse::linalg::solvers::SymbolicLu<u32>>,
}

/// High-level sparse-solver re-exports.
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::traits::ComplexField as FaerComplexField;

/// Pattern-side data shared with the block-diagonal builder: exact
/// near-pair mutuals (non-orthogonal upper triangle), self terms, and
/// branch incidence (which internal rows contain which branch).
struct NearPairData {
    partners: Vec<Vec<(u32, f64)>>,
    self_l: Vec<f64>,
    branch_rows: Vec<Vec<u32>>,
}

fn near_pair_data(
    mesh: &MeshMatrix,
    filaments: &[Filament],
    internal: usize,
    near_radius: f64,
) -> Result<NearPairData, KernelError> {
    let branches = filaments.len();
    let pairs = near_pairs(filaments, near_radius);
    let mut partners: Vec<Vec<(u32, f64)>> = vec![Vec::new(); branches];
    for a in 0..branches {
        let direction = filaments[a].direction();
        for &j in &pairs[a] {
            let j = j as usize;
            if j <= a {
                continue;
            }
            if direction.dot(&filaments[j].direction()).abs() < ORTHOGONAL_COS {
                continue;
            }
            let m = mutual_inductance(&filaments[a], &filaments[j])?;
            partners[a].push((j as u32, m));
        }
    }
    for list in &mut partners {
        list.sort_by_key(|&(p, _)| p);
    }
    let self_l: Vec<f64> = filaments.iter().map(self_inductance).collect();
    let mut branch_rows: Vec<Vec<u32>> = vec![Vec::new(); branches];
    for row in 0..internal {
        for &(b, _) in mesh.row(row) {
            branch_rows[b].push(row as u32);
        }
    }
    Ok(NearPairData {
        partners,
        self_l,
        branch_rows,
    })
}

impl CubelGlobal {
    /// Dimension accessor for the apply site.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Builds the pattern and the once-only symbolic factorization.
    /// Contributions are gathered without mirroring: a branch pair (a, b)
    /// keeps `b > a` (upper triangle), so a row-pair whose only
    /// incidence runs (b in row i, a in row j) still appears, from row
    /// j's own scan, as the transpose entry. Symmetric values emerge
    /// from the natural incidence sums; nothing is manually mirrored.
    pub(crate) fn build(
        mesh: &MeshMatrix,
        filaments: &[Filament],
        resistances: &[f64],
        near_radius: f64,
    ) -> Result<Self, KernelError> {
        let internal = mesh.internal_count();
        if internal == 0 {
            // Empty systems never reach an apply, but must build.
            return Ok(Self {
                n: 0,
                col_ptr: vec![0],
                row_idx: Vec::new(),
                re_part: Vec::new(),
                im_part: Vec::new(),
                symbolic: None,
            });
        }
        let wall = std::time::Instant::now();
        let pattern = near_pair_data(mesh, filaments, internal, near_radius)?;
        eprintln!(
            "cubel-global: pattern: {internal} rows, {} branches, radius {:.3e} m",
            filaments.len(),
            near_radius,
        );

        // Column-keyed contributions: (col, row, re, im).
        let mut coo: Vec<(u32, u32, f64, f64)> = Vec::new();
        for row in 0..internal {
            for &(a, sign_a) in mesh.row(row) {
                // Shared branch `a`: the resistance half (omega-free,
                // diagonal in branch space) and the self inductance; every
                // row containing `a` couples through it.
                let za_re = *resistances.get(a).unwrap_or(&0.0);
                let za_im = pattern.self_l[a];
                for &j in &pattern.branch_rows[a] {
                    let sign_j = sign_of(mesh.row(j as usize), a);
                    let prod = sign_a * sign_j;
                    coo.push((j as u32, row as u32, prod * za_re, prod * za_im));
                }
                // Near partners `b > a`: exact inductance, omega-free
                // values; the omega scale joins at factor time.
                for &(b, l_ab) in &pattern.partners[a] {
                    for &j in &pattern.branch_rows[b as usize] {
                        let sign_j = sign_of(mesh.row(j as usize), b as usize);
                        let prod = sign_a * sign_j;
                        coo.push((j as u32, row as u32, 0.0, prod * l_ab));
                    }
                }
            }
        }

        // Sort by (col, row), then merge duplicates by summation: the
        // row-column gather semantics of M Z M^T, just sorted once.
        coo.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        let mut col_ptr: Vec<u32> = vec![0; internal + 1];
        let mut row_idx: Vec<u32> = Vec::with_capacity(coo.len());
        let mut re_part: Vec<f64> = Vec::with_capacity(coo.len());
        let mut im_part: Vec<f64> = Vec::with_capacity(coo.len());
        let mut last = (u32::MAX, u32::MAX);
        let mut cur = 1u32;
        for &(col, row, re, im) in &coo {
            if (col, row) == last {
                let n = row_idx.len() - 1;
                re_part[n] += re;
                im_part[n] += im;
                continue;
            }
            // A new populated column closes off every column before it:
            // their slot value is the current offset (start of column
            // `cur`, end of the previous content; empty columns get
            // start = end).
            while cur <= col {
                col_ptr[cur as usize] = row_idx.len() as u32;
                cur += 1;
            }
            row_idx.push(row);
            re_part.push(re);
            im_part.push(im);
            last = (col, row);
        }
        while cur < internal as u32 + 1 {
            col_ptr[cur as usize] = row_idx.len() as u32;
            cur += 1;
        }
        eprintln!(
            "cubel-global: gathered {} contributions -> CSC nnz {} (wall {:.3} s)",
            coo.len(),
            row_idx.len(),
            wall.elapsed().as_secs_f64(),
        );

        // Symbolic LU; COLAMD ordering happens inside faer.
        let sym = faer::sparse::SymbolicSparseColMat::new_checked(
            internal,
            internal,
            col_ptr.clone(),
            None,
            row_idx.clone(),
        );
        let symbolic = SymbolicLu::try_new(sym.as_ref()).ok();
        if symbolic.is_none() {
            eprintln!("cubel-global: warning: symbolic LU build failed; identity fallback");
        }
        eprintln!("cubel-global: symbolic LU ready (n = {internal})");

        Ok(Self {
            n: internal,
            col_ptr,
            row_idx,
            re_part,
            im_part,
            symbolic,
        })
    }

    /// Factors `S(omega)` at this frequency (same pattern as the symbolic
    /// build, so the fill update is a numeric-only pass). Failure falls
    /// back to `None`, and the apply then hands back the identity - the
    /// same floor the `none` knob gives - with a one-shot warning.
    pub(crate) fn factor<T>(&self, omega: f64) -> Option<Lu<u32, T>>
    where
        T: FieldParts + FaerComplexField + Copy,
    {
        let wall = std::time::Instant::now();
        // Note: the column-major slice below wants values ordered by
        // (col, row) ascending; the CSC arrays already are, so the values
        // vector is CSC values verbatim.
        let values: Vec<T> = self
            .re_part
            .iter()
            .zip(&self.im_part)
            .map(|(&re, &im)| T::of(re, omega * im))
            .collect();
        let sym = faer::sparse::SymbolicSparseColMat::new_checked(
            self.n,
            self.n,
            self.col_ptr.clone(),
            None,
            self.row_idx.clone(),
        );
        let Some(symbolic) = &self.symbolic else {
            return None;
        };
        let mat = faer::sparse::SparseColMatRef::<u32, T>::new(sym.as_ref(), &values);
        match Lu::try_new_with_symbolic(symbolic.clone(), mat) {
            Ok(factors) => {
                eprintln!(
                    "cubel-global: factored S(omega={:.3e}), wall {:.3} s",
                    omega,
                    wall.elapsed().as_secs_f64(),
                );
                Some(factors)
            }
            Err(_) => {
                if !FALLBACK_SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "cubel-global: warning: numeric factorization failed; identity fallback"
                    );
                }
                None
            }
        }
    }

    /// Per-solve apply: the LU's triangular on `v`; identity if the
    /// factor failed.
    pub(crate) fn apply_global<T>(factors: Option<&Lu<u32, T>>, n: usize, v: &[T]) -> Vec<T>
    where
        T: FaerComplexField + Copy,
    {
        match factors {
            Some(factors) => {
                let mut out = v.to_vec();
                let rhs =
                    faer::MatMut::<T, usize, usize>::from_column_major_slice_mut(&mut out, n, 1);
                factors.solve_in_place_with_conj(faer::Conj::No, rhs);
                out
            }
            None => v.to_vec(),
        }
    }
}
