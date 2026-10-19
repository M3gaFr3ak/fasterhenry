//! Near-field pairs and their precorrection.
//!
//! A pair of filaments is *near* when a lower bound on the distance between
//! their volumes (the separating-axis gap of
//! [`neumann::separation`](crate::inductance::neumann::separation)) is below
//! the near-field radius. For those pairs the grid's approximation of the
//! partial inductance is inaccurate, so the operator stores
//!
//! ```text
//! Cᵢⱼ = L̃ᵢⱼ(exact kernel) − Lᵢⱼ(grid)
//! ```
//!
//! and adds `Σⱼ Cᵢⱼ xⱼ` to the far-field result: the grid contribution of a
//! near pair cancels to rounding error and the exact value takes its place.

use nalgebra::Vector3;
use rayon::prelude::*;
use wide::f64x4;

use super::grid::{Grid, Projection};
use crate::filament::Filament;
use crate::inductance::neumann::{separation, Bar};
use crate::inductance::{evaluate, KernelError, MU0_OVER_4PI};

/// `|cos|` below which two filaments are orthogonal, so that both their
/// exact partial inductance and its grid approximation vanish. Matches the
/// kernel's own threshold.
const ORTHOGONAL_COS: f64 = 1e-14;

/// Axis-aligned bounding box of a filament's volume.
fn bounds(bar: &Bar) -> (Vector3<f64>, Vector3<f64>) {
    let mut reach = Vector3::zeros();
    for k in 0..3 {
        reach += bar.axes[k].abs() * bar.half[k];
    }
    (bar.centre - reach, bar.centre + reach)
}

/// For each filament `i`, the filaments `j ≥ i` (including `i` itself)
/// within `radius` of it, ascending.
pub(crate) fn near_pairs(filaments: &[Filament], radius: f64) -> Vec<Vec<u32>> {
    let n = filaments.len();
    if n == 0 {
        return Vec::new();
    }
    let bars: Vec<Bar> = filaments.iter().map(Bar::new).collect();
    let boxes: Vec<_> = bars.iter().map(bounds).collect();
    let mut lo = boxes[0].0;
    let mut hi = boxes[0].1;
    for (a, b) in &boxes {
        lo = lo.inf(a);
        hi = hi.sup(b);
    }
    // Uniform bins about the radius in size, coarsened if that would make
    // far more bins than filaments.
    let extent = hi - lo;
    let mut size = radius.max(extent.max() * 1e-9).max(f64::MIN_POSITIVE);
    let bin_dims = |size: f64| extent.map(|e| (e / size).floor() as usize + 1);
    while bin_dims(size).iter().product::<usize>() > 8 * n + 64 {
        size *= 1.5;
    }
    let dims = bin_dims(size);
    let cell = |x: f64, axis: usize| {
        (((x - lo[axis]) / size).floor().max(0.0) as usize).min(dims[axis] - 1)
    };
    let range = |a: &Vector3<f64>, b: &Vector3<f64>| {
        let first = [cell(a.x, 0), cell(a.y, 1), cell(a.z, 2)];
        let last = [cell(b.x, 0), cell(b.y, 1), cell(b.z, 2)];
        (first, last)
    };
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); dims.iter().product()];
    for (index, (a, b)) in boxes.iter().enumerate() {
        let (first, last) = range(a, b);
        for i in first[0]..=last[0] {
            for j in first[1]..=last[1] {
                for k in first[2]..=last[2] {
                    bins[(i * dims[1] + j) * dims[2] + k].push(index as u32);
                }
            }
        }
    }
    let grow = Vector3::repeat(radius);
    (0..n)
        .into_par_iter()
        .map(|index| {
            let (a, b) = &boxes[index];
            let (first, last) = range(&(a - grow), &(b + grow));
            let mut found = Vec::new();
            for i in first[0]..=last[0] {
                for j in first[1]..=last[1] {
                    for k in first[2]..=last[2] {
                        found.extend(
                            bins[(i * dims[1] + j) * dims[2] + k]
                                .iter()
                                .filter(|&&other| other as usize >= index),
                        );
                    }
                }
            }
            found.sort_unstable();
            found.dedup();
            found.retain(|&other| {
                let other = other as usize;
                other == index || separation(&bars[index], &bars[other]) < radius
            });
            found
        })
        .collect()
}

/// The grid kernel `1/|r_g − r_h|` for an integer offset, with the
/// singular self term set to zero (it only ever enters near pairs, whose
/// grid contribution the precorrection removes).
pub(crate) fn grid_kernel(offset: [i64; 3], spacing: f64) -> f64 {
    let r2 = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]) as f64;
    if r2 == 0.0 {
        0.0
    } else {
        1.0 / (spacing * r2.sqrt())
    }
}

/// Near-field correction matrix in compressed sparse rows, both triangles.
#[derive(Clone, Debug, Default)]
pub(crate) struct NearField {
    pub(crate) offsets: Vec<usize>,
    pub(crate) columns: Vec<u32>,
    pub(crate) values: Vec<f64>,
}

/// [`grid_kernel`] tabulated over the integer offsets `|dₐ| ≤ reachₐ`,
/// `z` fastest, so that a run of offsets along `z` is a contiguous slice.
struct KernelTable {
    reach: [usize; 3],
    dims: [usize; 3],
    values: Vec<f64>,
}

impl KernelTable {
    fn new(reach: [usize; 3], spacing: f64) -> Self {
        let dims = reach.map(|r| 2 * r + 1);
        let mut values = vec![0.0; dims.iter().product()];
        values
            .par_chunks_mut(dims[1] * dims[2])
            .enumerate()
            .for_each(|(x, plane)| {
                let dx = x as i64 - reach[0] as i64;
                for (rest, value) in plane.iter_mut().enumerate() {
                    let dy = (rest / dims[2]) as i64 - reach[1] as i64;
                    let dz = (rest % dims[2]) as i64 - reach[2] as i64;
                    *value = grid_kernel([dx, dy, dz], spacing);
                }
            });
        Self {
            reach,
            dims,
            values,
        }
    }

    /// Index of offset `d`, which must be within reach.
    fn index(&self, d: [i64; 3]) -> usize {
        let at = |a: usize| (d[a] + self.reach[a] as i64) as usize;
        (at(0) * self.dims[1] + at(1)) * self.dims[2] + at(2)
    }
}

/// `row[k] += wg * kernel[k]` for every `k`, four lanes at a time (`wg4` is
/// `wg` broadcast, passed in rather than recomputed per call since callers
/// invoke this once per box row -- `by` times per source point).
///
/// # Panics
///
/// If `row.len() != kernel.len()`.
#[inline]
fn axpy(row: &mut [f64], kernel: &[f64], wg: f64, wg4: f64x4) {
    assert_eq!(row.len(), kernel.len());
    let mut k_chunks = kernel.chunks_exact(4);
    let mut r_chunks = row.chunks_exact_mut(4);
    while let (Some(kc), Some(rc)) = (k_chunks.next(), r_chunks.next()) {
        let k4 = f64x4::new(kc.try_into().unwrap());
        let r4 = f64x4::new((&*rc).try_into().unwrap());
        rc.copy_from_slice(&wg4.mul_add(k4, r4).to_array());
    }
    for (value, &k) in r_chunks
        .into_remainder()
        .iter_mut()
        .zip(k_chunks.remainder())
    {
        *value += wg * k;
    }
}

/// Grid coordinates of a projection's entries, with their weights.
fn points(grid: &Grid, projection: &Projection) -> Vec<([i64; 3], f64)> {
    projection
        .entries
        .iter()
        .map(|&(g, w)| (grid.point(g as usize).map(|c| c as i64), w))
        .collect()
}

/// Inclusive grid box spanned by the projections of `partners`.
fn partner_box(partners: &[usize], projections: &[Projection]) -> ([usize; 3], [usize; 3]) {
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for &j in partners {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(projections[j].lo[axis]);
            hi[axis] = hi[axis].max(projections[j].hi[axis]);
        }
    }
    (lo, hi)
}

/// Computes the precorrected near-field matrix.
pub(crate) fn precorrect(
    filaments: &[Filament],
    grid: &Grid,
    projections: &[Projection],
    near: &[Vec<u32>],
) -> Result<NearField, KernelError> {
    let n = filaments.len();
    // Partners j ≥ i that are not orthogonal to i: both their exact partial
    // inductance and its grid approximation vanish otherwise.
    let partners: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let direction = filaments[i].direction();
            near[i]
                .iter()
                .map(|&j| j as usize)
                .filter(|&j| direction.dot(&filaments[j].direction()).abs() >= ORTHOGONAL_COS)
                .collect()
        })
        .collect();
    // Every grid offset the precorrection evaluates lies between a point of
    // a filament's projection and a point of its partners' box.
    let reach = (0..n)
        .into_par_iter()
        .filter(|&i| !partners[i].is_empty())
        .map(|i| {
            let (lo, hi) = partner_box(&partners[i], projections);
            let own = &projections[i];
            [0, 1, 2].map(|a| (hi[a] - own.lo[a].min(hi[a])).max(own.hi[a] - lo[a].min(own.hi[a])))
        })
        .reduce(|| [0; 3], |a, b| [0, 1, 2].map(|k| a[k].max(b[k])));
    let table = KernelTable::new(reach, grid.spacing);

    // Progress: the precorrection kernel loop is the dominant set-up cost
    // (~75%); report every ~5% so a large build is not a silent hang.
    let done = std::sync::atomic::AtomicUsize::new(0);
    let step = (n / 20).max(1);
    let upper: Vec<Vec<(u32, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let r = row(i, filaments, grid, projections, &partners[i], &table);
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if d % step == 0 || d == n {
                let rss = crate::memgate::rss_bytes()
                    .map(|b| format!("{:.1} GB", b as f64 / 1e9))
                    .unwrap_or_else(|| "unreadable".to_string());
                eprintln!("pfft: precorrect {d}/{n} (rss {rss})");
            }
            r
        })
        .collect::<Result<_, _>>()?;
    // Mirror the upper triangle into full rows.
    let mut counts = vec![0usize; n];
    for (i, entries) in upper.iter().enumerate() {
        for &(j, _) in entries {
            counts[i] += 1;
            if j as usize != i {
                counts[j as usize] += 1;
            }
        }
    }
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0);
    for count in &counts {
        offsets.push(offsets.last().unwrap() + count);
    }
    let total = *offsets.last().unwrap();
    let mut columns = vec![0u32; total];
    let mut values = vec![0.0; total];
    let mut fill = offsets[..n].to_vec();
    for (i, entries) in upper.iter().enumerate() {
        for &(j, value) in entries {
            let j = j as usize;
            columns[fill[i]] = j as u32;
            values[fill[i]] = value;
            fill[i] += 1;
            if j != i {
                columns[fill[j]] = i as u32;
                values[fill[j]] = value;
                fill[j] += 1;
            }
        }
    }
    Ok(NearField {
        offsets,
        columns,
        values,
    })
}

/// Correction entries `(j, Cᵢⱼ)` of row `i` for its near partners `j ≥ i`.
fn row(
    i: usize,
    filaments: &[Filament],
    grid: &Grid,
    projections: &[Projection],
    partners: &[usize],
    table: &KernelTable,
) -> Result<Vec<(u32, f64)>, KernelError> {
    if partners.is_empty() {
        return Ok(Vec::new());
    }
    let fi = &filaments[i];
    let grid_values = grid_interactions(i, partners, grid, projections, table);
    partners
        .iter()
        .zip(grid_values)
        .map(|(&j, approx)| {
            let fj = &filaments[j];
            let exact = evaluate(fi, fj, true).map_err(|e| e.at(i, j))?.value;
            let cos = fi.direction().dot(&fj.direction());
            Ok((j as u32, exact - MU0_OVER_4PI * cos * approx))
        })
        .collect()
}

/// `Σ_g Σ_h W_ig W_jh K(g − h)` for every partner `j`, by whichever of two
/// schemes is cheaper: the direct double sum, or the potential
/// `ψ = K ⊛ Wᵢ` of `i`'s projection over the box spanned by the partners'
/// projections followed by one inner product per partner.
fn grid_interactions(
    i: usize,
    partners: &[usize],
    grid: &Grid,
    projections: &[Projection],
    table: &KernelTable,
) -> Vec<f64> {
    let source = points(grid, &projections[i]);
    let (lo, hi) = partner_box(partners, projections);
    let box_dims = [0, 1, 2].map(|a| hi[a] - lo[a] + 1);
    let box_len: usize = box_dims.iter().product();
    let partner_entries: usize = partners.iter().map(|&j| projections[j].entries.len()).sum();
    let direct_cost = source.len() * partner_entries;
    let box_cost = source.len() * box_len + partner_entries;

    if direct_cost <= box_cost {
        return partners
            .iter()
            .map(|&j| {
                points(grid, &projections[j])
                    .iter()
                    .map(|&(h, wh)| {
                        let potential: f64 = source
                            .iter()
                            .map(|&(g, wg)| {
                                wg * table.values
                                    [table.index([h[0] - g[0], h[1] - g[1], h[2] - g[2]])]
                            })
                            .sum();
                        wh * potential
                    })
                    .sum()
            })
            .collect();
    }

    // ψ over the box, one source point at a time: each (x, y) row of the box
    // takes a contiguous run of the table, a SIMD multiply-add (see `axpy`).
    // `table.index` is inlined and hoisted here rather than called per (x,
    // y): for a fixed source point and x, consecutive y advance the table
    // index by exactly `dims[2]` (issue #51 profiling found the naive
    // per-(x, y) `table.index` call plus a scalar accumulate loop cost about
    // as much as the FFT precorrection it belongs to; both parts of that are
    // addressed here).
    let [bx, by, bz] = box_dims;
    let (dims1, dims2) = (table.dims[1], table.dims[2]);
    let [r0, r1, r2] = table.reach.map(|r| r as i64);
    let mut psi = vec![0.0; box_len];
    for &(g, wg) in &source {
        let wg4 = f64x4::splat(wg);
        let at0_base = r0 - g[0];
        let at1_0 = (lo[1] as i64 - g[1] + r1) as usize;
        let at2 = (lo[2] as i64 - g[2] + r2) as usize;
        for x in 0..bx {
            let at0 = (lo[0] + x) as i64 + at0_base;
            let mut start = (at0 as usize * dims1 + at1_0) * dims2 + at2;
            for y in 0..by {
                let kernel = &table.values[start..start + bz];
                let row = &mut psi[(x * by + y) * bz..(x * by + y + 1) * bz];
                axpy(row, kernel, wg, wg4);
                start += dims2;
            }
        }
    }
    partners
        .iter()
        .map(|&j| {
            projections[j]
                .entries
                .iter()
                .map(|&(h, wh)| {
                    let p = grid.point(h as usize);
                    let local = ((p[0] - lo[0]) * by + (p[1] - lo[1])) * bz + (p[2] - lo[2]);
                    wh * psi[local]
                })
                .sum()
        })
        .collect()
}

/// Manual, phase-by-phase profiling of `PfftOperator::new`'s set-up cost
/// (issue #51). Not run by `cargo test`; the phases mirror `PfftOperator::new`
/// and `precorrect` exactly (rather than sampling with an external profiler)
/// so that the "grid precorrection" and "exact kernel" costs -- interleaved
/// row by row inside `precorrect` -- can be attributed separately. Run with:
///
/// ```text
/// cargo test --release -p fasterhenry --lib pfft::near::profile -- --ignored --nocapture
/// ```
///
/// `FASTERHENRY_PROFILE_N` overrides the filament count (default 10 000).
#[cfg(test)]
mod profile {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::filament::Filament;
    use crate::geometry::{Node, Segment};
    use crate::pfft::fft as fft_mod;
    use crate::pfft::grid as grid_mod;
    use crate::pfft::{auto_spacing, bounding_box, GridSpacing, PfftParams};

    const COPPER: f64 = 5.8e7;

    /// The same generator as `benches/pfft.rs::cloud` and
    /// `tests/pfft.rs::random_set(.., manhattan: true, ..)`: `n` random
    /// axis-aligned filaments at constant density (400 per 1 000 mm³), from a
    /// fixed sequence so every run sees the same geometry. Duplicated here
    /// because a private module can't share code with an integration test or
    /// bench crate.
    fn cloud(n: usize) -> Vec<Filament> {
        let side = 10e-3 * (n as f64 / 400.0).cbrt();
        let mut state = 11u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let a = [side * next(), side * next(), side * next()];
                let length = 1e-3 * (0.5 + next());
                let mut d = [0.0; 3];
                d[((next() * 3.0) as usize).min(2)] = if next() < 0.5 { -1.0 } else { 1.0 };
                let b = [0, 1, 2].map(|k| a[k] + length * d[k]);
                let width = 0.1e-3 + 0.2e-3 * next();
                let height = 0.05e-3 + 0.1e-3 * next();
                Filament::new(&Segment::new(
                    Node::from(a),
                    Node::from(b),
                    width,
                    height,
                    COPPER,
                ))
                .expect("valid segment")
            })
            .collect()
    }

    #[test]
    #[ignore]
    fn setup_phase_breakdown() {
        let n: usize = std::env::var("FASTERHENRY_PROFILE_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);
        let params = PfftParams::default();
        let order = params.interpolation_order;
        let filaments = cloud(n);

        let t = Instant::now();
        let (lo, hi) = bounding_box(&filaments);
        let spacing = match params.grid_spacing {
            GridSpacing::Fixed(h) => h,
            GridSpacing::Auto { cells_per_filament } => {
                auto_spacing(hi - lo, cells_per_filament * n as f64)
            }
        };
        let grid = Grid::new(lo, hi, spacing, order + 1);
        let bbox_time = t.elapsed();
        let radius = params.near_field_radius * spacing;

        let t = Instant::now();
        let projections = grid_mod::project_all(&grid, &filaments, radius);
        let projection_time = t.elapsed();

        let t = Instant::now();
        let pairs = near_pairs(&filaments, radius);
        let near_pair_time = t.elapsed();

        // From here on, mirrors `precorrect` step by step.
        let t = Instant::now();
        let partners: Vec<Vec<usize>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let direction = filaments[i].direction();
                pairs[i]
                    .iter()
                    .map(|&j| j as usize)
                    .filter(|&j| direction.dot(&filaments[j].direction()).abs() >= ORTHOGONAL_COS)
                    .collect()
            })
            .collect();
        let partner_filter_time = t.elapsed();

        let t = Instant::now();
        let reach = (0..n)
            .into_par_iter()
            .filter(|&i| !partners[i].is_empty())
            .map(|i| {
                let (lo, hi) = partner_box(&partners[i], &projections);
                let own = &projections[i];
                [0, 1, 2]
                    .map(|a| (hi[a] - own.lo[a].min(hi[a])).max(own.hi[a] - lo[a].min(own.hi[a])))
            })
            .reduce(|| [0; 3], |a, b| [0, 1, 2].map(|k| a[k].max(b[k])));
        let table = KernelTable::new(reach, grid.spacing);
        let reach_table_time = t.elapsed();

        // Size of `table` versus each row's own local box (its span, not the
        // global worst case `reach` is sized to): if rows typically use only
        // a small slice of `table`, the table itself stays cache-resident
        // (ruling out "table too big to cache" as the row loop's cost
        // driver -- it was checked and ruled out this way; see the module
        // documentation).
        let mut box_lens: Vec<usize> = (0..n)
            .into_par_iter()
            .filter(|&i| !partners[i].is_empty())
            .map(|i| {
                let (lo, hi) = partner_box(&partners[i], &projections);
                [0, 1, 2].map(|a| hi[a] - lo[a] + 1).iter().product()
            })
            .collect();
        box_lens.sort_unstable();
        let mean = box_lens.iter().sum::<usize>() as f64 / box_lens.len().max(1) as f64;
        eprintln!(
            "  table.dims {:?} ({} cells, {:.1} MB); row box_len: min {} p50 {} p90 {} max {} mean {mean:.0}",
            table.dims,
            table.values.len(),
            table.values.len() as f64 * 8.0 / 1e6,
            box_lens.first().copied().unwrap_or(0),
            box_lens.get(box_lens.len() / 2).copied().unwrap_or(0),
            box_lens.get(box_lens.len() * 9 / 10).copied().unwrap_or(0),
            box_lens.last().copied().unwrap_or(0),
        );

        // The row loop, with `grid_interactions` (grid precorrection) and
        // `evaluate` (exact near-pair kernel) timed separately per row and
        // summed as core-nanoseconds across the parallel rows -- a self-time
        // attribution, not wall time (rows run concurrently), but the right
        // quantity to compare the two against each other. `checksum` is
        // printed so neither call is optimized away.
        let t = Instant::now();
        let (grid_ns, kernel_ns, checksum) = (0..n)
            .into_par_iter()
            .filter(|&i| !partners[i].is_empty())
            .fold(
                || (0u64, 0u64, 0.0f64),
                |(g, k, sum), i| {
                    let tg = Instant::now();
                    let approx = grid_interactions(i, &partners[i], &grid, &projections, &table);
                    let g = g + tg.elapsed().as_nanos() as u64;

                    let fi = &filaments[i];
                    let tk = Instant::now();
                    let mut exact_sum = 0.0;
                    for &j in &partners[i] {
                        let fj = &filaments[j];
                        exact_sum += evaluate(fi, fj, true).expect("evaluable pair").value;
                    }
                    let k = k + tk.elapsed().as_nanos() as u64;

                    (g, k, sum + exact_sum + approx.iter().sum::<f64>())
                },
            )
            .reduce(|| (0, 0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let row_loop_wall_time = t.elapsed();

        let t = Instant::now();
        let h = grid.spacing;
        let convolver = fft_mod::Convolver::new(grid.dims, |d| grid_kernel(d, h));
        let fft_planning_time = t.elapsed();

        // Cross-check: the real, unmodified `precorrect` call, for comparison
        // against partner_filter + reach_table + row_loop above.
        let t = Instant::now();
        let near = precorrect(&filaments, &grid, &projections, &pairs).expect("precorrects");
        let real_precorrect_time = t.elapsed();

        let grid_core_time = Duration::from_nanos(grid_ns);
        let kernel_core_time = Duration::from_nanos(kernel_ns);
        eprintln!("pFFT set-up phase breakdown at n = {n} (default params):");
        eprintln!(
            "  grid dims {:?}, padded {:?}",
            grid.dims,
            convolver.padded_dims()
        );
        eprintln!(
            "  near entries {} ({:.1}/filament), projection entries {} ({:.1}/filament)",
            near.columns.len(),
            near.columns.len() as f64 / n as f64,
            projections.iter().map(|p| p.entries.len()).sum::<usize>(),
            projections.iter().map(|p| p.entries.len()).sum::<usize>() as f64 / n as f64,
        );
        eprintln!("  bounding box + grid init         {bbox_time:>10.2?}");
        eprintln!("  projection (grid::project_all)   {projection_time:>10.2?}");
        eprintln!("  near-pair search (near_pairs)     {near_pair_time:>10.2?}");
        eprintln!("  precorrection, of which:");
        eprintln!("    partner (orthogonality) filter  {partner_filter_time:>10.2?}");
        eprintln!("    reach + kernel table build       {reach_table_time:>10.2?}");
        eprintln!("    row loop, wall time              {row_loop_wall_time:>10.2?}");
        eprintln!("      grid precorrection, core-time   {grid_core_time:>10.2?}");
        eprintln!("      exact kernel eval, core-time     {kernel_core_time:>10.2?}");
        eprintln!("    (cross-check) real precorrect() {real_precorrect_time:>10.2?}");
        eprintln!("  FFT planning (Convolver::new)    {fft_planning_time:>10.2?}");
        let measured_total =
            bbox_time + projection_time + near_pair_time + real_precorrect_time + fft_planning_time;
        eprintln!("  measured total (excl. final CSR arrays in PfftOperator::new) {measured_total:>10.2?}");
        eprintln!("  checksum (ignore): {checksum:e}");
    }
}
