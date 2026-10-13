//! Batched evaluation of partial-inductance matrices.

use nalgebra::DMatrix;
use rayon::prelude::*;

use super::{evaluate, KernelError, Method, Mutual};
use crate::filament::Filament;

/// How a batch is executed. [`mutual_batch`] uses [`Execution::Parallel`];
/// the other modes exist so that the benchmark can separate the speed-up of
/// the SIMD inner quadrature from that of the thread pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Execution {
    /// One thread, one kernel evaluation at a time.
    Scalar,
    /// One thread, the inner quadrature in `wide::f64x4` lanes.
    Simd,
    /// SIMD lanes within each pair and `rayon` across the rows.
    #[default]
    Parallel,
}

/// Which filament pairs an assembly evaluates, and which it truncates to
/// zero.
///
/// Each filament carries a group index; a symmetric table says which pairs of
/// groups are coupled. A pair whose groups are not coupled is never handed to
/// a kernel — the saving is in evaluations skipped, not in entries zeroed
/// afterwards. Every group is always coupled to itself, so self and
/// intra-group terms always survive.
///
/// See [`crate::coupling`] for building one from a [`Geometry`] and for when
/// the truncation is physically safe.
///
/// [`Geometry`]: crate::geometry::Geometry
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairMask {
    /// Group index of each filament.
    group: Vec<usize>,
    /// `groups × groups`, row-major, symmetric, diagonal true.
    coupled: Vec<bool>,
    groups: usize,
}

impl PairMask {
    /// The mask that keeps every pair of `filaments` filaments — today's
    /// behaviour, and what [`partial_inductance_matrix`] does.
    pub fn all(filaments: usize) -> Self {
        Self {
            group: vec![0; filaments],
            coupled: vec![true],
            groups: 1,
        }
    }

    /// A mask from a per-filament group index and a predicate on group
    /// indices.
    ///
    /// The predicate is symmetrized (`a` and `b` are coupled if it accepts
    /// either order) and the diagonal is forced true, so a caller cannot
    /// accidentally truncate a filament against itself or its own group.
    /// Group indices at or above `groups`, and filament indices beyond
    /// `group`'s length, are treated as coupled: an incompletely specified
    /// mask loses accuracy, never silently drops a term it was never told
    /// about.
    pub fn from_groups(
        group: Vec<usize>,
        groups: usize,
        coupled: impl Fn(usize, usize) -> bool,
    ) -> Self {
        let mut table = vec![false; groups * groups];
        for a in 0..groups {
            for b in a..groups {
                let keep = a == b || coupled(a, b) || coupled(b, a);
                table[a * groups + b] = keep;
                table[b * groups + a] = keep;
            }
        }
        Self {
            group,
            coupled: table,
            groups,
        }
    }

    /// Number of filaments the mask describes.
    pub fn len(&self) -> usize {
        self.group.len()
    }

    /// Whether the mask describes no filaments at all.
    pub fn is_empty(&self) -> bool {
        self.group.is_empty()
    }

    /// Number of groups.
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Whether the partial inductance of filaments `i` and `j` is evaluated.
    pub fn keeps(&self, i: usize, j: usize) -> bool {
        match (self.group.get(i), self.group.get(j)) {
            (Some(&a), Some(&b)) => self
                .coupled
                .get(a * self.groups + b)
                .copied()
                .unwrap_or(true),
            _ => true,
        }
    }

    /// Whether every pair is kept, so that masking costs nothing.
    pub fn keeps_everything(&self) -> bool {
        self.coupled.iter().all(|&keep| keep)
    }

    /// Unordered pairs (including the `n` self terms) the mask keeps, out of
    /// the [`total_pairs`](Self::total_pairs) a full assembly evaluates.
    pub fn kept_pairs(&self) -> usize {
        // Counted as "every pair, less the cross-group pairs dropped": a
        // filament whose group index is out of range is coupled to
        // everything (see `keeps`) and so is never subtracted.
        let mut sizes = vec![0usize; self.groups];
        for &g in &self.group {
            if let Some(slot) = sizes.get_mut(g) {
                *slot += 1;
            }
        }
        let mut dropped = 0usize;
        for a in 0..self.groups {
            for b in (a + 1)..self.groups {
                if !self.coupled[a * self.groups + b] {
                    dropped += sizes[a] * sizes[b];
                }
            }
        }
        self.total_pairs() - dropped
    }

    /// Unordered pairs a full assembly of the same filaments evaluates,
    /// `n·(n+1)/2`.
    pub fn total_pairs(&self) -> usize {
        let n = self.group.len();
        n * (n + 1) / 2
    }
}

/// Partial-inductance values together with entries whose accuracy criterion
/// was not met. See [`Mutual::resolved`] for the reduced-accuracy regime.
#[derive(Clone, Debug, PartialEq)]
pub struct MutualBatch {
    /// Partial inductances in henries, with the same shape and values as the
    /// corresponding value-only batch function.
    pub values: DMatrix<f64>,
    /// Zero-based `(row, column)` entries with [`Mutual::resolved`]` == false`,
    /// sorted in row-major order, without duplicates. An empty list means
    /// every entry met its accuracy criterion.
    ///
    /// A symmetric matrix reports both `(i, j)` and `(j, i)`, so the length
    /// counts matrix entries, not unordered filament pairs.
    pub unresolved_pairs: Vec<(usize, usize)>,
}

/// Partial mutual inductances between every filament of `rows` and every
/// filament of `cols`: entry `(i, j)` is
/// [`mutual_inductance`](super::mutual_inductance)`(&rows[i], &cols[j])`,
/// in henries.
///
/// The inner Gauss–Legendre quadrature runs in SIMD lanes and the rows are
/// distributed over the `rayon` thread pool. Results do not depend on the
/// number of threads.
/// Use [`mutual_batch_detailed`] to also identify reduced-accuracy entries.
///
/// # Errors
///
/// The first [`KernelError`] encountered, carrying the `(row, column)` of the
/// offending pair.
pub fn mutual_batch(rows: &[Filament], cols: &[Filament]) -> Result<DMatrix<f64>, KernelError> {
    mutual_batch_with(rows, cols, Execution::Parallel)
}

/// [`mutual_batch`] with an explicit [`Execution`] mode. All modes agree to
/// rounding error.
///
/// # Errors
///
/// As [`mutual_batch`].
pub fn mutual_batch_with(
    rows: &[Filament],
    cols: &[Filament],
    execution: Execution,
) -> Result<DMatrix<f64>, KernelError> {
    let data = evaluate_batch(rows, cols, execution, |m| m.value)?;
    Ok(DMatrix::from_row_slice(rows.len(), cols.len(), &data))
}

/// [`mutual_batch`] with a report of reduced-accuracy entries. Uses
/// [`Execution::Parallel`], and evaluates each pair only once.
///
/// # Errors
///
/// As [`mutual_batch`]. Unresolved pairs are reported, not treated as errors.
pub fn mutual_batch_detailed(
    rows: &[Filament],
    cols: &[Filament],
) -> Result<MutualBatch, KernelError> {
    mutual_batch_detailed_with(rows, cols, Execution::Parallel)
}

/// [`mutual_batch_detailed`] with an explicit [`Execution`] mode. The report
/// order is independent of the execution mode and thread count.
///
/// # Errors
///
/// As [`mutual_batch`].
pub fn mutual_batch_detailed_with(
    rows: &[Filament],
    cols: &[Filament],
    execution: Execution,
) -> Result<MutualBatch, KernelError> {
    let data = evaluate_batch(rows, cols, execution, |m| (m.value, m.resolved))?;
    let (nrows, ncols) = (rows.len(), cols.len());
    let unresolved_pairs = data
        .iter()
        .enumerate()
        .filter(|(_, (_, resolved))| !resolved)
        .map(|(index, _)| (index / ncols, index % ncols))
        .collect();
    Ok(MutualBatch {
        values: DMatrix::from_fn(nrows, ncols, |i, j| data[i * ncols + j].0),
        unresolved_pairs,
    })
}

// Keep the value-only path's storage and execution policy: it retains only
// f64s, while the detailed path retains each pair's flag as well.
fn evaluate_batch<T: Clone + Default + Send>(
    rows: &[Filament],
    cols: &[Filament],
    execution: Execution,
    select: impl Fn(Mutual) -> T + Sync,
) -> Result<Vec<T>, KernelError> {
    let (nrows, ncols) = (rows.len(), cols.len());
    if nrows == 0 || ncols == 0 {
        return Ok(Vec::new());
    }
    let simd = execution != Execution::Scalar;
    let fill = |(i, out): (usize, &mut [T])| -> Result<(), KernelError> {
        for (j, slot) in out.iter_mut().enumerate() {
            *slot = select(evaluate(&rows[i], &cols[j], simd).map_err(|e| e.at(i, j))?);
        }
        Ok(())
    };
    let mut data = vec![T::default(); nrows * ncols];
    if execution == Execution::Parallel {
        data.par_chunks_mut(ncols).enumerate().try_for_each(fill)?;
    } else {
        data.chunks_mut(ncols).enumerate().try_for_each(fill)?;
    }
    Ok(data)
}

/// The full symmetric partial-inductance matrix `L` of a set of filaments:
/// self terms on the diagonal, mutual terms elsewhere.
///
/// Equal to `mutual_batch(filaments, filaments)` — which is itself symmetric
/// to the last bit — at half the cost, since only the upper triangle is
/// evaluated.
/// Use [`partial_inductance_matrix_detailed`] to also identify reduced-accuracy
/// entries.
///
/// # Errors
///
/// As [`mutual_batch`].
pub fn partial_inductance_matrix(filaments: &[Filament]) -> Result<DMatrix<f64>, KernelError> {
    let upper = evaluate_upper(filaments, |m| m.value)?;
    let n = filaments.len();
    Ok(DMatrix::from_fn(n, n, |i, j| {
        let (lo, hi) = (i.min(j), i.max(j));
        upper[lo][hi - lo]
    }))
}

/// [`partial_inductance_matrix`] with a report of reduced-accuracy entries.
/// Only the upper triangle is evaluated, then values and accuracy flags are
/// mirrored. The report includes both triangles in row-major order.
///
/// # Errors
///
/// As [`partial_inductance_matrix`], with errors attributed to the evaluated
/// upper-triangle pair. Unresolved pairs are reported, not treated as errors.
pub fn partial_inductance_matrix_detailed(
    filaments: &[Filament],
) -> Result<MutualBatch, KernelError> {
    let upper = evaluate_upper(filaments, |m| (m.value, m.resolved))?;
    let n = filaments.len();
    let mut unresolved_pairs = Vec::new();
    for (i, row) in upper.iter().enumerate() {
        for (offset, &(_, resolved)) in row.iter().enumerate() {
            if !resolved {
                let j = i + offset;
                unresolved_pairs.push((i, j));
                if i != j {
                    unresolved_pairs.push((j, i));
                }
            }
        }
    }
    unresolved_pairs.sort_unstable();
    Ok(MutualBatch {
        values: DMatrix::from_fn(n, n, |i, j| {
            let (lo, hi) = (i.min(j), i.max(j));
            upper[lo][hi - lo].0
        }),
        unresolved_pairs,
    })
}

/// How many partial inductances of a filament set took each evaluation path
/// ([`Method`]), counted over the upper triangle (unordered pairs, including
/// the `n` self terms). Produced by
/// [`partial_inductance_method_counts`](partial_inductance_method_counts) as
/// the Hoer-Love fallback profile of a mesh against the kernel dispatch (the
/// spec 7.2 test-4 count).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MethodCounts {
    /// [`Method::Orthogonal`]: exactly zero.
    pub orthogonal: usize,
    /// [`Method::BarClosedForm`]: the 64-term Hoer-Love/Ruehli closed form.
    pub bar_closed_form: usize,
    /// [`Method::AlignedQuadrature`]: the closed form's rounding-error
    /// fallback. `aligned_quadrature` alone is the Hoer-Love fallback count;
    /// `bar_closed_form + aligned_quadrature` is the aligned-bar total.
    pub aligned_quadrature: usize,
    /// [`Method::PointQuadrature`]: adaptive-order volume quadrature.
    pub point_quadrature: usize,
    /// [`Method::SampledFilaments`]: cross-section sampling.
    pub sampled_filaments: usize,
}

impl MethodCounts {
    /// Unordered pairs counted, `n·(n+1)/2` — the sum of every field.
    pub fn total(&self) -> usize {
        self.orthogonal
            + self.bar_closed_form
            + self.aligned_quadrature
            + self.point_quadrature
            + self.sampled_filaments
    }

    fn add(&mut self, method: Method) {
        match method {
            Method::Orthogonal => self.orthogonal += 1,
            Method::BarClosedForm => self.bar_closed_form += 1,
            Method::AlignedQuadrature => self.aligned_quadrature += 1,
            Method::PointQuadrature => self.point_quadrature += 1,
            Method::SampledFilaments => self.sampled_filaments += 1,
        }
    }

    fn merge(&mut self, other: Self) {
        self.orthogonal += other.orthogonal;
        self.bar_closed_form += other.bar_closed_form;
        self.aligned_quadrature += other.aligned_quadrature;
        self.point_quadrature += other.point_quadrature;
        self.sampled_filaments += other.sampled_filaments;
    }
}

/// [`partial_inductance_matrix`] reduced to a histogram of the [`Method`] used
/// for each unordered pair (upper triangle, including the `n` self terms): the
/// Hoer-Love fallback profile of a mesh against the kernel dispatch (the spec
/// 7.2 test-4 count).
///
/// Unlike [`partial_inductance_matrix_detailed`], this never builds the matrix.
/// It dispatches each pair, tallies its [`Method`], and drops the value, so
/// peak memory is O(threads), not O(n²) — usable on a real board mesh where a
/// dense matrix would not fit.
///
/// # Errors
///
/// As [`partial_inductance_matrix`], attributed to the evaluated pair.
pub fn partial_inductance_method_counts(
    filaments: &[Filament],
) -> Result<MethodCounts, KernelError> {
    let n = filaments.len();
    // Per-row tallies, then a fold: no global lock, and independent of the
    // thread count. The rows are O(n) to hold, not O(n²).
    let rows: Vec<MethodCounts> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut c = MethodCounts::default();
            for j in i..n {
                let m = evaluate(&filaments[i], &filaments[j], true).map_err(|e| e.at(i, j))?;
                c.add(m.method);
            }
            Ok(c)
        })
        .collect::<Result<_, KernelError>>()?;
    Ok(rows
        .into_iter()
        .fold(MethodCounts::default(), |mut acc, c| {
            acc.merge(c);
            acc
        }))
}

/// [`partial_inductance_matrix`] with the pairs a [`PairMask`] rejects
/// truncated to zero — and, more to the point, never evaluated.
///
/// Entries the mask keeps are bit-for-bit those of
/// [`partial_inductance_matrix`]; entries it rejects are exactly `0.0`. With
/// [`PairMask::all`] the two functions agree everywhere.
///
/// The result is symmetric, so it is still a valid branch-inductance matrix —
/// but truncation is a physical approximation, not a free lunch: see
/// [`crate::coupling`] for when it is safe.
///
/// # Errors
///
/// As [`partial_inductance_matrix`], for the pairs actually evaluated. A pair
/// the mask rejects can no longer report an error.
pub fn partial_inductance_matrix_masked(
    filaments: &[Filament],
    mask: &PairMask,
) -> Result<DMatrix<f64>, KernelError> {
    partial_inductance_matrix_masked_with(filaments, mask, |a, b| {
        evaluate(a, b, true).map(|m| m.value)
    })
}

/// [`partial_inductance_matrix_masked`] with a caller-supplied kernel.
///
/// `kernel` is called exactly once per unordered pair the mask keeps — which
/// is [`PairMask::kept_pairs`] times — and never for a pair it rejects. That
/// makes the truncation measurable: pass a counting wrapper around
/// [`mutual_inductance`](super::mutual_inductance) and compare against
/// [`PairMask::total_pairs`].
///
/// # Errors
///
/// The first error `kernel` returns, in row-major order of the upper
/// triangle, attributed to that pair.
pub fn partial_inductance_matrix_masked_with<K>(
    filaments: &[Filament],
    mask: &PairMask,
    kernel: K,
) -> Result<DMatrix<f64>, KernelError>
where
    K: Fn(&Filament, &Filament) -> Result<f64, KernelError> + Sync,
{
    let n = filaments.len();
    let upper: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (i..n)
                .map(|j| {
                    if mask.keeps(i, j) {
                        kernel(&filaments[i], &filaments[j]).map_err(|e| e.at(i, j))
                    } else {
                        Ok(0.0)
                    }
                })
                .collect()
        })
        .collect::<Result<_, KernelError>>()?;
    Ok(DMatrix::from_fn(n, n, |i, j| {
        let (lo, hi) = (i.min(j), i.max(j));
        upper[lo][hi - lo]
    }))
}

fn evaluate_upper<T: Send>(
    filaments: &[Filament],
    select: impl Fn(Mutual) -> T + Sync,
) -> Result<Vec<Vec<T>>, KernelError> {
    let n = filaments.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            (i..n)
                .map(|j| {
                    evaluate(&filaments[i], &filaments[j], true)
                        .map(&select)
                        .map_err(|e| e.at(i, j))
                })
                .collect()
        })
        .collect()
}
