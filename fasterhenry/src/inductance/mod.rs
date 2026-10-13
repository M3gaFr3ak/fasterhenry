//! Partial self and mutual inductances of rectangular filaments — the
//! numerical core of the PEEC method.
//!
//! For two straight filaments carrying uniform current densities along unit
//! vectors `l̂ᵢ`, `l̂ⱼ`, Ruehli's partial inductance is the Neumann integral
//! averaged over both cross-sections (Ruehli, *IBM J. Res. Dev.* 16 (1972);
//! Kamon, Tsuk & White, *IEEE T-MTT* 42 (1994), §III):
//!
//! ```text
//! Lᵢⱼ = μ0/(4π) · (l̂ᵢ·l̂ⱼ)/(AᵢAⱼ) · ∫_{Vᵢ} ∫_{Vⱼ} dV dV' / |r − r'|
//! ```
//!
//! # Methods
//!
//! [`mutual_inductance`] picks one of four evaluations of that integral;
//! [`mutual_inductance_detailed`] reports which one ([`Method`]).
//!
//! | Pair | Method |
//! |------|--------|
//! | orthogonal | exactly zero |
//! | separated by ≳ half a length | [`Method::PointQuadrature`]: adaptive-order Gauss–Legendre over all six dimensions, SIMD over the inner points |
//! | parallel, cross-section axes aligned (a filament with itself, its bundle neighbours, collinear or stacked segments) | [`Method::BarClosedForm`]: the exact 64-term Hoer–Love/Ruehli formula, when its rounding error is provably small; otherwise [`Method::AlignedQuadrature`]: exact in both lengths, singularity-aware quadrature over the cross-sections |
//! | anything else that is close | [`Method::SampledFilaments`]: exact Grover line-to-line formulas sampled over both cross-sections |
//!
//! The aligned-bar methods reach a relative accuracy of about `1e-9` and the
//! point quadrature about `1e-8`; see [`Method::SampledFilaments`] for the
//! last, and the crate's integration tests for the validation of every path
//! against independent numerical integration.
//!
//! # Validated range and limitations
//!
//! * Self and aligned-bar terms are validated against independent references
//!   for length : cross-section ratios from `0.2` to `1e7` and cross-section
//!   aspect ratios up to `1e4`, and exercised (finite, monotone, bounded by
//!   the Cauchy–Schwarz inequality) down to `1e-3`.
//! * Non-aligned bars that touch, overlap, or lie closer than about half a
//!   cross-section extent are flagged [`Mutual::resolved`]` == false`; on the
//!   bends and crossings of the test-suite they are within `1e-4` of a fine
//!   reference, and should be trusted to about `1e-3`.
//! * Filaments between `1e-6` and `1e-4` rad from parallel that are also
//!   close and not aligned lose accuracy as `1e-17/sin²ε` (to `1e-5` at
//!   `1e-6` rad); below `1e-6` rad they are treated as exactly parallel.
//! * Current density is uniform over each cross-section and directed along
//!   the filament — the PEEC assumption. Nothing here is frequency dependent.
//!
//! # Units
//!
//! SI throughout: metres in, henries out.

mod aligned;
mod bar;
mod batch;
pub mod closed_form;
pub(crate) mod gauss;
mod lines;
mod memo;
pub(crate) mod neumann;

use std::cmp::Ordering;

use thiserror::Error;

use crate::filament::Filament;
use aligned::AlignedBars;
use neumann::Bar;

pub use batch::{
    mutual_batch, mutual_batch_detailed, mutual_batch_detailed_with, mutual_batch_with,
    partial_inductance_matrix, partial_inductance_matrix_detailed,
    partial_inductance_matrix_masked, partial_inductance_matrix_masked_with,
    partial_inductance_method_counts, partial_inductance_method_counts_with, Execution,
    MethodCounts, MutualBatch, PairMask,
};

/// Vacuum permeability `μ0 = 4π × 10⁻⁷ H/m` (the classical defined value,
/// which the 2019 SI value matches to better than 1e-9).
pub const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

/// `μ0 / (4π)`, the prefactor of the Neumann integral, in H/m.
pub(crate) const MU0_OVER_4PI: f64 = 1.0e-7;

/// Largest rounding-error estimate at which the 64-term closed form is used.
const CLOSED_FORM_MAX_ERROR: f64 = 1e-10;
/// `|cos ε|` below which two filaments are orthogonal to rounding error.
const ORTHOGONAL_COS: f64 = 1e-14;
/// Point quadrature is preferred over the aligned-bar methods when it needs
/// no more than this many kernel evaluations (a few microseconds in SIMD
/// lanes — about the cost of the 64-term closed form).
const CHEAP_POINT_PAIRS: usize = 4096;

/// Why a partial inductance could not be computed.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum KernelError {
    /// Two parallel filaments with differently oriented cross-sections
    /// overlap in such a way that two of the sample lines used to integrate
    /// over them coincide, where the line-to-line integral diverges. Rotate
    /// the cross-sections into alignment (any multiple of 90°) or separate
    /// the filaments.
    #[error("overlapping parallel filaments {first} and {second} have misaligned cross-sections and coincident sample lines")]
    CoincidentSamples {
        /// Row index of the pair in a batch (0 for a single evaluation).
        first: usize,
        /// Column index of the pair in a batch (0 for a single evaluation).
        second: usize,
    },
    /// The integral evaluated to a non-finite number — the filament data
    /// contained infinities or NaNs, or lay outside the validated range.
    #[error("partial inductance of filaments {first} and {second} is not finite")]
    NotFinite {
        /// Row index of the pair in a batch (0 for a single evaluation).
        first: usize,
        /// Column index of the pair in a batch (0 for a single evaluation).
        second: usize,
    },
}

impl KernelError {
    /// The same error, attributed to entry `(first, second)` of a batch.
    pub(crate) fn at(self, first: usize, second: usize) -> Self {
        match self {
            Self::CoincidentSamples { .. } => Self::CoincidentSamples { first, second },
            Self::NotFinite { .. } => Self::NotFinite { first, second },
        }
    }
}

/// How a partial inductance was evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Method {
    /// The filaments are orthogonal, so `l̂ᵢ·l̂ⱼ = 0` and the result is
    /// exactly zero.
    Orthogonal,
    /// Exact closed form for parallel bars with aligned cross-sections (Hoer
    /// & Love 1965; Ruehli 1972), used only while its estimated rounding
    /// error stays below `1e-10`.
    BarClosedForm,
    /// Parallel bars with aligned cross-sections: both length integrals in
    /// closed form (Grover's parallel-filament formula), then a graded,
    /// singularity-aware Gauss–Legendre quadrature over the cross-section
    /// differences. Relative accuracy about `1e-9` for any aspect ratio,
    /// including touching and overlapping bars.
    AlignedQuadrature,
    /// Adaptive-order Gauss–Legendre quadrature of the Neumann integral over
    /// both volumes. Relative accuracy about `1e-8`.
    PointQuadrature,
    /// Close bars that are not parallel-and-aligned: both length integrals
    /// in closed form (Grover's parallel or inclined filament formulas),
    /// sampled over both cross-sections with adaptive Gauss–Legendre order.
    /// Relative accuracy about `1e-6` when [`Mutual::resolved`] is true.
    SampledFilaments,
}

/// A partial inductance together with how it was obtained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mutual {
    /// The partial inductance, in henries. Negative for filaments whose
    /// directions oppose each other.
    pub value: f64,
    /// The evaluation path taken.
    pub method: Method,
    /// Whether the method's a-priori accuracy criterion was met. It is false
    /// only for [`Method::SampledFilaments`] on bars that touch, overlap, or
    /// lie closer than about half a cross-section extent: the cross-section
    /// integrand is then weakly singular and the fixed maximum sampling order
    /// is used, which the test-suite shows to be within `1e-4` of a fine
    /// reference for bars meeting at a bend or interpenetrating at a crossing.
    pub resolved: bool,
}

/// Partial self-inductance of a filament, in henries.
///
/// Evaluated as the mutual inductance of the bar with itself, by the exact
/// closed form of Hoer & Love (1965) and Ruehli (1972) when its cancellation
/// error is small, and otherwise by the equivalent semi-analytic quadrature
/// (see [`Method::AlignedQuadrature`]), which stays accurate for length to
/// cross-section ratios beyond `1e6`.
///
/// # Example
///
/// ```
/// use fasterhenry::{self_inductance, Filament, Node, Segment};
///
/// // 10 mm of 1 mm × 35 µm copper trace: about 7 nH.
/// let trace = Segment::new(
///     Node::new(0.0, 0.0, 0.0),
///     Node::new(10e-3, 0.0, 0.0),
///     1e-3,
///     35e-6,
///     5.8e7,
/// );
/// let l = self_inductance(&Filament::new(&trace)?);
/// assert!((l - 6.99e-9).abs() < 0.01e-9);
/// # Ok::<(), fasterhenry::SegmentError>(())
/// ```
pub fn self_inductance(filament: &Filament) -> f64 {
    let bars = AlignedBars::same(filament.length(), filament.width(), filament.height());
    let (integral, _) = aligned_integral(&bars);
    // Same arithmetic as the aligned branch of `evaluate`, so that the
    // diagonal of a batch equals this to the last bit.
    MU0_OVER_4PI * (integral * (1.0 / (filament.area() * filament.area())))
}

/// Partial mutual inductance of two filaments, in henries.
///
/// The result is symmetric to the last bit: `mutual_inductance(a, b)` and
/// `mutual_inductance(b, a)` are evaluated identically. Passing the same
/// filament twice yields its [`self_inductance`].
///
/// # Errors
///
/// See [`KernelError`]. No error is possible for filaments produced by one
/// [`discretize`](crate::discretize) call, or for any pair of filaments with
/// aligned cross-sections.
pub fn mutual_inductance(a: &Filament, b: &Filament) -> Result<f64, KernelError> {
    mutual_inductance_detailed(a, b).map(|m| m.value)
}

/// As [`mutual_inductance`], also reporting the [`Method`] used and whether
/// it met its accuracy criterion.
///
/// # Errors
///
/// See [`KernelError`].
pub fn mutual_inductance_detailed(a: &Filament, b: &Filament) -> Result<Mutual, KernelError> {
    evaluate(a, b, true)
}

/// Deterministic total order on filaments, used to evaluate every unordered
/// pair the same way round.
fn canonical(a: &Filament, b: &Filament) -> Ordering {
    let key = |f: &Filament| {
        let (s, e, w) = (f.start(), f.end(), f.width_dir());
        [
            s.x,
            s.y,
            s.z,
            e.x,
            e.y,
            e.z,
            f.width(),
            f.height(),
            w.x,
            w.y,
            w.z,
        ]
    };
    let (ka, kb) = (key(a), key(b));
    ka.iter()
        .zip(&kb)
        .map(|(x, y)| x.total_cmp(y))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
}

/// The closed form when trustworthy, the quadrature otherwise, memoized on
/// the geometry — which is the whole of what the answer depends on, so a
/// repeated relative position costs a table lookup instead of a kernel. See
/// [`memo`] for why a bundle is full of repeats.
fn aligned_integral(bars: &AlignedBars) -> (f64, Method) {
    memo::aligned_integral(bars, evaluate_aligned_integral)
}

/// [`aligned_integral`] without the memo.
fn evaluate_aligned_integral(bars: &AlignedBars) -> (f64, Method) {
    let exact = bar::bar_integral(
        &bar::corner_differences(bars.length[0], bars.length[1], bars.offset[0]),
        &bar::corner_differences(bars.width[0], bars.width[1], bars.offset[1]),
        &bar::corner_differences(bars.height[0], bars.height[1], bars.offset[2]),
    );
    if exact.relative_error <= CLOSED_FORM_MAX_ERROR {
        (exact.value, Method::BarClosedForm)
    } else {
        (aligned::integral(bars), Method::AlignedQuadrature)
    }
}

/// If the cross-section axes of `b` coincide (up to sign and a swap) with
/// those of `a`, the pair expressed in `a`'s frame.
fn aligned_bars(a: &Bar, b: &Bar) -> Option<AlignedBars> {
    const ALIGNED: f64 = 1.0 - 1e-12;
    let (width_b, height_b) = if a.axes[1].dot(&b.axes[1]).abs() > ALIGNED {
        (b.half[1], b.half[2])
    } else if a.axes[1].dot(&b.axes[2]).abs() > ALIGNED {
        (b.half[2], b.half[1])
    } else {
        return None;
    };
    let delta = b.centre - a.centre;
    Some(AlignedBars {
        length: [2.0 * a.half[0], 2.0 * b.half[0]],
        width: [2.0 * a.half[1], 2.0 * width_b],
        height: [2.0 * a.half[2], 2.0 * height_b],
        offset: [
            delta.dot(&a.axes[0]),
            delta.dot(&a.axes[1]),
            delta.dot(&a.axes[2]),
        ],
    })
}

/// Shared implementation; `simd` selects the lane-parallel inner quadrature.
pub(crate) fn evaluate(a: &Filament, b: &Filament, simd: bool) -> Result<Mutual, KernelError> {
    let (a, b) = if canonical(a, b) == Ordering::Greater {
        (b, a)
    } else {
        (a, b)
    };
    let cos = a.direction().dot(&b.direction());
    if cos.abs() < ORTHOGONAL_COS {
        return Ok(Mutual {
            value: 0.0,
            method: Method::Orthogonal,
            resolved: true,
        });
    }
    let (bar_a, bar_b) = (Bar::new(a), Bar::new(b));
    let parallel = a.direction().cross(&b.direction()).norm() < neumann::PARALLEL_SIN;
    let aligned = if parallel {
        aligned_bars(&bar_a, &bar_b)
    } else {
        None
    };
    let gap = neumann::separation(&bar_a, &bar_b);
    let orders = neumann::point_orders(&bar_a, gap).zip(neumann::point_orders(&bar_b, gap));
    let pairs = |(oa, ob): &([usize; 3], [usize; 3])| {
        oa.iter().product::<usize>() * ob.iter().product::<usize>()
    };

    // Point quadrature whenever it applies — except that for aligned bars the
    // dedicated methods win unless the point rule is cheap.
    let point_orders = orders.filter(|o| aligned.is_none() || pairs(o) <= CHEAP_POINT_PAIRS);

    let (integral, method, resolved) = if let Some((oa, ob)) = point_orders {
        let sum = neumann::point_sum(&bar_a, oa, &bar_b, ob, simd);
        (sum * cos, Method::PointQuadrature, true)
    } else if let Some(bars) = aligned {
        let (integral, method) = aligned_integral(&bars);
        // The aligned-bar integral carries the m⁴ of the two areas, and the
        // relative direction is exactly ±1 once the axes are snapped.
        let scale = cos.signum() / (a.area() * b.area());
        (integral * scale, method, true)
    } else {
        let sampled = neumann::sampled_filaments(&bar_a, &bar_b, gap, parallel).ok_or(
            KernelError::CoincidentSamples {
                first: 0,
                second: 0,
            },
        )?;
        let direction = if parallel { cos.signum() } else { cos };
        (
            sampled.value * direction,
            Method::SampledFilaments,
            sampled.resolved,
        )
    };
    let value = MU0_OVER_4PI * integral;
    if !value.is_finite() {
        return Err(KernelError::NotFinite {
            first: 0,
            second: 0,
        });
    }
    Ok(Mutual {
        value,
        method,
        resolved,
    })
}
