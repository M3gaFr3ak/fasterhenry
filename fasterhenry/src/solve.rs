//! Assembly of the PEEC mesh system and its solution for the port impedance
//! matrix `Z(ω)`.
//!
//! # Formulation
//!
//! Following Kamon, Tsuk & White (IEEE T-MTT 1994, §II.B): with one branch per
//! filament, the branch voltages and currents obey `v = Z_b · I`, where
//!
//! ```text
//! Z_b = R + jωL,   R = diag(lₖ / (σₖ·Aₖ)),   L = partial-inductance matrix.
//! ```
//!
//! Writing the branch currents in terms of loop currents, `I = Mᵀ·I_m` with
//! the loop basis `M` of [`crate::mesh`], Kirchhoff's voltage law around every
//! loop gives the mesh system
//!
//! ```text
//! (M · Z_b · Mᵀ) · I_m = V_s ,
//! ```
//!
//! where `V_s` is zero for the internal loops and the port voltage for each
//! port loop. Partitioning the loops into internal (`e`) and port (`p`),
//!
//! ```text
//! ⎡ Z_ee  Z_ep ⎤ ⎡ I_e ⎤   ⎡ 0   ⎤
//! ⎣ Z_pe  Z_pp ⎦ ⎣ I_p ⎦ = ⎣ V_p ⎦      ⇒      V_p = Z(ω) · I_p ,
//!
//! Z(ω) = Z_pp − Z_pe · Z_ee⁻¹ · Z_ep .
//! ```
//!
//! That Schur complement *is* the port impedance matrix, so one dense LU
//! factorization of `Z_ee` ([`crate::dense::lu_solve`]: partial pivoting,
//! blocked and parallel) and one solve with a right-hand side per port is
//! all a frequency costs. It is algebraically identical to exciting one port
//! at a time and inverting the resulting admittance matrix, without the
//! second inversion. When there are no internal loops — every segment a
//! single filament, no closed rings — `Z(ω) = Z_pp` and nothing is solved.
//!
//! [`IterativeSystem`](crate::IterativeSystem) is the matrix-free
//! alternative for large problems: the same Schur complement, with
//! `Z_ee⁻¹·Z_ep` solved by GMRES on the precorrected-FFT operator instead of
//! by dense LU. [`MeshSystem`] remains the default up to
//! [`DENSE_PATH_MAX_FILAMENTS`] filaments — the size threshold
//! [`SolverChoice::Auto`] switches on, measured in `docs/benchmarks.md`.
//!
//! `Z_ee` is never singular for a valid geometry: `R` is positive definite and
//! `L` positive semidefinite, so the Hermitian part of `M_e (R + jωL) M_eᵀ` is
//! positive definite for independent loops, at every `ω ≥ 0`. In particular a
//! floating conductor without a port is harmless — it only adds internal
//! loops, in which eddy currents are induced.
//!
//! # Frequency independence and the DC limit
//!
//! `M R Mᵀ` and `M L Mᵀ` are real, frequency independent, and assembled once
//! by [`MeshSystem::assemble`] (the `O(b²)` kernel evaluations dominate);
//! each frequency then only forms `R_m + jωL_m` and factorizes it. At exactly
//! `f = 0` the system is solved in real arithmetic from `M R Mᵀ` alone, so the
//! result is the resistive network's, with an imaginary part of exactly zero
//! and no division by `ω` anywhere.
//!
//! # Units and conventions
//!
//! SI: geometry in metres, conductivity in S/m, frequencies in **hertz**
//! (`ω = 2πf`), impedances in ohms. Port orientation is documented in
//! [`crate::mesh`]; `Z` is symmetric (reciprocity) up to rounding.

use std::time::Instant;

use nalgebra::{ComplexField, DMatrix};
use num_complex::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::coupling::{Coupling, CouplingError, TruncationWarning};
use crate::dense::lu_solve;
use crate::filament::{
    discretize, discretize_graded_per_axis, graded_surface_extent, DiscretizeError, Filament,
};
use crate::geometry::Geometry;
use crate::inductance::{
    partial_inductance_matrix, partial_inductance_matrix_masked, KernelError, MU0,
};
use crate::mesh::{MeshError, MeshMatrix, Port};
use crate::pfft::PfftError;
use crate::result::{Counts, Provenance, SweepResult, Timing};

/// Working memory the frequency sweep may spend on concurrent factorizations
/// before it reduces the number of frequencies solved at once.
pub(crate) const SWEEP_MEMORY_BUDGET_BYTES: usize = 2 << 30;

/// Largest filament count at which [`SolverChoice::Auto`] keeps the dense
/// path ([`MeshSystem`]); above it the matrix-free
/// [`IterativeSystem`](crate::IterativeSystem) is selected instead.
///
/// The dense path holds the `n × n` partial-inductance matrix (`8n²` bytes)
/// and factorizes the complex internal-loop block (`16m²` bytes for `m`
/// internal loops, twice over while it does), so its working set grows
/// quadratically where the pFFT operator and GMRES's Krylov basis grow
/// linearly. Measured end to end — assembly plus a one-frequency sweep, by
/// `fasterhenry/benches/scaling.rs`, on the square-meander fixture, one
/// thread; hardware, date and the full table in `docs/benchmarks.md`. The
/// wall times were measured before #58 and #60, which only shorten them; the
/// memory column, and so the memory-based threshold, is unaffected:
///
/// | filaments | dense | iterative | dense working set |
/// |---|---|---|---|
/// | 2 400 | 4.2 s | 4.7 s | 0.15 GB |
/// | 4 760 | 20.9 s | 8.9 s | 0.6 GB |
/// | 9 800 | 156 s | 19.4 s | 2.5 GB |
/// | 29 928 | not attempted | 61 s | 23 GB |
/// | 99 224 | not attempted | 207 s | 256 GB |
///
/// The wall-clock crossover on that fixture is near 3 000 filaments, and
/// this threshold deliberately sits above it: it marks where the dense path
/// stops being *tractable*, not where it stops being fastest. The dense path
/// evaluates every pair exactly, while the iterative one approximates the
/// far field (about `1e-6` on `L·x`, under `1e-4` on `Z`), and its advantage
/// is geometry-dependent — a set with a few filaments much longer than the
/// rest drives the pFFT grid spacing down and its projection cost up, moving
/// the crossover to larger sizes. Staying dense to 10 000 filaments keeps
/// the exact path for every problem where it is affordable on any geometry
/// (its working set is still a couple of gigabytes there) and hands over one
/// size before it is not affordable on any: 23 GB at 30 000 filaments. The
/// price of that conservatism is visible in the table — a default run at
/// 9 800 filaments is the slow column — and one flag wide:
/// `--solver iterative` takes the other path at any size.
///
/// It is a default for callers that do not want to choose
/// ([`SolverChoice::Auto`]), never a limit: [`SolverChoice::Dense`] and
/// [`SolverChoice::Iterative`] force either path at any size.
pub const DENSE_PATH_MAX_FILAMENTS: usize = 10_000;

/// Which of the two solve paths runs.
///
/// Deliberately exhaustive (unlike the crate's error enums and
/// [`Discretization`]): a new solve path is an architectural change, and
/// code that dispatches on the path — such as deciding which features a path
/// can honour — should fail to compile until it has decided for the new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Solver {
    /// [`MeshSystem`]: dense assembly of `L` and a dense LU of `Z_ee`.
    Dense,
    /// [`IterativeSystem`](crate::IterativeSystem): GMRES on the
    /// precorrected-FFT operator, matrix-free.
    Iterative,
}

impl std::fmt::Display for Solver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Dense => "dense",
            Self::Iterative => "iterative",
        })
    }
}

/// Which solve path a caller wants, before the problem size is known.
///
/// [`Auto`](Self::Auto) — the default — resolves to [`Solver::Dense`] at or
/// below [`DENSE_PATH_MAX_FILAMENTS`] filaments and to
/// [`Solver::Iterative`] above it; the other two variants force a path.
///
/// Deliberately exhaustive, for the same reason as [`Solver`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverChoice {
    /// Pick from the filament count and [`DENSE_PATH_MAX_FILAMENTS`].
    #[default]
    Auto,
    /// Always [`MeshSystem`].
    Dense,
    /// Always [`IterativeSystem`](crate::IterativeSystem).
    Iterative,
}

impl SolverChoice {
    /// The path this choice takes for a problem of `filaments` filaments.
    pub const fn resolve(self, filaments: usize) -> Solver {
        match self {
            Self::Dense => Solver::Dense,
            Self::Iterative => Solver::Iterative,
            Self::Auto if filaments <= DENSE_PATH_MAX_FILAMENTS => Solver::Dense,
            Self::Auto => Solver::Iterative,
        }
    }
}

/// Number of filaments across a segment's width (`nw`) and height (`nh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Subdivision {
    /// Filaments across the width.
    pub nw: usize,
    /// Filaments across the height.
    pub nh: usize,
}

impl Subdivision {
    /// `nw` filaments across the width and `nh` across the height.
    pub const fn new(nw: usize, nh: usize) -> Self {
        Self { nw, nh }
    }

    /// One filament per segment: uniform current density, no skin effect.
    pub const SINGLE: Self = Self::new(1, 1);
}

/// An `nw × nh` grid graded geometrically toward the conductor surfaces:
/// each filament is `ratio` times the extent of its neighbour one step
/// nearer the surface (see [`crate::filament::discretize_graded`]).
///
/// Supported grids at `ratio == 1.0` equal the uniform grid bit for bit.
/// Assembly rejects grids outside [`discretize_graded`](crate::filament::discretize_graded)'s scale-aware
/// numerical limits, reporting the segment, axis, counts and ratio.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grading {
    /// Filaments across the width.
    pub nw: usize,
    /// Filaments across the height.
    pub nh: usize,
    /// Ratio of adjacent filament extents, coarsening inward; at least 1.
    pub ratio: f64,
}

impl Grading {
    /// `nw × nh` filaments graded inward by `ratio`.
    pub const fn new(nw: usize, nh: usize, ratio: f64) -> Self {
        Self { nw, nh, ratio }
    }

    fn validate(&self) -> Result<(), SolveError> {
        if self.ratio.is_finite() && self.ratio >= 1.0 {
            Ok(())
        } else {
            Err(SolveError::InvalidGrading {
                reason: format!(
                    "the grading ratio must be finite and at least 1, got {}",
                    self.ratio
                ),
            })
        }
    }
}

/// One segment's `nw × nh` grid with an independent grading ratio along
/// each cross-section axis (see
/// [`crate::filament::discretize_graded_per_axis`]): each filament across
/// the width is `width_ratio` times the extent of its neighbour one step
/// nearer the surface, and likewise across the height at `height_ratio`.
///
/// Both ratios at `1.0` is the uniform [`Subdivision`] of the same counts,
/// bit for bit; equal ratios are the [`Grading`] of that ratio.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AxisGrading {
    /// Filaments across the width.
    pub nw: usize,
    /// Filaments across the height.
    pub nh: usize,
    /// Ratio of adjacent filament widths, coarsening inward; at least 1.
    pub width_ratio: f64,
    /// Ratio of adjacent filament heights, coarsening inward; at least 1.
    pub height_ratio: f64,
}

impl AxisGrading {
    /// `nw × nh` filaments, graded inward by `width_ratio` across the width
    /// and by `height_ratio` across the height.
    pub const fn new(nw: usize, nh: usize, width_ratio: f64, height_ratio: f64) -> Self {
        Self {
            nw,
            nh,
            width_ratio,
            height_ratio,
        }
    }

    fn validate(&self) -> Result<(), SolveError> {
        for (axis, ratio) in [("width", self.width_ratio), ("height", self.height_ratio)] {
            if !(ratio.is_finite() && ratio >= 1.0) {
                return Err(SolveError::InvalidGrading {
                    reason: format!(
                        "the {axis} grading ratio must be finite and at least 1, got {ratio}"
                    ),
                });
            }
        }
        Ok(())
    }
}

impl From<Subdivision> for AxisGrading {
    fn from(subdivision: Subdivision) -> Self {
        Self::new(subdivision.nw, subdivision.nh, 1.0, 1.0)
    }
}

/// A graded grid whose filament counts are chosen per segment from the skin
/// depth at a frequency of interest.
///
/// For each segment and each cross-section axis, the count is the smallest
/// one whose *surface* filament is at most `target_skin_depths · δ` thick,
/// where `δ = 1/√(π·f·μ0·σ)` is the skin depth of that segment's material at
/// `frequency_hz` ([`skin_depth`]). The count stops at `max_per_axis` even if
/// the surface filament remains thicker than the target. Because grading buys resolution
/// geometrically, that count grows only logarithmically as the frequency
/// rises, where a uniform grid's grows as `√f`.
///
/// `max_per_axis` caps the result, so a very high frequency degrades to a
/// merely-fine grid instead of an unaffordable one. At `f = 0` the skin
/// depth is infinite and every segment is a single filament.
///
/// The cap controls cost, not numerical safety. After counts are selected,
/// assembly applies [`discretize_graded`](crate::filament::discretize_graded)'s scale-aware limits and returns
/// [`SolveError::Discretize`] with the segment, axis and selected grid if it
/// is unusable. It does not silently replace the selected grid with a coarser
/// one when the requested resolution is numerically unsafe.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkinDepthGrading {
    /// The frequency whose skin depth the grid must resolve, in hertz.
    pub frequency_hz: f64,
    /// Thickness of the surface filaments, in skin depths. Values below 1
    /// resolve the profile within the skin; `0.5` is a reasonable default.
    pub target_skin_depths: f64,
    /// Ratio of adjacent filament extents, coarsening inward; at least 1.
    pub ratio: f64,
    /// Upper bound on the filament count along each axis.
    pub max_per_axis: usize,
}

impl SkinDepthGrading {
    /// Cap used by [`SkinDepthGrading::new`]: 32 filaments per axis, i.e. up
    /// to 1024 per segment.
    pub const DEFAULT_MAX_PER_AXIS: usize = 32;

    /// Surface filaments `target_skin_depths` skin depths thick at
    /// `frequency_hz`, graded inward by `ratio`, capped at
    /// [`DEFAULT_MAX_PER_AXIS`](Self::DEFAULT_MAX_PER_AXIS).
    pub const fn new(frequency_hz: f64, target_skin_depths: f64, ratio: f64) -> Self {
        Self {
            frequency_hz,
            target_skin_depths,
            ratio,
            max_per_axis: Self::DEFAULT_MAX_PER_AXIS,
        }
    }

    /// The same, with an explicit cap on the filaments per axis.
    pub const fn with_max_per_axis(mut self, max_per_axis: usize) -> Self {
        self.max_per_axis = max_per_axis;
        self
    }

    fn validate(&self) -> Result<(), SolveError> {
        let invalid = |reason: String| Err(SolveError::InvalidGrading { reason });
        if !(self.ratio.is_finite() && self.ratio >= 1.0) {
            return invalid(format!(
                "the grading ratio must be finite and at least 1, got {}",
                self.ratio
            ));
        }
        if !(self.target_skin_depths.is_finite() && self.target_skin_depths > 0.0) {
            return invalid(format!(
                "target_skin_depths must be finite and positive, got {}",
                self.target_skin_depths
            ));
        }
        if !(self.frequency_hz.is_finite() && self.frequency_hz >= 0.0) {
            return invalid(format!(
                "frequency_hz must be finite and non-negative, got {}",
                self.frequency_hz
            ));
        }
        if self.max_per_axis == 0 {
            return invalid("max_per_axis must be at least 1".to_owned());
        }
        Ok(())
    }

    /// Smallest filament count whose surface cell is within the target, or
    /// `max_per_axis` when the cap is reached first. Assumes
    /// [`validate`](Self::validate) has passed.
    fn count_for(&self, extent: f64, sigma: f64) -> usize {
        let target = self.target_skin_depths * skin_depth(self.frequency_hz, sigma);
        if target.is_infinite() || extent <= target {
            return 1;
        }
        for count in 2..=self.max_per_axis {
            match graded_surface_extent(extent, count, self.ratio) {
                Ok(surface) if surface <= target => return count,
                // The progression overflowed: no larger count can help.
                Err(_) => return count - 1,
                Ok(_) => {}
            }
        }
        self.max_per_axis
    }
}

/// How the segments of a geometry are cut into filaments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Discretization {
    /// The same uniform subdivision for every segment.
    Uniform(Subdivision),
    /// One uniform subdivision per segment, in segment order.
    PerSegment(Vec<Subdivision>),
    /// The same surface-graded grid for every segment.
    Graded(Grading),
    /// A surface-graded grid whose counts follow the skin depth at a
    /// frequency of interest, segment by segment.
    SkinDepth(SkinDepthGrading),
    /// One grid per segment, in segment order, each with its own counts and
    /// its own grading ratio along each axis — the per-segment counterpart
    /// of [`Graded`](Self::Graded), as a deck with per-segment `rw`/`rh`
    /// needs.
    PerSegmentGraded(Vec<AxisGrading>),
}

impl Discretization {
    /// The same `nw × nh` subdivision for every segment.
    pub const fn uniform(nw: usize, nh: usize) -> Self {
        Self::Uniform(Subdivision::new(nw, nh))
    }

    /// The same `nw × nh` grid for every segment, graded inward by `ratio`.
    pub const fn graded(nw: usize, nh: usize, ratio: f64) -> Self {
        Self::Graded(Grading::new(nw, nh, ratio))
    }

    /// Filament counts chosen per segment to put surface filaments within
    /// `target_skin_depths` skin depths at `frequency_hz`, graded inward by
    /// `ratio`, subject to the per-axis cap. See [`SkinDepthGrading`].
    pub const fn skin_depth(frequency_hz: f64, target_skin_depths: f64, ratio: f64) -> Self {
        Self::SkinDepth(SkinDepthGrading::new(
            frequency_hz,
            target_skin_depths,
            ratio,
        ))
    }

    /// How many filaments this discretization cuts `geometry` into, without
    /// cutting it: the sum of `nw·nh` over the segments.
    ///
    /// Cheap (one pass over the segments), so a front end can size the
    /// problem — and pick a solve path with
    /// [`SolverChoice::resolve`] — before paying for assembly.
    ///
    /// # Errors
    ///
    /// [`SolveError::InvalidGrading`] for an out-of-range grading parameter
    /// and [`SolveError::Mesh`] for a per-segment discretization of the wrong
    /// length, exactly as assembly would report them.
    pub fn filament_count(&self, geometry: &Geometry) -> Result<usize, SolveError> {
        Ok(self
            .resolve(geometry)?
            .iter()
            .map(|grid| grid.nw * grid.nh)
            .sum())
    }

    fn resolve(&self, geometry: &Geometry) -> Result<Vec<SegmentGrid>, SolveError> {
        let segment_count = geometry.segment_count();
        match self {
            Self::Uniform(subdivision) => Ok(vec![SegmentGrid::from(*subdivision); segment_count]),
            Self::PerSegment(list) if list.len() == segment_count => {
                Ok(list.iter().copied().map(SegmentGrid::from).collect())
            }
            Self::PerSegment(list) => Err(SolveError::Mesh(MeshError::SegmentCountMismatch {
                expected: segment_count,
                got: list.len(),
            })),
            Self::PerSegmentGraded(list) if list.len() == segment_count => list
                .iter()
                .map(|grading| {
                    grading.validate()?;
                    Ok(SegmentGrid::from(*grading))
                })
                .collect(),
            Self::PerSegmentGraded(list) => {
                Err(SolveError::Mesh(MeshError::SegmentCountMismatch {
                    expected: segment_count,
                    got: list.len(),
                }))
            }
            Self::Graded(grading) => {
                grading.validate()?;
                Ok(vec![SegmentGrid::from(*grading); segment_count])
            }
            Self::SkinDepth(grading) => {
                grading.validate()?;
                Ok(geometry
                    .segments()
                    .map(|segment| SegmentGrid {
                        nw: grading.count_for(segment.width, segment.sigma),
                        nh: grading.count_for(segment.height, segment.sigma),
                        width_ratio: grading.ratio,
                        height_ratio: grading.ratio,
                    })
                    .collect())
            }
        }
    }
}

/// The grid one segment is actually cut on, once the [`Discretization`] has
/// been resolved against the geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SegmentGrid {
    nw: usize,
    nh: usize,
    width_ratio: f64,
    height_ratio: f64,
}

impl From<Subdivision> for SegmentGrid {
    fn from(subdivision: Subdivision) -> Self {
        Self {
            nw: subdivision.nw,
            nh: subdivision.nh,
            width_ratio: 1.0,
            height_ratio: 1.0,
        }
    }
}

impl From<Grading> for SegmentGrid {
    fn from(grading: Grading) -> Self {
        Self {
            nw: grading.nw,
            nh: grading.nh,
            width_ratio: grading.ratio,
            height_ratio: grading.ratio,
        }
    }
}

impl From<AxisGrading> for SegmentGrid {
    fn from(grading: AxisGrading) -> Self {
        Self {
            nw: grading.nw,
            nh: grading.nh,
            width_ratio: grading.width_ratio,
            height_ratio: grading.height_ratio,
        }
    }
}

/// Skin depth `δ = 1/√(π·f·μ0·σ)` in metres, for a frequency `f` in hertz and
/// a conductivity `sigma` in S/m — the `1/e` depth of the exponential decay
/// of a field diffusing into a good conductor.
///
/// Infinite at `f = 0`, where the current fills the conductor.
pub fn skin_depth(frequency_hz: f64, sigma: f64) -> f64 {
    1.0 / (std::f64::consts::PI * frequency_hz * MU0 * sigma).sqrt()
}

/// Why an impedance extraction failed.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum SolveError {
    /// A segment could not be cut into filaments.
    #[error("segment {segment}: {source}")]
    Discretize {
        /// Index of the offending segment.
        segment: usize,
        /// The underlying error.
        source: DiscretizeError,
    },
    /// A partial inductance could not be evaluated; the indices in the error
    /// are filament (branch) indices.
    #[error(transparent)]
    Kernel(#[from] KernelError),
    /// The ports or the discretization do not fit the geometry.
    #[error(transparent)]
    Mesh(#[from] MeshError),
    /// A [`Grading`] or [`SkinDepthGrading`] parameter is out of range.
    #[error("invalid grading: {reason}")]
    InvalidGrading {
        /// Which parameter was wrong, and what it was.
        reason: String,
    },
    /// The [`Coupling`] does not fit the geometry.
    #[error(transparent)]
    Coupling(#[from] CouplingError),
    /// A frequency is negative or not finite.
    #[error("frequency {index} is {value} Hz; frequencies must be finite and non-negative")]
    InvalidFrequency {
        /// Position in the frequency list.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The mesh system could not be solved at this frequency: the
    /// factorization hit a zero pivot or produced a non-finite impedance.
    /// A valid geometry never does; see the [module documentation](self).
    #[error("the mesh system is singular or ill-conditioned at {frequency} Hz")]
    Singular {
        /// The frequency, in hertz.
        frequency: f64,
    },
    /// The precorrected-FFT operator of an
    /// [`IterativeSystem`](crate::iterative::IterativeSystem) could not be
    /// built.
    #[error(transparent)]
    Pfft(#[from] PfftError),
    /// GMRES did not reach its tolerance within its iteration budget in an
    /// [`IterativeSystem`](crate::iterative::IterativeSystem) solve.
    #[error(
        "GMRES did not converge at {frequency} Hz for port {port}: relative residual \
         {relative_residual:e} after {iterations} iterations"
    )]
    NotConverged {
        /// The frequency, in hertz.
        frequency: f64,
        /// Index of the port whose right-hand side failed to converge.
        port: usize,
        /// Products with `Z_ee` spent.
        iterations: usize,
        /// The relative residual reached.
        relative_residual: f64,
    },
}

/// The assembled, frequency-independent mesh system of a geometry and its
/// ports. Assemble once, then evaluate [`impedance`](Self::impedance) or
/// [`sweep`](Self::sweep) at as many frequencies as needed.
#[derive(Clone, Debug)]
pub struct MeshSystem {
    ports: Vec<Port>,
    filaments: Vec<Filament>,
    resistances: Vec<f64>,
    mesh: MeshMatrix,
    /// `M R Mᵀ` and `M L Mᵀ`, partitioned into internal (`e`) and port (`p`)
    /// loops. The `pe` blocks are the transposes of the `ep` blocks.
    r_ee: DMatrix<f64>,
    r_ep: DMatrix<f64>,
    r_pp: DMatrix<f64>,
    l_ee: DMatrix<f64>,
    l_ep: DMatrix<f64>,
    l_pp: DMatrix<f64>,
    counts: Counts,
    truncation_warnings: Vec<TruncationWarning>,
    inductance_seconds: f64,
    assembly_seconds: f64,
}

impl MeshSystem {
    /// Discretizes `geometry`, evaluates every filament's resistance
    /// `l / (σ·A)` and the partial-inductance matrix, builds the loop basis
    /// for `ports`, and reduces both to mesh quantities.
    ///
    /// # Errors
    ///
    /// * [`SolveError::Discretize`] for a zero `nw` or `nh`, or graded cells
    ///   outside [`discretize_graded`](crate::filament::discretize_graded)'s numerical limits;
    /// * [`SolveError::InvalidGrading`] for an out-of-range grading
    ///   parameter;
    /// * [`SolveError::Mesh`] for ports that do not fit the geometry (none
    ///   given, unknown or repeated node, terminals on different conductors)
    ///   or a per-segment discretization of the wrong length;
    /// * [`SolveError::Kernel`] from the inductance kernels, which can only
    ///   happen for overlapping parallel segments whose cross-sections are
    ///   rotated against each other.
    pub fn assemble(
        geometry: &Geometry,
        ports: &[Port],
        discretization: &Discretization,
    ) -> Result<Self, SolveError> {
        Self::assemble_with_coupling(geometry, ports, discretization, &Coupling::all_pairs())
    }

    /// Builds the mesh system from a precomputed branch-resistance
    /// matrix `r`, partial-inductance matrix `l`, and loop basis `mesh` -
    /// the matrix-injection path (`peec-solve::FasterhenryMatrixSolver`).
    ///
    /// The `branches` carry the `Filament`s for the per-filament `R`
    /// accessor + the `mesh`'s loop basis (built via
    /// [`MeshMatrix::from_branches`]). The `counts` is **mandatory**:
    /// the matrix path has no `Geometry` source for
    /// `Counts.nodes` / `Counts.segments` (in the matrix path
    /// `Counts.segments == branch_count`; see the peec-solve audit D-3
    /// + BP#1 fold).
    ///
    /// `r` and `l` are **branch** matrices (n x n where n = `branches.len()`).
    /// They are reduced to mesh quantities internally via the existing
    /// [`congruence`] helper - pass branch `R` / branch `L`, NOT
    /// pre-reduced mesh blocks (sweep consumes the pre-reduced
    /// `r_ee..l_pp`).
    ///
    /// # Panics
    ///
    /// In debug builds if `mesh.branch_count() != branches.len()`, or if
    /// `r` / `l`'s shape doesn't match `branches.len()`.
    pub fn from_matrices(
        branches: Vec<Filament>,
        r: DMatrix<f64>,
        l: DMatrix<f64>,
        mesh: MeshMatrix,
        ports: Vec<Port>,
        counts: Counts,
    ) -> Result<Self, SolveError> {
        let n = branches.len();
        debug_assert_eq!(
            mesh.branch_count(),
            n,
            "mesh.branch_count must match branches.len (matrix path)"
        );
        debug_assert_eq!(r.nrows(), n, "r must be n x n where n = branches.len");
        debug_assert_eq!(r.ncols(), n, "r must be n x n where n = branches.len");
        debug_assert_eq!(l.nrows(), n, "l must be n x n where n = branches.len");
        debug_assert_eq!(l.ncols(), n, "l must be n x n where n = branches.len");

        // Per-filament R from the diagonal of `r` (the off-diagonal
        // entries of `r` are not used by `congruence`; we surface them
        // through `filament_resistances()` for the accessor).
        let resistances: Vec<f64> = (0..n).map(|i| r[(i, i)]).collect();

        // Reduce the branch matrices to mesh quantities via the existing
        // `congruence` helper.
        let r_mesh = congruence(&mesh, n, |branch, sign, row| {
            row[branch] += sign * resistances[branch];
        });
        let l_mesh = congruence(&mesh, n, |branch, sign, row| {
            for (slot, value) in row.iter_mut().zip(l.column(branch).iter()) {
                *slot += sign * value;
            }
        });

        let (internal, port_count) = (mesh.internal_count(), mesh.port_count());
        let block = |m: &DMatrix<f64>, r0, c0, nr, nc| m.view((r0, c0), (nr, nc)).into_owned();

        Ok(Self {
            ports,
            filaments: branches,
            resistances,
            mesh,
            r_ee: block(&r_mesh, 0, 0, internal, internal),
            r_ep: block(&r_mesh, 0, internal, internal, port_count),
            r_pp: block(&r_mesh, internal, internal, port_count, port_count),
            l_ee: block(&l_mesh, 0, 0, internal, internal),
            l_ep: block(&l_mesh, 0, internal, internal, port_count),
            l_pp: block(&l_mesh, internal, internal, port_count, port_count),
            counts,
            truncation_warnings: Vec::new(),
            inductance_seconds: 0.0,
            assembly_seconds: 0.0,
        })
    }

    /// [`assemble`](Self::assemble) with the mutual inductance of uncoupled
    /// conductor groups truncated to zero.
    ///
    /// Pairs of filaments whose segments sit in two groups the [`Coupling`]
    /// does not couple are never handed to a kernel, which is where the
    /// saving is. [`Coupling::all_pairs`] — what [`assemble`](Self::assemble)
    /// passes — truncates nothing and costs nothing. Groups that are close
    /// enough for the approximation to be doubtful are reported by
    /// [`truncation_warnings`](Self::truncation_warnings); see
    /// [`crate::coupling`] for when it is safe.
    ///
    /// # Errors
    ///
    /// As [`assemble`](Self::assemble), plus [`SolveError::Coupling`] when
    /// the coupling does not fit the geometry.
    pub fn assemble_with_coupling(
        geometry: &Geometry,
        ports: &[Port],
        discretization: &Discretization,
        coupling: &Coupling,
    ) -> Result<Self, SolveError> {
        let start = Instant::now();
        // Validate the coupling against the geometry before paying for
        // anything, so a typo in a group name costs nothing.
        let truncation_warnings = coupling.truncation_warnings(geometry)?;
        let (filaments, per_segment) = discretize_geometry(geometry, discretization)?;
        // Validate the ports before paying for the inductance matrix.
        let mesh = MeshMatrix::build(geometry, &per_segment, ports)?;

        let resistances: Vec<f64> = filaments.iter().map(filament_resistance).collect();

        let inductance_start = Instant::now();
        let inductance = if coupling.is_all_pairs() {
            partial_inductance_matrix(&filaments)?
        } else {
            partial_inductance_matrix_masked(&filaments, &coupling.pair_mask(&per_segment)?)?
        };
        let inductance_seconds = inductance_start.elapsed().as_secs_f64();

        let branches = filaments.len();
        let r_mesh = congruence(&mesh, branches, |branch, sign, row| {
            row[branch] += sign * resistances[branch];
        });
        let l_mesh = congruence(&mesh, branches, |branch, sign, row| {
            // `L` is symmetric: its column `branch` is its row `branch`.
            for (slot, value) in row.iter_mut().zip(inductance.column(branch).iter()) {
                *slot += sign * value;
            }
        });
        drop(inductance);

        let (internal, port_count) = (mesh.internal_count(), mesh.port_count());
        let block = |m: &DMatrix<f64>, r0, c0, nr, nc| m.view((r0, c0), (nr, nc)).into_owned();
        let counts = Counts {
            nodes: geometry.nodes().len(),
            segments: geometry.segment_count(),
            filaments: branches,
            meshes: mesh.loop_count(),
            internal_meshes: internal,
            ports: port_count,
        };
        Ok(Self {
            ports: ports.to_vec(),
            filaments,
            resistances,
            r_ee: block(&r_mesh, 0, 0, internal, internal),
            r_ep: block(&r_mesh, 0, internal, internal, port_count),
            r_pp: block(&r_mesh, internal, internal, port_count, port_count),
            l_ee: block(&l_mesh, 0, 0, internal, internal),
            l_ep: block(&l_mesh, 0, internal, internal, port_count),
            l_pp: block(&l_mesh, internal, internal, port_count, port_count),
            mesh,
            counts,
            truncation_warnings,
            inductance_seconds,
            assembly_seconds: start.elapsed().as_secs_f64(),
        })
    }

    /// The ports, in the order of the rows and columns of `Z`.
    pub fn ports(&self) -> &[Port] {
        &self.ports
    }

    /// Every filament (branch), segment by segment.
    pub fn filaments(&self) -> &[Filament] {
        &self.filaments
    }

    /// DC resistance `l / (σ·A)` of every filament, in ohms, in the order of
    /// [`filaments`](Self::filaments).
    pub fn filament_resistances(&self) -> &[f64] {
        &self.resistances
    }

    /// The loop basis.
    pub fn mesh(&self) -> &MeshMatrix {
        &self.mesh
    }

    /// Problem size: nodes, segments, filaments, meshes, ports.
    pub fn counts(&self) -> Counts {
        self.counts
    }

    /// Truncated group pairs that are too close for the approximation to be
    /// obviously justified — empty unless
    /// [`assemble_with_coupling`](Self::assemble_with_coupling) truncated
    /// something. Front ends should show these to the user.
    pub fn truncation_warnings(&self) -> &[TruncationWarning] {
        &self.truncation_warnings
    }

    /// The port impedance matrix `Z(ω)`, in ohms, at `frequency` hertz
    /// (`ω = 2πf`): entry `(i, j)` is the voltage across port `i` per unit
    /// current into port `j` with every other port open.
    ///
    /// # Errors
    ///
    /// [`SolveError::InvalidFrequency`] for a negative or non-finite
    /// frequency (reported with index 0), [`SolveError::Singular`] if the
    /// factorization breaks down.
    pub fn impedance(&self, frequency: f64) -> Result<DMatrix<Complex<f64>>, SolveError> {
        check_frequency(0, frequency)?;
        let singular = SolveError::Singular { frequency };
        let z = if frequency == 0.0 {
            schur_complement(self.r_ee.clone(), &self.r_ep, self.r_pp.clone())
                .ok_or(singular.clone())?
                .map(|re| Complex::new(re, 0.0))
        } else {
            let omega = std::f64::consts::TAU * frequency;
            let complex = |r: &DMatrix<f64>, l: &DMatrix<f64>| {
                r.zip_map(l, |r, l| Complex::new(r, omega * l))
            };
            schur_complement(
                complex(&self.r_ee, &self.l_ee),
                &complex(&self.r_ep, &self.l_ep),
                complex(&self.r_pp, &self.l_pp),
            )
            .ok_or(singular.clone())?
        };
        if z.iter()
            .all(|entry| entry.re.is_finite() && entry.im.is_finite())
        {
            Ok(z)
        } else {
            Err(singular)
        }
    }

    /// `Z(ω)` at every one of `frequencies` (hertz), with provenance.
    ///
    /// Frequencies are solved concurrently on the `rayon` thread pool, as many
    /// at a time as fit a fixed memory budget; every frequency is an
    /// independent deterministic computation, so the impedances do not depend
    /// on the number of threads.
    ///
    /// # Errors
    ///
    /// As [`impedance`](Self::impedance); the first failing frequency in list
    /// order is reported.
    pub fn sweep(&self, frequencies: &[f64]) -> Result<SweepResult, SolveError> {
        let start = Instant::now();
        for (index, &frequency) in frequencies.iter().enumerate() {
            check_frequency(index, frequency)?;
        }
        let internal = self.counts.internal_meshes;
        // The matrix being factorized plus its right-hand sides and scratch.
        let bytes_per_solve = 2 * internal * (internal + self.counts.ports) * 16;
        let concurrent = (SWEEP_MEMORY_BUDGET_BYTES / bytes_per_solve.max(1)).max(1);

        let mut impedance = Vec::with_capacity(frequencies.len());
        for chunk in frequencies.chunks(concurrent) {
            let solved: Vec<_> = chunk.par_iter().map(|&f| self.impedance(f)).collect();
            for z in solved {
                impedance.push(z?);
            }
        }
        let solve_seconds = start.elapsed().as_secs_f64();
        Ok(SweepResult {
            schema_version: SweepResult::SCHEMA_VERSION,
            ports: self.ports.clone(),
            frequencies_hz: frequencies.to_vec(),
            impedance_ohm: impedance,
            provenance: Provenance {
                fasterhenry_version: crate::VERSION.to_owned(),
                counts: self.counts,
                timing: Some(Timing {
                    threads: rayon::current_num_threads(),
                    inductance_s: self.inductance_seconds,
                    assembly_s: self.assembly_seconds,
                    solve_s: solve_seconds,
                    total_s: self.assembly_seconds + solve_seconds,
                }),
            },
        })
    }
}

/// Extracts the port impedance matrix of `geometry` at every one of
/// `frequencies` (hertz): [`MeshSystem::assemble`] followed by
/// [`MeshSystem::sweep`].
///
/// # Errors
///
/// As those two.
///
/// # Example
///
/// ```
/// use fasterhenry::{solve, Discretization, Geometry, Node, Port, SegmentDef};
///
/// // A 10 mm × 1 mm × 35 µm copper trace, 5 × 2 filaments.
/// let mut geometry = Geometry::new();
/// let a = geometry.add_node(Node::new(0.0, 0.0, 0.0))?;
/// let b = geometry.add_node(Node::new(10e-3, 0.0, 0.0))?;
/// geometry.add_segment(SegmentDef::new(a, b, 1e-3, 35e-6, 5.8e7))?;
///
/// let result = solve(
///     &geometry,
///     &[Port::new(a, b)],
///     &Discretization::uniform(5, 2),
///     &[0.0, 1e6, 1e9],
/// )?;
///
/// // At DC the trace is its resistance l / (σ·w·h) …
/// let dc = result.impedance_ohm[0][(0, 0)];
/// assert!((dc.re - 10e-3 / (5.8e7 * 1e-3 * 35e-6)).abs() < 1e-12 * dc.re);
/// assert_eq!(dc.im, 0.0);
/// // … and the skin effect raises it with frequency.
/// assert!(result.impedance_ohm[2][(0, 0)].re > dc.re);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn solve(
    geometry: &Geometry,
    ports: &[Port],
    discretization: &Discretization,
    frequencies: &[f64],
) -> Result<SweepResult, SolveError> {
    MeshSystem::assemble(geometry, ports, discretization)?.sweep(frequencies)
}

/// DC resistance of a filament, `R = l / (σ·A)`, in ohms.
pub fn filament_resistance(filament: &Filament) -> f64 {
    filament.length() / (filament.sigma() * filament.area())
}

/// Every segment of `geometry` cut into filaments, segment by segment, with
/// the number of filaments of each segment.
pub(crate) fn discretize_geometry(
    geometry: &Geometry,
    discretization: &Discretization,
) -> Result<(Vec<Filament>, Vec<usize>), SolveError> {
    let grids = discretization.resolve(geometry)?;
    let mut filaments = Vec::new();
    let mut per_segment = Vec::with_capacity(grids.len());
    for (index, (segment, grid)) in geometry.segments().zip(&grids).enumerate() {
        let bundle = match discretization {
            Discretization::Uniform(_) | Discretization::PerSegment(_) => {
                discretize(&segment, grid.nw, grid.nh)
            }
            _ => discretize_graded_per_axis(
                &segment,
                grid.nw,
                grid.nh,
                grid.width_ratio,
                grid.height_ratio,
            ),
        }
        .map_err(|source| SolveError::Discretize {
            segment: index,
            source,
        })?;
        per_segment.push(bundle.len());
        filaments.extend(bundle);
    }
    Ok((filaments, per_segment))
}

pub(crate) fn check_frequency(index: usize, value: f64) -> Result<(), SolveError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(SolveError::InvalidFrequency { index, value })
    }
}

/// `pp − epᵀ · ee⁻¹ · ep` by LU factorization with partial pivoting; `None`
/// if `ee` is exactly singular. The transpose is plain, not conjugated: the
/// mesh impedance matrix is complex *symmetric*.
fn schur_complement<T: ComplexField + Copy + Send + Sync>(
    ee: DMatrix<T>,
    ep: &DMatrix<T>,
    pp: DMatrix<T>,
) -> Option<DMatrix<T>> {
    if ee.is_empty() {
        return Some(pp);
    }
    let solved = lu_solve(ee, ep)?;
    Some(pp - ep.transpose() * solved)
}

/// The congruence `M · B · Mᵀ` of a symmetric branch matrix `B` with the loop
/// basis. `accumulate(branch, sign, row)` must add `sign ·` (row `branch` of
/// `B`) to `row`.
///
/// Only the upper triangle is computed and then mirrored, so the result is
/// symmetric to the last bit.
fn congruence(
    mesh: &MeshMatrix,
    branches: usize,
    accumulate: impl Fn(usize, f64, &mut [f64]) + Sync,
) -> DMatrix<f64> {
    let loops = mesh.loop_count();
    if loops == 0 || branches == 0 {
        return DMatrix::zeros(loops, loops);
    }
    // Row `i` of `M · B`, then its products with rows `j ≥ i` of `M`.
    let mut upper = vec![0.0; loops * loops];
    upper.par_chunks_mut(loops).enumerate().for_each_init(
        || vec![0.0; branches],
        |product, (i, out)| {
            product.fill(0.0);
            for &(branch, sign) in mesh.row(i) {
                accumulate(branch, sign, product);
            }
            for (j, slot) in out.iter_mut().enumerate().skip(i) {
                *slot = mesh
                    .row(j)
                    .iter()
                    .map(|&(branch, sign)| sign * product[branch])
                    .sum();
            }
        },
    );
    DMatrix::from_fn(loops, loops, |i, j| upper[i.min(j) * loops + i.max(j)])
}

#[cfg(test)]
mod solver_choice_tests {
    use super::*;

    #[test]
    fn auto_switches_paths_at_the_threshold() {
        let auto = SolverChoice::default();
        assert_eq!(auto, SolverChoice::Auto);
        assert_eq!(auto.resolve(0), Solver::Dense);
        assert_eq!(auto.resolve(DENSE_PATH_MAX_FILAMENTS), Solver::Dense);
        assert_eq!(
            auto.resolve(DENSE_PATH_MAX_FILAMENTS + 1),
            Solver::Iterative
        );
    }

    #[test]
    fn an_explicit_choice_ignores_the_threshold() {
        for filaments in [1, DENSE_PATH_MAX_FILAMENTS, 10 * DENSE_PATH_MAX_FILAMENTS] {
            assert_eq!(SolverChoice::Dense.resolve(filaments), Solver::Dense);
            assert_eq!(
                SolverChoice::Iterative.resolve(filaments),
                Solver::Iterative
            );
        }
    }

    #[test]
    fn paths_print_as_their_command_line_names() {
        assert_eq!(Solver::Dense.to_string(), "dense");
        assert_eq!(Solver::Iterative.to_string(), "iterative");
    }
}

#[cfg(test)]
mod bridge_tests {
    use super::*;
    use crate::geometry::{Geometry, Node, NodeId, SegmentDef};
    use crate::mesh::Port;

    /// Trace in parallel with a via–plane–via return path: the topology
    /// that caught the loop-partition bug. Plain resistors at DC; the
    /// answer is the parallel combination, independent of the forest.
    #[test]
    fn bridge_topology_dc_parallel() {
        let sigma = 5.8e7;
        let (w, t) = (0.2e-3, 35e-6);
        let mut g = Geometry::new();
        // Node/segment order matters for the forest: plane first (the
        // ground-plane builder's order), trace second, vias last.
        let n1 = NodeId(g.add_node(Node::new(2.5e-3, 0.0, -t / 2.0)).unwrap().0);
        let n2 = NodeId(g.add_node(Node::new(7.5e-3, 0.0, -t / 2.0)).unwrap().0);
        let a = NodeId(g.add_node(Node::new(2.5e-3, 0.0, 0.5e-3)).unwrap().0);
        let b = NodeId(g.add_node(Node::new(7.5e-3, 0.0, 0.5e-3)).unwrap().0);
        g.add_segment(SegmentDef::new(n1, n2, 2.0e-3, t, sigma))
            .unwrap(); // plane bar
        g.add_segment(SegmentDef::new(a, b, w, t, sigma)).unwrap(); // trace
        g.add_segment(SegmentDef::new(a, n1, w, t, sigma)).unwrap(); // via a
        g.add_segment(SegmentDef::new(b, n2, w, t, sigma)).unwrap(); // via b
        let ports = vec![Port::new(a, b)];
        let d = Discretization::PerSegment(vec![Subdivision::SINGLE; 4]);
        let system = MeshSystem::assemble(&g, &ports, &d).unwrap();
        println!(
            "r_ee {}\nr_ep {}\nr_pp {}",
            system.r_ee, system.r_ep, system.r_pp
        );
        let r_trace = 5.0e-3 / (sigma * w * t);
        let r_path = 2.0 * 0.5175e-3 / (sigma * w * t) + 5.0e-3 / (sigma * 2.0e-3 * t);
        let parallel = r_trace * r_path / (r_trace + r_path);
        let z = system.impedance(0.0).unwrap();
        println!(
            "Z = {:.9e}, parallel {parallel:.9e}, trace {r_trace:.9e}",
            z[(0, 0)].re
        );
        assert!(
            (z[(0, 0)].re - parallel).abs() < 1e-6 * parallel,
            "bridge DC {} vs parallel {}",
            z[(0, 0)].re,
            parallel
        );
    }
}
