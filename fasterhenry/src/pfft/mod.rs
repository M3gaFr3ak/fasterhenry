//! Matrix-free partial-inductance products by the precorrected FFT (pFFT)
//! method.
//!
//! [`PfftOperator`] evaluates `y = L·x` for the partial-inductance matrix
//! `L` of a set of filaments — the matrix
//! [`partial_inductance_matrix`](crate::partial_inductance_matrix) builds
//! densely — in memory and time that grow as `O(n)` plus an
//! `O(G log G)` FFT over a grid of `G ∝ n` points, instead of `O(n²)`.
//! The method is the precorrected FFT of Phillips & White (*IEEE T-CAD* 16,
//! 1997), which accelerates the same `1/r` kernel that the multipole method
//! of Kamon, Tsuk & White (*IEEE T-MTT* 42, 1994) does; this is a clean-room
//! implementation from those papers.
//!
//! # The four pieces
//!
//! The partial inductance of two filaments is
//! `Lᵢⱼ = μ0/(4π) · (l̂ᵢ·l̂ⱼ) · (1/AᵢAⱼ) ∫_{Vᵢ}∫_{Vⱼ} dV dV'/|r − r'|`, so
//! `y = L·x` splits into three scalar potential problems, one per Cartesian
//! component `c` of the current direction:
//!
//! 1. **Grid.** A uniform grid of spacing `h` covers the filaments'
//!    bounding box (see [`GridSpacing`]).
//! 2. **Projection.** Each filament's uniform current density is replaced
//!    by weights `Wᵢ_g` on nearby grid points: a Gauss–Legendre rule over
//!    the filament's volume, each node interpolated onto its
//!    `(order + 1)³` Lagrange stencil (see
//!    [`PfftParams::interpolation_order`]). The grid sources are
//!    `q_c(g) = Σᵢ Wᵢ_g l̂ᵢ_c xᵢ`.
//! 3. **Far field.** The grid potentials `φ_c = K ⊛ q_c`, with
//!    `K(g − h) = 1/|r_g − r_h|`, are an aperiodic convolution done by
//!    zero-padded 3-D FFT (two real components per complex transform), and
//!    are interpolated back: `yᵢ ≈ μ0/(4π) Σ_c l̂ᵢ_c Σ_g Wᵢ_g φ_c(g)`.
//! 4. **Near-field precorrection.** For filament pairs closer than the
//!    near-field radius (see [`PfftParams::near_field_radius`]) the grid
//!    value is wrong — the interpolation cannot resolve `1/r` at short
//!    range. Their grid contribution is computed exactly as the FFT would
//!    and subtracted, and the exact partial inductance from
//!    [`mutual_inductance`](crate::mutual_inductance) is added instead, as
//!    one sparse matrix of corrections.
//!
//! Near pairs, including every self term, are therefore exactly as accurate
//! as the dense matrix; only well-separated pairs are approximated, with an
//! error that falls off as `(h/d)^(order+1)` with their separation `d`.
//!
//! # Accuracy and cost
//!
//! The three parameters of [`PfftParams`] trade accuracy against cost:
//!
//! | Parameter | Larger value | Accuracy | Cost |
//! |-----------|--------------|----------|------|
//! | [`interpolation_order`](PfftParams::interpolation_order) `p` | wider stencils | far-field error falls as `(h/d)^(p+1)` | projection weights and set-up grow as `(p+1)³`; the near-field radius must be at least the stencil width `p + 1` |
//! | [`near_field_radius`](PfftParams::near_field_radius) `k` (in cells) | more exact pairs | the nearest approximated pair is `k·h` away, so the error falls as `k^−(p+1)` | near pairs grow as `k³`, each costing one exact kernel evaluation at set-up and one multiply-add per product |
//! | grid density ([`GridSpacing`]) | smaller `h` | none at fixed `k`: accuracy depends on `h/d`, and `k` is in units of `h` | fewer near pairs and a cheaper set-up, a larger FFT per product |
//!
//! Measured on 1 000 random filaments (`parameter_study` in
//! `fasterhenry/tests/pfft.rs`), the norm-wise relative error of `L·x`
//! against the dense product is essentially independent of the grid
//! density and falls with `k` and `p`:
//!
//! | `k` \ `p` | 2 | 3 | 4 | 5 |
//! |-----------|------|------|------|------|
//! | 4 | 2e-4 | 1e-4 | – | – |
//! | 6 | 7e-5 | 3e-5 | 4e-6 | 1.4e-6 |
//! | 8 | 3e-5 | 1e-5 | 1e-6 | 2.5e-7 |
//!
//! The [`Default`] (`p = 5`, `k = 8`, 8 grid cells per filament) is the
//! cheapest setting in that table that meets the `1e-6` accuracy target
//! with a margin; the crate's tests check `1e-6` on random sets, the spiral
//! and coupled-structure fixtures, and flat and line-like geometries. For a
//! preconditioner or a first iterate, `p = 3`, `k = 6` is about three times
//! cheaper to set up at `3e-5`.
//!
//! Memory and each [`apply`](PfftOperator::apply) grow linearly with the
//! number of filaments at a fixed filament density: the projection and the
//! near field hold a bounded number of entries per filament, and the grid
//! has a fixed number of cells per filament. One caveat: the projection of
//! a filament is proportional to its length in grid cells, so a set with a
//! few filaments far longer than the rest drives the automatic grid spacing
//! down and the projection up; such filaments are best cut into shorter
//! pieces first.
//!
//! # Set-up cost
//!
//! [`new`](PfftOperator::new) is `O(n)` (issue #42), but its constant is
//! high next to one [`apply`](PfftOperator::apply) -- at 10 000 filaments
//! and the default parameters, set-up (about 20 s on a Linux CI-class host,
//! `RAYON_NUM_THREADS=4`) costs about as much as 300 products (each about
//! 65 ms; `cargo bench -p fasterhenry --bench pfft`). A phase-by-phase
//! profile at that size
//! (`cargo test --release -p fasterhenry --lib pfft::near::profile --
//! --ignored --nocapture`, issue #51) attributes it as:
//!
//! | Phase | Share of set-up |
//! |-------|-----------------|
//! | Projection (`grid::project_all`) | ~10% |
//! | Near-pair search (`near::near_pairs`) | ~5% |
//! | Precorrection: grid precorrection (`near::grid_interactions`) | ~75% |
//! | Precorrection: exact kernel (`inductance::evaluate`) | ~10% |
//! | FFT planning (`Convolver::new`) | <1% |
//!
//! Grid precorrection -- the local box potential summed per near pair to
//! subtract the grid's (wrong, at that range) approximation -- dominates,
//! by roughly 7-8× over the exact kernel evaluation it is paired with; the
//! #50 review's guess that the two split the cost evenly does not hold at
//! these parameters. `near::grid_interactions` already picks the cheaper of
//! two evaluations per row (a direct double sum, or a local dense
//! accumulation over the box spanned by a filament's near partners) and the
//! accumulation itself is a SIMD multiply-add (`near::axpy`, four lanes),
//! which is about 10-15% cheaper in aggregate set-up time than the
//! equivalent scalar loop -- measured with `cargo bench -p fasterhenry
//! --bench pfft -- --quick pfft_setup/10000` on the same host, both before
//! and after, since the effect is within run-to-run noise on a single
//! sample and needs the comparison.
//!
//! Two larger levers were considered and set aside for now, not because
//! they are wrong but because neither clearly pays off at the sizes and
//! parameters this crate ships by default:
//!
//! - **A small FFT for the local box potential**, in place of the direct
//!   accumulation. At the default `k = 8`, `p = 5`, a row's box is only
//!   about 20-25 cells per axis; an aperiodic (non-wrapping) FFT convolution
//!   needs the box roughly doubled per axis to pad against wraparound, so
//!   the padded volume is about 8× the box itself. At that scale the FFT's
//!   `O(N log N)` no longer clearly beats the current `O(source × box)`
//!   direct sum -- the two were within a small constant factor of each
//!   other by estimate. It would pay off at a much larger near-field
//!   radius or interpolation order than the defaults use.
//! - **Reusing precorrection entries across translated identical filament
//!   pairs.** `fasterhenry/benches/pfft.rs`'s filament clouds (and the
//!   crate's other accuracy fixtures) have continuously-varying lengths and
//!   cross-sections by construction, so no pair repeats exactly and a cache
//!   keyed on canonical pair geometry would never hit on them. It would
//!   help a literal regular mesh (e.g. a uniformly discretized ground
//!   plane) but that is not what the shipped benchmark measures, and adding
//!   a geometry-canonicalization cache is a real correctness surface
//!   (floating-point key stability) for an unverified win; left for a
//!   follow-up if a real workload shows the repetition.
//!
//! Given set-up is already `O(n)`, and [`PfftParams`] exposes the exact
//! knobs (grid density, `near_field_radius`, `interpolation_order`) that
//! trade set-up cost against accuracy directly (see the table above), the
//! existing cost is considered acceptable for #43 (GMRES needs one set-up
//! per solve, amortized over many products) and for the dense-vs-iterative
//! threshold of #44.
//!
//! # Scope
//!
//! This is the operator alone. [`IterativeSystem`](crate::IterativeSystem)
//! solves for the port impedance with GMRES on top of it; the default
//! [`MeshSystem`](crate::MeshSystem) still assembles `L` densely.

mod fft;
mod grid;
pub(crate) mod near;

pub(crate) use near::near_pairs_wanted;

use nalgebra::Vector3;
use rayon::prelude::*;
use thiserror::Error;

use crate::filament::Filament;
use crate::inductance::{KernelError, MU0_OVER_4PI};
use fft::Convolver;
use grid::Grid;
use near::NearField;

/// Largest padded FFT grid the operator will allocate, in points (each a
/// 16-byte complex number, several buffers of which exist at once). 2^38 =
/// 5.4e8 pts: ~8.6 GB per spectrum-class array, 2-3 of those at once is
/// 17-26 GB - sized to fit a 59 GB box with the near-field estimate on top;
/// the local 15 GB box refuses the big grids loudly by this cap regardless.
pub const MAX_FFT_POINTS: usize = 1 << 29;

/// Filaments per parallel task in the per-filament passes of
/// [`PfftOperator::apply`].
const MIN_ROWS_PER_TASK: usize = 64;

/// How the grid spacing `h` is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GridSpacing {
    /// `h` such that the filaments' bounding box holds about
    /// `cells_per_filament · n` grid cells (axes thinner than `h` count as
    /// one cell). The near-field radius being fixed in cells, this keeps the
    /// number of near pairs per filament roughly constant as `n` grows,
    /// which is what makes the operator scale as `O(n)`. Larger values mean
    /// a finer grid: fewer near pairs, a larger FFT.
    Auto {
        /// Target grid cells per filament.
        cells_per_filament: f64,
    },
    /// A fixed spacing, in metres.
    Fixed(f64),
}

/// The accuracy-versus-cost parameters of a [`PfftOperator`]. See the
/// [module documentation](self) for the trade-off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PfftParams {
    /// Grid spacing. Default: [`GridSpacing::Auto`] with
    /// [`DEFAULT_CELLS_PER_FILAMENT`].
    pub grid_spacing: GridSpacing,
    /// Radius, in grid spacings, below which a pair of filaments is
    /// evaluated exactly: pairs whose volumes are closer than
    /// `near_field_radius · h` are precorrected. It must be at least
    /// `interpolation_order + 1`, the width of a stencil, for the far field
    /// to converge at all. Default: [`DEFAULT_NEAR_FIELD_RADIUS`].
    pub near_field_radius: f64,
    /// Degree `p` of the Lagrange interpolation onto the grid, `p + 1`
    /// points per axis (`1` is trilinear). Far-field error falls as
    /// `(h/d)^(p+1)`. Default: [`DEFAULT_INTERPOLATION_ORDER`].
    pub interpolation_order: usize,
    /// Whether the near band is precorrected (exact kernels stored per
    /// near pair). Default `true` = the historical operator. `false`
    /// builds the FAR-ONLY operator (no near pairs enumerated or
    /// stored at all: the apply keeps every pair's grid approximation,
    /// close ones included) - the operator for the conv-matvec
    /// composition, where the near band is owned by the conv model
    /// (see the increment-5 decision in the peec books). The engine's
    /// `FASTERHENRY_CONV_MATVEC` path answers for that flag; the pFFT
    /// crate never flips it itself.
    pub precorrect_near: bool,
}

/// Default `cells_per_filament` of [`GridSpacing::Auto`]. Accuracy hardly
/// depends on it (see the [module documentation](self)); 8 balances the
/// set-up cost of the near field against the size of the FFT.
pub const DEFAULT_CELLS_PER_FILAMENT: f64 = 8.0;
/// Default [`PfftParams::near_field_radius`], in grid spacings.
pub const DEFAULT_NEAR_FIELD_RADIUS: f64 = 8.0;
/// Default [`PfftParams::interpolation_order`]: quintic, 216-point stencils.
pub const DEFAULT_INTERPOLATION_ORDER: usize = 5;

impl Default for PfftParams {
    /// About `2.5e-7` relative error in `L·x` on random filament sets; see
    /// the [module documentation](self).
    fn default() -> Self {
        Self {
            grid_spacing: GridSpacing::Auto {
                cells_per_filament: DEFAULT_CELLS_PER_FILAMENT,
            },
            near_field_radius: DEFAULT_NEAR_FIELD_RADIUS,
            interpolation_order: DEFAULT_INTERPOLATION_ORDER,
            precorrect_near: true,
        }
    }
}

/// Why a [`PfftOperator`] could not be built.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum PfftError {
    /// A near-field pair's exact partial inductance could not be computed.
    #[error(transparent)]
    Kernel(#[from] KernelError),
    /// A parameter is out of range.
    #[error("invalid pFFT parameter: {0}")]
    InvalidParameter(&'static str),
    /// The grid would exceed [`MAX_FFT_POINTS`]; use a coarser spacing.
    #[error("pFFT grid of {points} padded points exceeds the limit of {limit}")]
    GridTooLarge {
        /// Padded FFT grid points the parameters asked for.
        points: usize,
        /// [`MAX_FFT_POINTS`].
        limit: usize,
    },
    /// The pre-allocation memory census exceeded
    /// `FASTERHENRY_MEMORY_BUDGET_GB`; nothing was allocated.
    #[error(
        "pfft memory census: projected ~{estimate_gb:.1} GB (FFT ~{fft_gb:.1} + near pairs ~{pairs_gb:.1}, {pairs_est:.3e} pairs estimated) exceeds the FASTERHENRY_MEMORY_BUDGET_GB={budget_gb:.0} GB budget; lower cpf / near radius, or raise the budget (resolution is fixture-calibrated and holds +-3x)"
    )]
    MemoryBudget {
        /// Projected census total, GB.
        estimate_gb: f64,
        /// Configured budget, GB.
        budget_gb: f64,
        /// FFT-half of the estimate, GB (deterministic).
        fft_gb: f64,
        /// Near-pair estimate half, GB (fixture-calibrated).
        pairs_gb: f64,
        /// Estimated directed near pairs.
        pairs_est: f64,
    },
}

/// Sizes of a [`PfftOperator`], for diagnostics and scaling measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PfftStats {
    /// Number of filaments.
    pub filaments: usize,
    /// Grid spacing `h`, in metres.
    pub grid_spacing: f64,
    /// Grid points along each axis.
    pub grid_dims: [usize; 3],
    /// Zero-padded FFT grid points along each axis.
    pub fft_dims: [usize; 3],
    /// Stored projection weights, over all filaments.
    pub projection_entries: usize,
    /// Stored near-field corrections (both triangles, self terms once).
    pub near_entries: usize,
    /// Approximate heap memory held by the operator, in bytes.
    pub memory_bytes: usize,
}

/// A matrix-free partial-inductance operator: [`apply`](Self::apply)
/// computes `L·x` without forming `L`. See the [module documentation](self).
///
/// # Example
///
/// ```
/// use fasterhenry::pfft::{PfftOperator, PfftParams};
/// use fasterhenry::{discretize, partial_inductance_matrix, Node, Segment};
///
/// let trace = Segment::new(
///     Node::new(0.0, 0.0, 0.0),
///     Node::new(1e-3, 0.0, 0.0),
///     20e-6,
///     10e-6,
///     5.8e7,
/// );
/// let filaments = discretize(&trace, 4, 2)?;
/// let op = PfftOperator::new(&filaments, &PfftParams::default())?;
/// let x = vec![1.0; filaments.len()];
/// let y = op.apply(&x);
/// let dense = partial_inductance_matrix(&filaments)? * nalgebra::DVector::from_vec(x);
/// for (a, b) in y.iter().zip(dense.iter()) {
///     assert!((a - b).abs() <= 1e-9 * b.abs());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PfftOperator {
    n: usize,
    grid: Option<Grid>,
    directions: Vec<[f64; 3]>,
    /// Projection weights, per filament (CSR).
    projection_offsets: Vec<usize>,
    projection_index: Vec<u32>,
    projection_weight: Vec<f64>,
    /// The same weights per grid point (the transpose, CSR), for a
    /// race-free parallel scatter.
    spread_offsets: Vec<usize>,
    spread_filament: Vec<u32>,
    spread_weight: Vec<f64>,
    /// Cartesian components carried by some filament's direction.
    components: Vec<usize>,
    near: NearField,
    convolver: Option<Convolver>,
    /// Near-field pairing radius in metres (`near_field_radius` grid cells
    /// times the grid spacing), so downstream set-up (the CUBEL pattern)
    /// can reuse the operator's own pairing.
    near_radius: f64,
}

impl PfftOperator {
    /// Builds the operator for `filaments`.
    ///
    /// Set-up evaluates the exact partial inductance of every near-field
    /// pair (in parallel), so it costs about as much as that many entries of
    /// the dense matrix; each [`apply`](Self::apply) afterwards is cheap.
    ///
    /// # Errors
    ///
    /// [`PfftError::InvalidParameter`] for a non-positive or non-finite
    /// spacing, a zero interpolation order, or a near-field radius smaller
    /// than the stencil; [`PfftError::GridTooLarge`] if the grid would exceed
    /// [`MAX_FFT_POINTS`]; [`PfftError::Kernel`] as for
    /// [`partial_inductance_matrix`](crate::partial_inductance_matrix).
    pub fn new(filaments: &[Filament], params: &PfftParams) -> Result<Self, PfftError> {
        let order = params.interpolation_order;
        if order == 0 {
            return Err(PfftError::InvalidParameter(
                "interpolation_order must be at least 1",
            ));
        }
        // Written so that NaN fails: it is neither finite nor comparable.
        let radius_ok =
            params.near_field_radius.is_finite() && params.near_field_radius >= (order + 1) as f64;
        if !radius_ok {
            return Err(PfftError::InvalidParameter(
                "near_field_radius must be finite and at least interpolation_order + 1",
            ));
        }
        match params.grid_spacing {
            GridSpacing::Auto { cells_per_filament }
                if !(cells_per_filament > 0.0 && cells_per_filament.is_finite()) =>
            {
                return Err(PfftError::InvalidParameter(
                    "cells_per_filament must be positive and finite",
                ));
            }
            GridSpacing::Fixed(h) if !(h > 0.0 && h.is_finite()) => {
                return Err(PfftError::InvalidParameter(
                    "grid spacing must be positive and finite",
                ));
            }
            _ => {}
        }
        let n = filaments.len();
        if n == 0 {
            return Ok(Self::empty());
        }

        let (lo, hi) = bounding_box(filaments);
        let spacing = match params.grid_spacing {
            GridSpacing::Fixed(h) => h,
            GridSpacing::Auto { cells_per_filament } => {
                auto_spacing(hi - lo, cells_per_filament * n as f64)
            }
        };
        let grid = Grid::new(lo, hi, spacing, order + 1);
        let padded: usize = grid.dims.iter().map(|&d| fft::padded_size(d)).product();
        if padded > MAX_FFT_POINTS || grid.len() > u32::MAX as usize {
            return Err(PfftError::GridTooLarge {
                points: padded,
                limit: MAX_FFT_POINTS,
            });
        }
        // Memory census before any big allocation. The FFT half is
        // deterministic from the dims (grid.len() and `padded` are already
        // exact).
        let fft_bytes = 3 * padded * 16;
        let fft_gb = fft_bytes as f64 / 1e9;
        eprintln!(
            "pfft: census: grid {:?}, padded {} pts, FFT ~{:.1} GB (3 x {:.1} GB spectra)",
            grid.dims,
            padded,
            fft_gb,
            fft_gb / 3.0,
        );
        // **[superseded 2026-10-19, ekart A/B lane - the whole
        // fixture-calibrated near-pair extrapolation
        // `pairs_per_fil ~ 35 x k^3 / cpf^1.5` never read the deck
        // (no volume, no density, no aspect); safe for its own
        // constant-density cloud generator, untrustworthy beyond it
        // (orders wrong either way on slab/pancake decks). The gate
        // below refuses on the TRUE directed-pair count instead
        // (near_pairs_count, O(n) memory, the near walk's bin
        // structure listed without materializing).]**
        // The census's near-field half: the TRUE directed-pair count
        // (the near walk's bin structure listed without materializing;
        // O(n) memory). Computed here so the budget gate refuses on
        // truth before the (potentially enormous) pair+scratch
        // allocations - the same mid-run refusal policy the OOM
        // governor enforces, moved to pre-flight where it is honest.
        // The far-only mode (`precorrect_near = false`) skips all of
        // this: its census total is the FFT half alone, and the near
        // band is owned downstream (the conv-matvec composition).
        let radius = params.near_field_radius * spacing;
        let (pairs_gb, census_gb, pairs_est) = if params.precorrect_near {
            let near_rows = near::near_pairs_count(filaments, radius);
            let pair_estimate: usize = near_rows.iter().map(|&c| c as usize).sum();
            let pair_bytes = 24usize.saturating_mul(pair_estimate);
            let pair_gb = pair_bytes as f64 / 1e9;
            let total_gb = fft_gb + pair_gb;
            eprintln!(
                "pfft: near-field census: {} rows, {} directed pairs, exact ({:.1} GB at 24 B/pair)",
                near_rows.len(),
                pair_estimate,
                pair_gb,
            );
            eprintln!(
                "pfft: census total: grid {:?}, padded {} pts, FFT ~{:.1} GB + near pairs ~{:.1} GB = ~{:.1} GB",
                grid.dims, padded, fft_gb, pair_gb, total_gb,
            );
            (pair_gb, total_gb, pair_estimate as f64)
        } else {
            eprintln!(
                "pfft: far-only operator (precorrect_near=false): census total = FFT ~{fft_gb:.1} GB (grid {:?}, padded {} pts); near band owned downstream (conv-matvec), no pair enumeration",
                grid.dims, padded,
            );
            (0.0, fft_gb, 0.0)
        };
        if let Ok(budget) = std::env::var("FASTERHENRY_MEMORY_BUDGET_GB") {
            let budget: f64 = budget.trim().parse::<f64>().map_err(|_| {
                PfftError::InvalidParameter(
                    "FASTERHENRY_MEMORY_BUDGET_GB must be a finite number of GB",
                )
            })?;
            if !(budget.is_finite() && budget > 0.0) {
                return Err(PfftError::InvalidParameter(
                    "FASTERHENRY_MEMORY_BUDGET_GB must be a finite number of GB",
                ));
            }
            let override_set = std::env::var("FASTERHENRY_CENSUS_OVERRIDE")
                .map(|v| v.trim() == "1")
                .unwrap_or(false);
            if census_gb > budget {
                if override_set {
                    eprintln!(
                        "conv: warning: FASTERHENRY_CENSUS_OVERRIDE=1: the census total ~{census_gb:.1} GB exceeds the FASTERHENRY_MEMORY_BUDGET_GB={budget:.1} GB budget - proceeding anyway on explicit override; the runtime governor still refuses at the wall (exit 134) and this row carries the caveat on record (the count is exact, not fixture-extrapolated).",
                    );
                } else {
                    return Err(PfftError::MemoryBudget {
                        estimate_gb: census_gb,
                        budget_gb: budget,
                        fft_gb,
                        pairs_gb,
                        pairs_est: pairs_est as f64,
                    });
                }
            }
        }
        eprintln!("pfft: projecting {} filaments to the grid ...", n);
        let projections = grid::project_all(&grid, filaments, radius);
        let near = if params.precorrect_near {
            eprintln!("pfft: enumerating near-field pairs ...");
            crate::memgate::set_stage(crate::memgate::STAGE_PFFT_OPERATOR);
            let pairs = near::near_pairs(filaments, radius);
            let pair_total: usize = pairs.iter().map(|row| row.len()).sum();
            eprintln!(
                "pfft: near-field census: {} neighbor rows, {} directed pairs, exact ({:.1} GB at 24 B/pair; budget {})",
                pairs.len(),
                pair_total,
                24.0 * pair_total as f64 / 1e9,
                std::env::var("FASTERHENRY_MEMORY_BUDGET_GB").unwrap_or_else(|_| "none".to_string()),
            );
            eprintln!("pfft: precorrecting near-field (exact kernels) ...");
            near::precorrect(filaments, &grid, &projections, &pairs)?
        } else {
            near::NearField {
                offsets: vec![0; n + 1],
                columns: Vec::new(),
                values: Vec::new(),
            }
        };

        let mut projection_offsets = Vec::with_capacity(n + 1);
        projection_offsets.push(0);
        let mut projection_index = Vec::new();
        let mut projection_weight = Vec::new();
        let mut counts = vec![0usize; grid.len()];
        for p in &projections {
            for &(g, w) in &p.entries {
                projection_index.push(g);
                projection_weight.push(w);
                counts[g as usize] += 1;
            }
            projection_offsets.push(projection_index.len());
        }
        let mut spread_offsets = Vec::with_capacity(grid.len() + 1);
        spread_offsets.push(0);
        for count in &counts {
            spread_offsets.push(spread_offsets.last().unwrap() + count);
        }
        let mut fill = spread_offsets[..grid.len()].to_vec();
        let mut spread_filament = vec![0u32; projection_index.len()];
        let mut spread_weight = vec![0.0; projection_index.len()];
        for (i, p) in projections.iter().enumerate() {
            for &(g, w) in &p.entries {
                let slot = &mut fill[g as usize];
                spread_filament[*slot] = i as u32;
                spread_weight[*slot] = w;
                *slot += 1;
            }
        }

        let directions: Vec<[f64; 3]> = filaments
            .iter()
            .map(|f| {
                let d = f.direction();
                [d.x, d.y, d.z]
            })
            .collect();
        let components = (0..3)
            .filter(|&c| directions.iter().any(|d| d[c] != 0.0))
            .collect();
        let h = grid.spacing;
        eprintln!("pfft: planning FFT convolver ...");
        let convolver = Convolver::new(grid.dims, |d| near::grid_kernel(d, h));

        Ok(Self {
            n,
            grid: Some(grid),
            directions,
            projection_offsets,
            projection_index,
            projection_weight,
            spread_offsets,
            spread_filament,
            spread_weight,
            components,
            near,
            convolver: Some(convolver),
            near_radius: radius,
        })
    }

    fn empty() -> Self {
        Self {
            n: 0,
            grid: None,
            directions: Vec::new(),
            projection_offsets: vec![0],
            projection_index: Vec::new(),
            projection_weight: Vec::new(),
            spread_offsets: vec![0],
            spread_filament: Vec::new(),
            spread_weight: Vec::new(),
            components: Vec::new(),
            near: NearField {
                offsets: vec![0],
                ..NearField::default()
            },
            convolver: None,
            near_radius: 0.0,
        }
    }

    /// The near-field pairing radius in metres (`near_field_radius` grid
    /// cells times the operator's own grid spacing).
    pub fn near_radius(&self) -> f64 {
        self.near_radius
    }

    /// The grid's valid placement box for a synthetic pair's centres:
    /// `Grid::new` pads the origin by one stencil half-width, so a
    /// filament centred inside `[origin + pad, hi - pad]` projects in
    /// full. Synthetic pairs sampled OUTSIDE (the difference kernel's
    /// negative displacements) would carry truncated stencils - silent
    /// garbage - so the samplers translate their pairs into this box.
    /// `None` when the operator carries no grid (far-off posture).
    pub(crate) fn grid_box(&self) -> Option<(Vector3<f64>, Vector3<f64>)> {
        let grid = self.grid.as_ref()?;
        let pad = grid.stencil.div_ceil(2) as f64 * grid.spacing;
        let hi = grid.origin
            + Vector3::new(
                (grid.dims[0] - 1) as f64 * grid.spacing,
                (grid.dims[1] - 1) as f64 * grid.spacing,
                (grid.dims[2] - 1) as f64 * grid.spacing,
            );
        Some((
            grid.origin + Vector3::repeat(pad),
            hi - Vector3::repeat(pad),
        ))
    }

    /// The grid's own per-pair estimate of the mutual inductance of two
    /// filaments: the closed-form double sum over both projection
    /// stencils, exactly the value the spread-convolve-gather path
    /// returns for the pair (linear superposition of the grid kernel
    /// `1/(h r)` over both stencils), evaluated without an FFT.
    ///
    /// This is the conv-matvec difference kernel's subtraction term:
    /// `Kdiff(u) = L_exact(synthetic pair at u) - pair_grid_estimate(same
    /// pair)`, and the tab/spine substitution stages' per-pair
    /// `negation` - see the increment-5 decision in the peec books.
    /// The grid's own per-pair estimate of the mutual inductance of two
    /// filaments: the closed-form double sum over both projection
    /// stencils, exactly the value the spread-convolve-gather path
    /// returns for the pair (linear superposition of the grid kernel
    /// `1/(h r)` over both stencils), evaluated without an FFT. The
    /// physics rides the same factors as the far path and the
    /// precorrection's `exact - MU0_OVER_4PI * cos * approx` split
    /// (`near.rs::row`): `MU0_OVER_4PI` times the direction cosine of
    /// the pair times the bare double sum.
    pub fn pair_grid_estimate(&self, src: &Filament, dst: &Filament) -> f64 {
        let Some(grid) = &self.grid else {
            return 0.0;
        };
        let pair = [src.clone(), dst.clone()];
        let projections = grid::project_all(grid, &pair, self.near_radius);
        let mut total = 0.0;
        for &(g, sw) in &projections[0].entries {
            let sg = grid.point(g as usize);
            for &(h, dw) in &projections[1].entries {
                let hg = grid.point(h as usize);
                let offset = [
                    hg[0] as i64 - sg[0] as i64,
                    hg[1] as i64 - sg[1] as i64,
                    hg[2] as i64 - sg[2] as i64,
                ];
                let r2 =
                    (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]) as f64;
                // Same-cell interactions carry zero grid kernel (the
                // near-stage kernel's own zero rule).
                if r2 == 0.0 {
                    continue;
                }
                total += sw * dw / (grid.spacing * r2.sqrt());
            }
        }
        let cos = src.direction().dot(&dst.direction());
        MU0_OVER_4PI * cos * total
    }

    /// Number of filaments, the dimension of `L`.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the operator acts on no filaments at all.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `y = L·x`, in henries times the units of `x`.
    ///
    /// # Panics
    ///
    /// If `x.len()` differs from [`len`](Self::len).
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(
            x.len(),
            self.n,
            "pFFT operator of dimension {} applied to a vector of length {}",
            self.n,
            x.len()
        );
        let mut y = self.near_field(x);
        let (Some(grid), Some(convolver)) = (&self.grid, &self.convolver) else {
            return y;
        };
        // Grid sources, one per active direction component, convolved two
        // at a time.
        let sources: Vec<Vec<f64>> = self
            .components
            .iter()
            .map(|&c| self.spread(grid.len(), x, c))
            .collect();
        let mut potentials: Vec<Vec<f64>> = Vec::with_capacity(sources.len());
        for pair in sources.chunks(2) {
            let (a, b) = convolver.convolve(&pair[0], pair.get(1).map(Vec::as_slice));
            potentials.push(a);
            if pair.len() == 2 {
                potentials.push(b);
            }
        }
        y.par_iter_mut()
            .enumerate()
            .with_min_len(MIN_ROWS_PER_TASK)
            .for_each(|(i, yi)| {
                let range = self.projection_offsets[i]..self.projection_offsets[i + 1];
                let mut far = 0.0;
                for (&c, phi) in self.components.iter().zip(&potentials) {
                    let mut sum = 0.0;
                    for (&g, &w) in self.projection_index[range.clone()]
                        .iter()
                        .zip(&self.projection_weight[range.clone()])
                    {
                        sum += w * phi[g as usize];
                    }
                    far += self.directions[i][c] * sum;
                }
                *yi += MU0_OVER_4PI * far;
            });
        y
    }

    /// `q_c(g) = Σᵢ Wᵢ_g l̂ᵢ_c xᵢ`.
    fn spread(&self, points: usize, x: &[f64], c: usize) -> Vec<f64> {
        let mut q = vec![0.0; points];
        q.par_iter_mut()
            .enumerate()
            .with_min_len(16 * MIN_ROWS_PER_TASK)
            .for_each(|(g, value)| {
                let range = self.spread_offsets[g]..self.spread_offsets[g + 1];
                *value = self.spread_filament[range.clone()]
                    .iter()
                    .zip(&self.spread_weight[range])
                    .map(|(&i, &w)| {
                        let i = i as usize;
                        w * self.directions[i][c] * x[i]
                    })
                    .sum();
            });
        q
    }

    /// The precorrection product `C·x`.
    fn near_field(&self, x: &[f64]) -> Vec<f64> {
        let near = &self.near;
        (0..self.n)
            .into_par_iter()
            .with_min_len(MIN_ROWS_PER_TASK)
            .map(|i| {
                let range = near.offsets[i]..near.offsets[i + 1];
                near.columns[range.clone()]
                    .iter()
                    .zip(&near.values[range])
                    .map(|(&j, &c)| c * x[j as usize])
                    .sum()
            })
            .collect()
    }

    /// Sizes of the operator.
    pub fn stats(&self) -> PfftStats {
        let (grid_spacing, grid_dims, grid_points) = self
            .grid
            .as_ref()
            .map_or((0.0, [0; 3], 0), |g| (g.spacing, g.dims, g.len()));
        let (fft_dims, fft_points) = self
            .convolver
            .as_ref()
            .map_or(([0; 3], 0), |c| (c.padded_dims(), c.padded_len()));
        let projection_entries = self.projection_index.len();
        let near_entries = self.near.columns.len();
        let usize_bytes = std::mem::size_of::<usize>();
        let memory_bytes = projection_entries * 2 * (4 + 8)
            + (self.n + 1 + grid_points + 1) * usize_bytes
            + near_entries * (4 + 8)
            + (self.n + 1) * usize_bytes
            + self.n * 3 * 8
            + fft_points * 8;
        PfftStats {
            filaments: self.n,
            grid_spacing,
            grid_dims,
            fft_dims,
            projection_entries,
            near_entries,
            memory_bytes,
        }
    }
}

/// Axis-aligned bounding box of every filament's volume.
fn bounding_box(filaments: &[Filament]) -> (Vector3<f64>, Vector3<f64>) {
    let mut lo = Vector3::repeat(f64::INFINITY);
    let mut hi = Vector3::repeat(f64::NEG_INFINITY);
    for f in filaments {
        let reach = 0.5 * (f.width() * f.width_dir().abs() + f.height() * f.height_dir().abs());
        for end in [f.start(), f.end()] {
            lo = lo.inf(&(end - reach));
            hi = hi.sup(&(end + reach));
        }
    }
    (lo, hi)
}

/// The spacing `h` for which `Π_a max(extent_a / h, 1) = cells`: the box
/// holds about `cells` cells, an axis thinner than `h` counting as one.
fn auto_spacing(extent: Vector3<f64>, cells: f64) -> f64 {
    let largest = extent.max();
    let count = |h: f64| extent.iter().map(|&e| (e / h).max(1.0)).product::<f64>();
    if cells <= 1.0 || largest <= 0.0 {
        return largest.max(f64::MIN_POSITIVE);
    }
    // `count` is continuous and decreasing in h; bisect in log space.
    let (mut small, mut large) = (largest / cells, largest);
    for _ in 0..200 {
        let mid = (small * large).sqrt();
        if count(mid) > cells {
            small = mid;
        } else {
            large = mid;
        }
    }
    large
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_spacing_hits_the_cell_count_for_flat_and_solid_boxes() {
        let cube = auto_spacing(Vector3::new(1.0, 1.0, 1.0), 1000.0);
        assert!((cube - 0.1).abs() < 1e-9);
        // A planar box: the thin axis counts as one cell.
        let plate = auto_spacing(Vector3::new(1.0, 4.0, 1e-6), 400.0);
        assert!((plate - 0.1).abs() < 1e-9);
        // A line: all along the long axis.
        let line = auto_spacing(Vector3::new(2.0, 1e-5, 1e-5), 100.0);
        assert!((line - 0.02).abs() < 1e-9);
        assert_eq!(auto_spacing(Vector3::new(1.0, 0.5, 0.2), 0.5), 1.0);
    }
}
