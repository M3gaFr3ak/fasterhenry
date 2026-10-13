//! fasterhenry — clean-room PEEC (partial-element equivalent circuit)
//! resistance and inductance extraction for 3-D conductor geometries.
//!
//! This crate implements the published method (Ruehli's PEEC; Kamon, Tsuk &
//! White, *FASTHENRY: a multipole-accelerated 3-D inductance extraction
//! program*, IEEE T-MTT 1994; Grover/Rosa closed forms for filament partial
//! inductances) from the literature. It contains no code derived from MIT's
//! FastHenry/FastCap, whose license permits only internal, noncommercial use
//! and forbids redistribution. See CONTRIBUTING.md before adding anything.
//!
//! # Pipeline
//!
//! A [`Geometry`] of nodes and segments is cut into [`Filament`]s
//! ([`discretize`], or [`discretize_graded`] for a skin-depth-aware grid
//! graded toward the conductor surfaces); their resistances and partial
//! inductances
//! ([`mod@inductance`]) form the branch impedance `R + jωL`; a loop basis
//! ([`mesh`]) turns that into the mesh system, which [`solve()`] reduces to
//! the impedance matrix `Z(ω)` seen at user-defined [`Port`]s over a list of
//! frequencies, returned as a JSON-serializable [`SweepResult`].
//!
//! [`coupling`] is the optional sparsity lever on that pipeline: name groups
//! of segments, declare which pairs of them are coupled, and the mutual
//! inductance of the rest is never computed. The default couples everything.
//!
//! [`pfft`] is the scaling lever: a matrix-free [`PfftOperator`] that
//! evaluates `L·x` by the precorrected FFT method, in near-linear time and
//! memory, without assembling `L`. [`IterativeSystem`] builds on it: the
//! same port impedance matrix as [`MeshSystem`], solved by [`mod@gmres`] on
//! that operator instead of by dense LU. The dense [`MeshSystem`] remains
//! the default up to [`DENSE_PATH_MAX_FILAMENTS`] filaments, which is what
//! [`SolverChoice::Auto`] — the front end's default — switches on.
//!
//! # Example
//!
//! A 1 mm copper bar, 100 µm × 20 µm in cross-section, with a port across
//! its ends, swept from DC to 1 GHz. All quantities are SI.
//!
//! ```
//! use fasterhenry::{solve, Discretization, Geometry, Node, Port, SegmentDef, Subdivision};
//!
//! let sigma_cu = 5.8e7; // S/m
//! let mut geometry = Geometry::new();
//! let a = geometry.add_node(Node::new(0.0, 0.0, 0.0))?;
//! let b = geometry.add_node(Node::new(1e-3, 0.0, 0.0))?;
//! geometry.add_segment(SegmentDef::new(a, b, 100e-6, 20e-6, sigma_cu))?;
//!
//! let ports = [Port::new(a, b)];
//! // A 5 × 3 filament grid across the cross-section resolves skin effect.
//! let discretization = Discretization::Uniform(Subdivision::new(5, 3));
//! let result = solve(&geometry, &ports, &discretization, &[0.0, 1e6, 1e9])?;
//!
//! // At DC the resistance is l / (σ·A).
//! let r_dc = result.resistance(0)[(0, 0)];
//! assert!((r_dc - 1e-3 / (sigma_cu * 100e-6 * 20e-6)).abs() < 1e-9);
//! // Skin effect raises R at 1 GHz; the partial self-inductance is ~1 nH.
//! assert!(result.resistance(2)[(0, 0)] > r_dc);
//! let l_1mhz = result.inductance(1).expect("defined at f > 0")[(0, 0)];
//! assert!(l_1mhz > 0.5e-9 && l_1mhz < 1.5e-9);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod coupling;
pub mod dense;
pub mod filament;
pub mod geometry;
pub mod gmres;
pub mod inductance;
pub mod iterative;
pub mod mesh;
pub mod pfft;
pub mod plane;
pub mod result;
pub mod solve;

pub use coupling::{Coupling, CouplingError, TruncationWarning};
pub use filament::{
    discretize, discretize_graded, discretize_graded_per_axis, DiscretizeError, Filament,
};
pub use geometry::{
    Geometry, GeometryError, LocalBasis, Node, NodeId, Segment, SegmentDef, SegmentError,
};
pub use gmres::{GmresOutcome, GmresParams};
pub use inductance::{
    mutual_batch, mutual_batch_detailed, mutual_inductance, partial_inductance_matrix,
    partial_inductance_matrix_detailed, partial_inductance_matrix_masked,
    partial_inductance_matrix_masked_with, partial_inductance_method_counts, self_inductance,
    KernelError, MethodCounts, MutualBatch, PairMask, MU0,
};
pub use iterative::{IterativeParams, IterativeSolution, IterativeSystem};
pub use mesh::{Branch, MeshError, MeshMatrix, Port};
pub use pfft::{GridSpacing, PfftError, PfftOperator, PfftParams, PfftStats};
pub use result::{Counts, Provenance, SweepResult, Timing};
pub use solve::{
    filament_resistance, skin_depth, solve, AxisGrading, Discretization, Grading, MeshSystem,
    SkinDepthGrading, SolveError, Solver, SolverChoice, Subdivision, DENSE_PATH_MAX_FILAMENTS,
};

/// Crate version, for CLI `--version` and JSON output provenance.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
