//! Matrix-free port impedance extraction: GMRES on the precorrected-FFT
//! operator.
//!
//! [`IterativeSystem`] computes the same port impedance matrix as
//! [`MeshSystem`](crate::MeshSystem),
//!
//! ```text
//! Z(ω) = Z_pp − Z_pe · Z_ee⁻¹ · Z_ep ,     Z_xy = M_x (R + jωL) M_yᵀ ,
//! ```
//!
//! (see [`mod@crate::solve`] for the formulation), without ever forming the
//! partial-inductance matrix `L` or the dense internal-loop block `Z_ee`.
//! Instead:
//!
//! * `L·x` is the matrix-free [`PfftOperator`], whose near-field pairs are
//!   exact and whose far field is accurate to about `1e-7` at the default
//!   [`PfftParams`];
//! * the action of `Z_ee` on a vector of internal loop currents `x` is
//!   `M_e · (R·c + jω·L·c)` with `c = M_eᵀ·x` the branch currents: two
//!   sparse products with the loop basis, a diagonal, and two real pFFT
//!   products (one each for the real and imaginary part of `c`, since `L`
//!   is real);
//! * `Z_ee⁻¹ · Z_ep` is solved column by column — one restarted
//!   [GMRES](crate::gmres) solve per port — with a Jacobi (diagonal)
//!   preconditioner; set `FASTERHENRY_PRECOND=none` to run the same
//!   solves unpreconditioned (iteration-floor diagnostics);
//! * the port columns `Z_ep` and the block `Z_pp` need one pFFT product per
//!   port, done once at assembly, since `R` and `L` are frequency
//!   independent.
//!
//! Memory is therefore `O(n)` in the number of filaments — the operator, a
//! few branch vectors and `restart + 1` Krylov vectors of the internal loop
//! count — instead of the dense path's `O(n²)`, and each GMRES iteration
//! costs two pFFT products.
//!
//! # One GMRES solve per port
//!
//! GMRES solves one right-hand side at a time. The ports are solved one
//! after another, each from a zero initial guess; a block variant would
//! share one Krylov space between the ports, but ports are few, the pFFT
//! product is already parallel internally, and independent solves keep every
//! column's convergence test and failure report separate.
//!
//! # Accuracy
//!
//! Two approximations separate the result from the dense path: the pFFT
//! far field (see [`crate::pfft`]) and the GMRES tolerance (relative
//! residual [`GmresParams::tolerance`], `1e-10` by default). At the defaults
//! the result matches [`MeshSystem`](crate::MeshSystem) to far better than
//! `1e-4` relative on the crate's fixtures, which the tests check. The
//! result is symmetrized, `Z ← (Z + Zᵀ)/2`, since reciprocity holds exactly
//! and only the approximations break it.
//!
//! # The DC limit
//!
//! At `f = 0` the internal system is `M_e R M_eᵀ`: sparse, real, symmetric
//! positive definite, and independent of `L`. It is solved by GMRES in real
//! arithmetic without a single pFFT product, and the imaginary part of the
//! result is exactly zero — as for the dense path.
//!
//! # Scope
//!
//! Every pair of filaments is coupled; a [`Coupling`](crate::Coupling)
//! truncation is a dense-path feature. Which path to use for a given size
//! stays the caller's decision, with a default for callers that would rather
//! not choose: [`SolverChoice::Auto`](crate::SolverChoice::Auto) keeps
//! [`MeshSystem`](crate::MeshSystem) up to
//! [`DENSE_PATH_MAX_FILAMENTS`](crate::DENSE_PATH_MAX_FILAMENTS) filaments
//! and selects this path above it (measurements in `docs/benchmarks.md`).

use std::time::Instant;

use faer::traits::ComplexField as FaerComplexField;
use nalgebra::{ComplexField, DMatrix, Dyn, LU};
use num_complex::Complex;
use rayon::prelude::*;

use crate::conv::{mesh_gather, mesh_scatter, ConvPrecond, MatvecCore, TabComposition};
use crate::cubel::{CubelGlobal, CubelParams, CubelPattern, AVER_ROWS};
use crate::filament::Filament;
use crate::geometry::Geometry;
use crate::gmres::{gmres, GmresOutcome, GmresParams};
use crate::inductance::{mutual_inductance, self_inductance, KernelError};
use crate::mesh::{MeshError, MeshMatrix, Port};
use crate::pfft::{PfftOperator, PfftParams, PfftStats};
use crate::result::{Counts, Provenance, SweepResult, Timing};
use crate::solve::{
    check_frequency, discretize_geometry, filament_resistance, Discretization, SolveError,
    Subdivision, SWEEP_MEMORY_BUDGET_BYTES,
};
use crate::twolevel::{TwoLevel, TwoLevelFactors};

type C64 = Complex<f64>;

/// Loops of at most this many branches get the exact diagonal
/// `Σₐ Σ_b sₐ s_b L_ab` in the Jacobi preconditioner; longer loops (long
/// conductor loops through the spanning forest) use the sum of their
/// branches' self inductances, which bounds the cost of the preconditioner
/// at `O(n)` kernel evaluations. Either way the preconditioner only affects
/// the iteration count, never the answer.
pub const EXACT_DIAGONAL_MAX_LOOP_LENGTH: usize = 32;

/// Parameters of an [`IterativeSystem`]: the pFFT operator's accuracy and
/// GMRES's stopping rule.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IterativeParams {
    /// The precorrected-FFT operator's parameters.
    pub pfft: PfftParams,
    /// GMRES's tolerance, restart length and iteration cap.
    pub gmres: GmresParams,
}

/// `Z(ω)` at one frequency with the GMRES outcome of every port's solve.
#[derive(Clone, Debug, PartialEq)]
pub struct IterativeSolution {
    /// The port impedance matrix, in ohms.
    pub impedance: DMatrix<C64>,
    /// One entry per port, in port order; empty when there are no internal
    /// loops (nothing is solved).
    pub gmres: Vec<GmresOutcome>,
}

/// The assembled, frequency-independent, matrix-free mesh system of a
/// The coarse (un-subdivided) twin of a subdivided system: one filament per
/// segment. Built at `assemble` when the fine mesh is subdivided; the
/// coarse/fine precond (`cubel-global-sub`) factors this instead of the fine
/// system, so its footprint is the un-subdivided size.
#[derive(Debug)]
pub(crate) struct CoarseSystem {
    pub(crate) filaments: Vec<Filament>,
    pub(crate) resistances: Vec<f64>,
    pub(crate) mesh: MeshMatrix,
}

/// geometry and its ports. See the [module documentation](self).
#[derive(Debug)]
pub struct IterativeSystem {
    ports: Vec<Port>,
    filaments: Vec<Filament>,
    resistances: Vec<f64>,
    mesh: MeshMatrix,
    /// Filaments per segment (the discretization's per-segment counts). The
    /// coarse/fine precond aggregates fine branches into segments with it.
    per_segment: Vec<usize>,
    /// The coarse (one-filament-per-segment) twin, when the fine mesh is
    /// subdivided. `None` when the fine mesh is already un-subdivided.
    coarse: Option<CoarseSystem>,
    operator: PfftOperator,
    /// The conv-matvec near half ([`MatvecCore`], increment 5): built
    /// when `FASTERHENRY_CONV_MATVEC=1`, alongside a far-only pFFT
    /// operator. `None` = the historical precorrect operator.
    matvec: Option<MatvecCore>,
    gmres: GmresParams,
    /// `M R Mᵀ` and `M L Mᵀ` restricted to (internal × port) and
    /// (port × port) loops.
    r_ep: DMatrix<f64>,
    l_ep: DMatrix<f64>,
    r_pp: DMatrix<f64>,
    l_pp: DMatrix<f64>,
    /// Diagonals of `M_e R M_eᵀ` and (an approximation of) `M_e L M_eᵀ`,
    /// the Jacobi preconditioner.
    r_diag: Vec<f64>,
    l_diag: Vec<f64>,
    counts: Counts,
    operator_seconds: f64,
    assembly_seconds: f64,
}

impl IterativeSystem {
    /// Discretizes `geometry`, builds the loop basis for `ports` and the
    /// pFFT operator of the filaments, and precomputes the port columns and
    /// the preconditioner.
    ///
    /// # Errors
    ///
    /// As [`MeshSystem::assemble`](crate::MeshSystem::assemble), plus
    /// [`SolveError::Pfft`] if the operator cannot be built from
    /// `params.pfft`.
    pub fn assemble(
        geometry: &Geometry,
        ports: &[Port],
        discretization: &Discretization,
        params: &IterativeParams,
    ) -> Result<Self, SolveError> {
        let start = Instant::now();
        let (filaments, per_segment) = discretize_geometry(geometry, discretization)?;
        let mesh = MeshMatrix::build(geometry, &per_segment, ports)?;
        let resistances: Vec<f64> = filaments.iter().map(filament_resistance).collect();

        // The coarse (one-filament-per-segment) twin, for the coarse/fine
        // precond. Only built when the fine mesh is actually subdivided; else
        // the coarse == the fine and today's cubel-global already is it.
        let coarse = if per_segment.iter().all(|&n| n == 1) {
            None
        } else {
            let unit = Discretization::Uniform(Subdivision::new(1, 1));
            let (cf, cps) = discretize_geometry(geometry, &unit)?;
            let cmesh = MeshMatrix::build(geometry, &cps, ports)?;
            let cres: Vec<f64> = cf.iter().map(filament_resistance).collect();
            eprintln!(
                "cubel-global-sub: coarse twin: {} segments, {} loops (fine: {} branches, {} loops)",
                cmesh.branch_count(),
                cmesh.internal_count(),
                mesh.branch_count(),
                mesh.internal_count(),
            );
            Some(CoarseSystem {
                filaments: cf,
                resistances: cres,
                mesh: cmesh,
            })
        };

        let operator_start = Instant::now();
        // The conv-matvec lane (increment 5): `FASTERHENRY_CONV_MATVEC=1`
        // swaps the equation to `grid far + conv near model` - the
        // answer-exactness contract amendment (user-authorized; see the
        // increment-5 decision in the peec books). Default OFF keeps the
        // historical precorrect operator byte-for-byte.
        let conv_matvec = std::env::var("FASTERHENRY_CONV_MATVEC")
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        let mut pfft_params = params.pfft.clone();
        if conv_matvec {
            pfft_params.precorrect_near = false;
        }
        let operator = PfftOperator::new(&filaments, &pfft_params)?;
        let matvec = if conv_matvec {
            // The n matrix of the near model's stages is omega-free; the
            // tab/spine compositions reuse the engine's env knobs, so
            // the A/B lane can price them independently.
            let tab_composition = match std::env::var("FASTERHENRY_CONV_TABS")
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
                .as_deref()
            {
                None | Some("") | Some("both") => TabComposition::Both,
                Some("matvec") => TabComposition::Matvec,
                Some("off") => TabComposition::Off,
                Some(other) => {
                    eprintln!(
                        "conv: warning: FASTERHENRY_CONV_TABS={other:?} not |both|matvec|off; using both"
                    );
                    TabComposition::Both
                }
            };
            let tab_pour = match std::env::var("FASTERHENRY_CONV_TAB_POUR")
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
                .as_deref()
            {
                None | Some("") | Some("1") | Some("on") | Some("true") => true,
                Some("0") | Some("off") | Some("false") => false,
                Some(other) => {
                    eprintln!(
                        "conv: warning: FASTERHENRY_CONV_TAB_POUR={other:?} not 0/1; using 1"
                    );
                    true
                }
            };
            let spine_composition = match std::env::var("FASTERHENRY_CONV_SPINE")
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
                .as_deref()
            {
                None | Some("") | Some("both") => TabComposition::Both,
                Some("matvec") => TabComposition::Matvec,
                Some("off") => TabComposition::Off,
                Some(other) => {
                    eprintln!(
                        "conv: warning: FASTERHENRY_CONV_SPINE={other:?} not |both|matvec|off; using both"
                    );
                    TabComposition::Both
                }
            };
            let parse_env_m = |name: &str| -> Option<f64> {
                let text = std::env::var(name).ok()?;
                match text.trim().parse::<f64>() {
                    Ok(value) => Some(value),
                    Err(_) => {
                        eprintln!(
                            "conv: warning: {name}={text:?} not a number; falling through the bin chain"
                        );
                        None
                    }
                }
            };
            let tab_bin_m = parse_env_m("FASTERHENRY_CENSUS_VIA_BIN_M")
                .or_else(|| parse_env_m("FASTERHENRY_CENSUS_RADIUS_M"))
                .unwrap_or_else(|| crate::census::derived_radius(&filaments));
            Some(
                MatvecCore::build(
                    &operator,
                    &filaments,
                    tab_composition,
                    tab_bin_m,
                    tab_pour,
                    spine_composition,
                )
                .map_err(SolveError::from)?,
            )
        } else {
            None
        };
        let operator_seconds = operator_start.elapsed().as_secs_f64();

        let (internal, port_count) = (mesh.internal_count(), mesh.port_count());
        let branches = filaments.len();
        let mut r_ep = DMatrix::zeros(internal, port_count);
        let mut l_ep = DMatrix::zeros(internal, port_count);
        let mut r_pp = DMatrix::zeros(port_count, port_count);
        let mut l_pp = DMatrix::zeros(port_count, port_count);
        for p in 0..port_count {
            let mut current = vec![0.0; branches];
            for &(branch, sign) in mesh.row(internal + p) {
                current[branch] += sign;
            }
            let flux = operator.apply(&current);
            let drop: Vec<f64> = current
                .iter()
                .zip(&resistances)
                .map(|(c, r)| c * r)
                .collect();
            let r_column = gather_all(&mesh, &drop);
            let l_column = gather_all(&mesh, &flux);
            for i in 0..internal {
                r_ep[(i, p)] = r_column[i];
                l_ep[(i, p)] = l_column[i];
            }
            for q in 0..port_count {
                r_pp[(q, p)] = r_column[internal + q];
                l_pp[(q, p)] = l_column[internal + q];
            }
        }
        // Symmetric in exact arithmetic; make it so to the last bit.
        let r_pp = (&r_pp + r_pp.transpose()) * 0.5;
        let l_pp = (&l_pp + l_pp.transpose()) * 0.5;

        let diagonals: Vec<(f64, f64)> = (0..internal)
            .into_par_iter()
            .map(|i| {
                let row = mesh.row(i);
                let r: f64 = row.iter().map(|&(branch, _)| resistances[branch]).sum();
                loop_inductance(&filaments, row).map(|l| (r, l))
            })
            .collect::<Result<_, KernelError>>()?;
        let (r_diag, l_diag) = diagonals.into_iter().unzip();

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
            mesh,
            per_segment,
            coarse,
            operator,
            matvec,
            gmres: params.gmres,
            r_ep,
            l_ep,
            r_pp,
            l_pp,
            r_diag,
            l_diag,
            counts,
            operator_seconds,
            assembly_seconds: start.elapsed().as_secs_f64(),
        })
    }

    /// The number of intra-segment (redistribution) loops, `sum(n_i - 1)`:
    /// rows `0..intra_loop_count()` of the loop basis, with the trailing rows
    /// the coarse/forest loops. The coarse/fine precond splits on this.
    pub(crate) fn intra_loop_count(&self) -> usize {
        self.per_segment.iter().map(|&n| n - 1).sum()
    }

    /// The first segment whose filament count differs from segment 0's, or
    /// `None` if all agree. The coarse/fine precond's aggregation assumes
    /// uniformity; a non-uniform deck is a user error there.
    pub(crate) fn non_uniform_segment(&self) -> Option<usize> {
        let first = *self.per_segment.first()?;
        self.per_segment.iter().position(|&n| n != first)
    }

    /// Filaments per segment (the discretization's per-segment counts).
    pub(crate) fn per_segment(&self) -> &[usize] {
        &self.per_segment
    }

    /// The ports, in the order of the rows and columns of `Z`.
    pub fn ports(&self) -> &[Port] {
        &self.ports
    }

    /// Every filament (branch), segment by segment.
    pub fn filaments(&self) -> &[Filament] {
        &self.filaments
    }

    /// The loop basis.
    pub fn mesh(&self) -> &MeshMatrix {
        &self.mesh
    }

    /// Problem size: nodes, segments, filaments, meshes, ports.
    pub fn counts(&self) -> Counts {
        self.counts
    }

    /// Sizes of the pFFT operator.
    pub fn operator_stats(&self) -> PfftStats {
        self.operator.stats()
    }

    /// The GMRES parameters every solve uses.
    pub fn gmres_params(&self) -> GmresParams {
        self.gmres
    }

    /// The port impedance matrix `Z(ω)`, in ohms, at `frequency` hertz; the
    /// iterative counterpart of
    /// [`MeshSystem::impedance`](crate::MeshSystem::impedance).
    ///
    /// # Errors
    ///
    /// [`SolveError::InvalidFrequency`] for a negative or non-finite
    /// frequency (reported with index 0), [`SolveError::NotConverged`] if a
    /// GMRES solve misses its tolerance, [`SolveError::Singular`] if the
    /// result is not finite.
    pub fn impedance(&self, frequency: f64) -> Result<DMatrix<C64>, SolveError> {
        self.solve(frequency).map(|solution| solution.impedance)
    }

    /// As [`impedance`](Self::impedance), also returning how every port's
    /// GMRES solve went.
    ///
    /// # Errors
    ///
    /// As [`impedance`](Self::impedance).
    pub fn solve(&self, frequency: f64) -> Result<IterativeSolution, SolveError> {
        check_frequency(0, frequency)?;
        let kind = PrecondKind::from_env();
        let (z, outcomes) = if frequency == 0.0 {
            let precond = match kind {
                PrecondKind::Cubel => Precond::cubel(self, 0.0)?,
                PrecondKind::CubelGlobal => Precond::cubel_global(self, 0.0)?,
                PrecondKind::CubelGlobalSub => Precond::cubel_global_sub(self, 0.0)?,
                PrecondKind::TwoLevel => Precond::two_level(self, 0.0)?,
                PrecondKind::Conv => Precond::conv_kind(self, 0.0)?,
                PrecondKind::None => Precond::Identity,
                PrecondKind::Jacobi => Precond::Jacobi(self.r_diag.clone()),
            };
            crate::memgate::set_stage(crate::memgate::STAGE_GMRES);
            let (z, outcomes) = schur_complement_gmres(
                &self.r_ep,
                &self.r_pp,
                |x| self.apply_dc(x),
                &precond,
                &self.gmres,
                frequency,
            )?;
            (z.map(|re| C64::new(re, 0.0)), outcomes)
        } else {
            let omega = std::f64::consts::TAU * frequency;
            let complex =
                |r: &DMatrix<f64>, l: &DMatrix<f64>| r.zip_map(l, |r, l| C64::new(r, omega * l));
            let diagonal: Vec<C64> = self
                .r_diag
                .iter()
                .zip(&self.l_diag)
                .map(|(&r, &l)| C64::new(r, omega * l))
                .collect();
            let precond = match kind {
                PrecondKind::Cubel => Precond::cubel(self, omega)?,
                PrecondKind::CubelGlobal => Precond::cubel_global(self, omega)?,
                PrecondKind::CubelGlobalSub => Precond::cubel_global_sub(self, omega)?,
                PrecondKind::TwoLevel => Precond::two_level(self, omega)?,
                PrecondKind::Conv => Precond::conv_kind(self, omega)?,
                PrecondKind::None => Precond::Identity,
                PrecondKind::Jacobi => Precond::Jacobi(diagonal),
            };
            crate::memgate::set_stage(crate::memgate::STAGE_GMRES);
            schur_complement_gmres(
                &complex(&self.r_ep, &self.l_ep),
                &complex(&self.r_pp, &self.l_pp),
                |x| self.apply_ac(omega, x),
                &precond,
                &self.gmres,
                frequency,
            )?
        };
        let z = (&z + z.transpose()).map(|e| e * 0.5);
        if z.iter().all(|e| e.re.is_finite() && e.im.is_finite()) {
            Ok(IterativeSolution {
                impedance: z,
                gmres: outcomes,
            })
        } else {
            Err(SolveError::Singular { frequency })
        }
    }

    /// `Z(ω)` at every one of `frequencies` (hertz), with provenance; the
    /// iterative counterpart of [`MeshSystem::sweep`](crate::MeshSystem::sweep).
    ///
    /// Frequencies are solved concurrently, as many at a time as fit a
    /// fixed memory budget; each is an independent deterministic
    /// computation, so the result does not depend on the number of threads.
    /// `Provenance::timing.inductance_s` reports the time spent building the
    /// pFFT operator.
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
        let stats = self.operator.stats();
        let fft_points: usize = stats.fft_dims.iter().product();
        // Krylov basis and work vectors, branch vectors, FFT buffers.
        let bytes_per_solve = (self.gmres.restart + 4) * self.counts.internal_meshes * 16
            + 8 * self.counts.filaments * 8
            + 4 * fft_points * 16;
        let concurrent = (SWEEP_MEMORY_BUDGET_BYTES / bytes_per_solve.max(1)).max(1);

        // Progress: one marker per frequency. The sweep is concurrent within a
        // memory-budget chunk, so the "done k/N" counter is the reliable signal.
        let done = std::sync::atomic::AtomicUsize::new(0);
        let total = frequencies.len();
        let mut impedance = Vec::with_capacity(frequencies.len());
        for chunk in frequencies.chunks(concurrent) {
            let solved: Vec<_> = chunk
                .par_iter()
                .map(|&f| {
                    eprintln!("sweep: solving f = {f:e} Hz ...");
                    let z = self.impedance(f);
                    let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    eprintln!("sweep: freq {d}/{total} done (f = {f:e} Hz)");
                    z
                })
                .collect();
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
                    inductance_s: self.operator_seconds,
                    assembly_s: self.assembly_seconds,
                    solve_s: solve_seconds,
                    total_s: self.assembly_seconds + solve_seconds,
                }),
            },
        })
    }

    /// Branch currents `M_eᵀ·x` of internal loop currents `x`.
    fn scatter<T: ComplexField<RealField = f64> + Copy>(&self, x: &[T]) -> Vec<T> {
        let mut branch = vec![T::zero(); self.filaments.len()];
        for (i, &xi) in x.iter().enumerate() {
            for &(b, sign) in self.mesh.row(i) {
                branch[b] += xi.scale(sign);
            }
        }
        branch
    }

    /// `M_e R M_eᵀ · x`.
    fn apply_dc(&self, x: &[f64]) -> Vec<f64> {
        let mut drop = self.scatter(x);
        for (d, r) in drop.iter_mut().zip(&self.resistances) {
            *d *= r;
        }
        gather(&self.mesh, x.len(), &drop)
    }

    /// `M_e (R + j omega L) M_e^T x`, with `L` the pFFT operator, plus
    /// the conv-matvec near half when present (the two flux halves are
    /// additive by the substitution construction; the core's corr
    /// values cancel the stages' overlap on member pairs).
    fn apply_ac(&self, omega: f64, x: &[C64]) -> Vec<C64> {
        let current = self.scatter(x);
        let re: Vec<f64> = current.iter().map(|c| c.re).collect();
        let im: Vec<f64> = current.iter().map(|c| c.im).collect();
        let near_re = if let Some(matvec) = &self.matvec {
            Some((
                {
                    let mut flux = vec![0.0f64; re.len()];
                    matvec.apply(&re, &mut flux);
                    flux
                },
                {
                    let mut flux = vec![0.0f64; im.len()];
                    matvec.apply(&im, &mut flux);
                    flux
                },
            ))
        } else {
            None
        };
        let (flux_re, flux_im) =
            rayon::join(|| self.operator.apply(&re), || self.operator.apply(&im));
        let (flux_re, flux_im) = match &near_re {
            Some((near_re, near_im)) => {
                let mut re = flux_re;
                let mut im = flux_im;
                for (dst, src) in re.iter_mut().zip(near_re) {
                    *dst += *src;
                }
                for (dst, src) in im.iter_mut().zip(near_im) {
                    *dst += *src;
                }
                (re, im)
            }
            None => (flux_re, flux_im),
        };
        let drop: Vec<C64> = (0..current.len())
            .map(|b| {
                let r = self.resistances[b];
                C64::new(
                    r * re[b] - omega * flux_im[b],
                    r * im[b] + omega * flux_re[b],
                )
            })
            .collect();
        gather(&self.mesh, x.len(), &drop)
    }
}

/// `pp − epᵀ · ee⁻¹ · ep` with `ee⁻¹` applied by one Jacobi-preconditioned
/// GMRES solve per column of `ep`. The transpose is plain, not conjugated:
/// the mesh impedance matrix is complex symmetric.
/// Which preconditioner the knob selects. Read once per solve, never
/// per-iteration, and treated as transient runtime state (not recorded
/// anywhere in results beyond the solve's own logs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrecondKind {
    /// Default: the Jacobi diagonal.
    Jacobi,
    /// M = I: plain GMRES on the same operator, the unpreconditioned
    /// iteration floor (`FASTERHENRY_PRECOND=none`).
    None,
    /// The CUBEL mesh-block prototype (`FASTERHENRY_PRECOND=cubel`).
    Cubel,
    /// The D-4 letter: one global near-pattern matrix factored by faer's
    /// sparse LU (`FASTERHENRY_PRECOND=cubel-global`).
    CubelGlobal,
    /// The coarse/fine subdivision precond: the same global factorization,
    /// but on the one-filament-per-segment twin, so its footprint is the
    /// un-subdivided size (`FASTERHENRY_PRECOND=cubel-global-sub`).
    CubelGlobalSub,
    /// The D-17 first rung: one global near-pattern matrix split by
    /// copper layer into per-layer blocks plus a cross-layer CSR
    /// applied in a two-sweep block Gauss-Seidel
    /// (`FASTERHENRY_PRECOND=two-level`).
    /// LEGACY / DEPRECATED (2026-10-19): kept selectable but frozen-
    /// prone on cross-heavy patterns (see `crate::twolevel` module
    /// docs). New engines must land as NEW kinds, never reuse this.
    TwoLevel,
    /// The v2 conv engine (2026-10-20): fully matrix-free near model -
    /// per-layer-pair FFT kernel applies plus the branch-R diagonal -
    /// solved inside the apply by a fixed-count Richardson sweep
    /// whose smoother is the per-family Tikhonov inverse
    /// (`FASTERHENRY_PRECOND=conv`). Via-tab blocks and the branch-CSR
    /// spine are reserved slots (increments 2c/2d); until they land
    /// the model simply omits those couplings, which dims precond
    /// quality but never the operator-exact answer.
    Conv,
}

impl PrecondKind {
    fn from_env() -> Self {
        match std::env::var("FASTERHENRY_PRECOND").as_deref() {
            Ok("none") => PrecondKind::None,
            Ok("cubel") => PrecondKind::Cubel,
            Ok("cubel-global") => PrecondKind::CubelGlobal,
            Ok("cubel-global-sub") => PrecondKind::CubelGlobalSub,
            Ok("two-level") => PrecondKind::TwoLevel,
            Ok("conv") => PrecondKind::Conv,
            _ => PrecondKind::Jacobi,
        }
    }
}

/// The preconditioner of the internal (loop-space) block system, built once
/// per frequency.
enum Precond<T: ComplexField<RealField = f64> + Copy> {
    Jacobi(Vec<T>),
    /// Right-preconditioning with M = I: plain GMRES on the same operator,
    /// i.e. the unpreconditioned iteration floor.
    Identity,
    Cubel(Box<CubelState<T>>),
    CubelGlobal(Box<CubelGlobalState>),
    /// The coarse/fine (subdivision) precond: factored on the coarse twin.
    CubelGlobalSub(Box<CubelSubState<T>>),
    TwoLevel(Box<TwoLevelState<T>>),
    Conv(Box<ConvCore>),
}

/// The built cube preconditioner: the omega-independent pattern plus this
/// frequency's per-cube factorisations.
struct CubelState<T: ComplexField<RealField = f64> + Copy> {
    pattern: CubelPattern,
    factors: Vec<LU<T, Dyn, Dyn>>,
}

/// The built global preconditioner: the omega-independent CSC pattern with
/// its COLAMD symbolic factorization, plus this frequency's numeric LU
/// (`None` after a failed factorization: the apply falls back to the
/// identity, with a one-shot warning from the builder).
struct CubelGlobalState {
    global: CubelGlobal,
    factors: Option<faer::sparse::linalg::solvers::Lu<u32, crate::cubel::PrecC>>,
}

/// The coarse/fine ("subdivision") state: a **multiplicative two-level**
/// preconditioner, `z = B_b^-1 v + P A_c^-1 R (v - Z B_b^-1 v)`. `B_b` is a
/// full-rank local (block-local) smoother over the fine mesh that covers every
/// DOF - including the intra-segment (bundle) rows a coarse operator cannot
/// represent - while the coarse factor `A_c` (built on the one-filament-per-
/// segment twin) corrects the conductor part of the residual the local
/// smoother leaves behind. `R`/`P` select the fine conductor block
/// `r_off..r_off + c_len` (the coarse's loops, checked at build).
struct CubelSubState<T: ComplexField<RealField = f64> + Copy> {
    /// The local (`B_b^-1`) smoother: block-local over the fine mesh.
    local: CubelState<T>,
    factors: Option<faer::sparse::linalg::solvers::Lu<u32, crate::cubel::PrecC>>,
    /// First fine internal row that is a coarse (conductor) loop.
    r_off: usize,
    /// Number of coarse loops == the fine's conductor loop count.
    c_len: usize,
}

/// The built two-level state: the omega-independent layer split plus
/// this frequency's per-layer factorizations and cross values.
struct TwoLevelState<T: ComplexField<RealField = f64> + Copy> {
    pattern: TwoLevel,
    factors: TwoLevelFactors<T>,
}

/// The built v2 conv state. T-agnostic on purpose: the conv core runs
/// in C64 and `run` converts at the boundary through `FieldParts`, so
/// the enum variant itself carries no T.
struct ConvCore {
    core: ConvPrecond,
    /// The branch resistances (owned copy: the near-model diagonal).
    resistances: Vec<f64>,
    /// Clone of the incidence the sweeps scatter/gather through.
    mesh: MeshMatrix,
    omega: f64,
    /// Fixed Richardson count inside one precond apply - a fixed
    /// linear map, as the right-preconditioned GMRES contract needs.
    sweeps: usize,
    internal: usize,
}

impl ConvCore {
    /// One precond apply: `sweeps` Richardson passes of
    /// `y <- y + M P M^T (v - M (R + j omega L_conv) M^T y)`.
    fn run_c64(&self, v: &[C64]) -> Vec<C64> {
        let mut y = vec![C64::new(0.0, 0.0); self.internal];
        for _ in 0..self.sweeps {
            // Residual against the near model: matvec + lift.
            let b = mesh_scatter(&self.mesh, self.internal, &y);
            let drops = self.core.near_drops(&b, &self.resistances, self.omega);
            let ay = mesh_gather(&self.mesh, self.internal, &drops);
            let residual: Vec<C64> = v.iter().zip(&ay).map(|(&v, &a)| v - a).collect();
            // Smoother: branch-space Tikhonov inverse + lift.
            let b_residual = mesh_scatter(&self.mesh, self.internal, &residual);
            let corr_branch = self.core.smooth(&b_residual);
            let corr = mesh_gather(&self.mesh, self.internal, &corr_branch);
            for (y, c) in y.iter_mut().zip(&corr) {
                *y += *c;
            }
        }
        y
    }
}

/// The conv apply's T boundary: the core runs C64, `run` hands back T.
fn conv_apply<T: ComplexField<RealField = f64> + Copy + crate::cubel::FieldParts>(
    core: &ConvCore,
    v: &[T],
) -> Vec<T> {
    let input: Vec<C64> = v
        .iter()
        .map(|&t| {
            let (re, im) = t.into_parts();
            C64::new(re, im)
        })
        .collect();
    core.run_c64(&input)
        .into_iter()
        .map(|c| T::of(c.re, c.im))
        .collect()
}

impl ConvCore {
    /// Builds the v2 conv state at `omega`.
    ///
    /// Knobs: `FASTERHENRY_CONV_DAMP` (Tikhonov floor of the smoother
    /// kernels, default 0.01, valid range (0, 1)) and
    /// `FASTERHENRY_CONV_SWEEPS` (fixed Richardson count per apply,
    /// default 2). Kernels are resampled per frequency (`omega` rides
    /// the smoother's complex kernel, not the forward ones); kernel
    /// caching across frequencies is a booked cost lever.
    fn build_state(system: &IterativeSystem, omega: f64) -> Result<Self, SolveError> {
        crate::memgate::set_stage(crate::memgate::STAGE_PRECOND);
        let mut damp: f64 = std::env::var("FASTERHENRY_CONV_DAMP")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0.01);
        if !(0.0..1.0).contains(&damp) {
            eprintln!("conv: warning: FASTERHENRY_CONV_DAMP={damp} outside (0, 1); using 0.01");
            damp = 0.01;
        }
        if damp <= 0.0 {
            eprintln!("conv: warning: FASTERHENRY_CONV_DAMP=0 not permitted; using 0.01");
            damp = 0.01;
        }
        let sweeps: usize = std::env::var("FASTERHENRY_CONV_SWEEPS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(2)
            .max(1);
        // Tab chassis (2c): FASTERHENRY_CONV_TABS=both|matvec|off
        // (default both), FASTERHENRY_CONV_TAB_POUR=0|1 (default 1),
        // via-census bin chain mirrors census_for's fallback exactly
        // (F3 fold: VIA_BIN_M -> CENSUS_RADIUS_M -> derived_radius).
        // Unparseable bin values warn (fail-loud parity with the
        // census CLI's bail) and fall to the next hop.
        let composition = match std::env::var("FASTERHENRY_CONV_TABS")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            None | Some("") | Some("both") => TabComposition::Both,
            Some("matvec") => TabComposition::Matvec,
            Some("off") => TabComposition::Off,
            Some(other) => {
                eprintln!(
                    "conv: warning: FASTERHENRY_CONV_TABS={other:?} not |both|matvec|off; using both"
                );
                TabComposition::Both
            }
        };
        let tab_pour = match std::env::var("FASTERHENRY_CONV_TAB_POUR")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            None | Some("") | Some("1") | Some("on") | Some("true") => true,
            Some("0") | Some("off") | Some("false") => false,
            Some(other) => {
                eprintln!("conv: warning: FASTERHENRY_CONV_TAB_POUR={other:?} not 0/1; using 1");
                true
            }
        };
        // The bin chain mirrors the census CLI's `census_for`
        // fallback exactly (audit fold F3, 2026-10-20): histogram
        // pitch and tab-membership pitch must not skew apart on
        // one deck. Unparseable values warn and fall to the next
        // hop (the census CLI bails; deliberate divergence: a bin
        // pitch is precond cost, never the answer, so the engine
        // keeps the solve alive).
        let parse_env_m = |name: &str| -> Option<f64> {
            let text = std::env::var(name).ok()?;
            match text.trim().parse::<f64>() {
                Ok(value) => Some(value),
                Err(_) => {
                    eprintln!(
                        "conv: warning: {name}={text:?} not a number; falling through the bin chain"
                    );
                    None
                }
            }
        };
        let tab_bin_m = parse_env_m("FASTERHENRY_CENSUS_VIA_BIN_M")
            .or_else(|| parse_env_m("FASTERHENRY_CENSUS_RADIUS_M"))
            .unwrap_or_else(|| crate::census::derived_radius(&system.filaments));
        // Spine chassis (2d): FASTERHENRY_CONV_SPINE=both|matvec|off
        // (default both), damped Jacobi damping
        // FASTERHENRY_CONV_SPINE_DAMP in (0, 1], default 0.5.
        let spine_composition = match std::env::var("FASTERHENRY_CONV_SPINE")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            None | Some("") | Some("both") => TabComposition::Both,
            Some("matvec") => TabComposition::Matvec,
            Some("off") => TabComposition::Off,
            Some(other) => {
                eprintln!(
                    "conv: warning: FASTERHENRY_CONV_SPINE={other:?} not |both|matvec|off; using both"
                );
                TabComposition::Both
            }
        };
        let spine_damp: f64 = std::env::var("FASTERHENRY_CONV_SPINE_DAMP")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0.5);
        let spine_damp = if spine_damp.is_finite() && spine_damp > 0.0 && spine_damp <= 1.0 {
            spine_damp
        } else {
            eprintln!(
                "conv: warning: FASTERHENRY_CONV_SPINE_DAMP={spine_damp} outside (0, 1]; using 0.5"
            );
            0.5
        };
        let core = ConvPrecond::build(
            &system.filaments,
            &system.resistances,
            omega,
            damp,
            composition,
            tab_bin_m,
            tab_pour,
            spine_composition,
            spine_damp,
        )
        .map_err(SolveError::from)?;
        Ok(Self {
            core,
            resistances: system.resistances.clone(),
            mesh: system.mesh.clone(),
            omega,
            sweeps,
            internal: system.mesh.internal_count(),
        })
    }
}

impl<T: ComplexField<RealField = f64> + Copy + Send + Sync + FaerComplexField> Precond<T>
where
    T: crate::cubel::FieldParts,
{
    /// Builds the CUBEL state at `omega` from the system's own pieces.
    fn cubel(system: &IterativeSystem, omega: f64) -> Result<Self, SolveError> {
        let pattern = CubelPattern::build(
            &system.mesh,
            &system.filaments,
            CubelParams {
                max_block_rows: AVER_ROWS,
                near_radius: system.operator.near_radius(),
            },
        )?;
        let factors = pattern.factor_blocks::<T>(&system.mesh, &system.resistances, omega);
        Ok(Self::Cubel(Box::new(CubelState { pattern, factors })))
    }

    /// Builds the global (D-4 letter) state at `omega`.
    ///
    /// The pattern radius is independently tunable via
    /// `FASTERHENRY_CUBEL_RADIUS_MULT` (default 1.0): it scales only the
    /// preconditioner's near-pattern, never the operator's precorrect
    /// radius or the pFFT grid, so `A` is unchanged and the knob trades
    /// memory (pattern CSC + LU fill, ~r^2 scaling) for iterations.
    fn cubel_global(system: &IterativeSystem, omega: f64) -> Result<Self, SolveError> {
        crate::memgate::set_stage(crate::memgate::STAGE_PRECOND);
        let radius_mult: f64 = std::env::var("FASTERHENRY_CUBEL_RADIUS_MULT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1.0);
        if !(0.0..=f64::INFINITY).contains(&radius_mult) || radius_mult <= 0.0 {
            eprintln!(
                "cubel-global: warning: FASTERHENRY_CUBEL_RADIUS_MULT={radius_mult} invalid; using 1.0"
            );
        }
        let radius_mult = if radius_mult > 0.0 { radius_mult } else { 1.0 };
        let global = CubelGlobal::build(
            &system.mesh,
            &system.filaments,
            &system.resistances,
            system.operator.near_radius() * radius_mult,
        )
        .map_err(SolveError::from)?;
        let factors = global.factor(omega);
        Ok(Self::CubelGlobal(Box::new(CubelGlobalState {
            global,
            factors,
        })))
    }

    /// Builds the coarse/fine ("subdivision") state at `omega`.
    ///
    /// A **multiplicative two-level** precond. The local block-local smoother
    /// `B_b` covers every fine DOF (so the map is full rank); the coarse factor
    /// `A_c` - built on the **coarse** (one-filament-per-segment) twin
    /// (`system.coarse`), so its footprint is the un-subdivided size - corrects
    /// the conductor part of the residual `B_b` leaves. `R`/`P` select the
    /// fine's conductor rows (they ARE the coarse's loops, checked at build).
    /// With `nwinc=nhinc=1` there is no split, so this falls through to
    /// `cubel-global`.
    fn cubel_global_sub(system: &IterativeSystem, omega: f64) -> Result<Self, SolveError> {
        crate::memgate::set_stage(crate::memgate::STAGE_PRECOND);
        if let Some(seg) = system.non_uniform_segment() {
            return Err(SolveError::Mesh(MeshError::NonUniformSubdivision {
                segment: seg,
                got: system.per_segment()[seg],
                expected: system.per_segment()[0],
            }));
        }
        let Some(coarse) = &system.coarse else {
            return Self::cubel_global(system, omega);
        };

        // The local (full-rank) smoother B_b: the same block-local build as
        // `cubel`, over the fine mesh. Its job is the DOFs no coarse operator
        // can see (the intra-segment bundle loops).
        let local_pattern = CubelPattern::build(
            &system.mesh,
            &system.filaments,
            CubelParams {
                max_block_rows: AVER_ROWS,
                near_radius: system.operator.near_radius(),
            },
        )?;
        let local_factors =
            local_pattern.factor_blocks::<T>(&system.mesh, &system.resistances, omega);
        let local = CubelState {
            pattern: local_pattern,
            factors: local_factors,
        };

        let radius_mult: f64 = std::env::var("FASTERHENRY_CUBEL_RADIUS_MULT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1.0);
        let radius_mult = if radius_mult > 0.0 { radius_mult } else { 1.0 };

        let cglobal = CubelGlobal::build(
            &coarse.mesh,
            &coarse.filaments,
            &coarse.resistances,
            system.operator.near_radius() * radius_mult,
        )
        .map_err(SolveError::from)?;
        let factors = cglobal.factor(omega);

        // R/P select the fine's conductor block (the trailing rows) for the
        // coarse correction; the bundle rows are intra-segment, invisible to
        // any coarse operator, and are handled by B_b alone.
        let r_off = system.intra_loop_count();
        let c_len = coarse.mesh.internal_count();
        assert_eq!(
            system.mesh.internal_count() - r_off,
            c_len,
            "cubel-global-sub: fine conductor loops {} != coarse loops {c_len}",
            system.mesh.internal_count() - r_off
        );
        sub_nesting_report(system, coarse, r_off, c_len);
        if std::env::var("FASTERHENRY_CUBEL_SUB_CHECK")
            .map(|v| v.trim() == "1")
            .unwrap_or(false)
        {
            sub_identity_check(system, coarse, r_off, c_len);
        }

        Ok(Self::CubelGlobalSub(Box::new(CubelSubState {
            local,
            factors,
            r_off,
            c_len,
        })))
    }

    /// Builds the v2 conv state at `omega` (see `ConvCore::build_state`).
    fn conv_kind(system: &IterativeSystem, omega: f64) -> Result<Self, SolveError> {
        let core = ConvCore::build_state(system, omega)?;
        Ok(Self::Conv(Box::new(core)))
    }

    /// Builds the two-level (D17 v0) state at `omega`.
    ///
    /// The pattern radius is independently tunable via
    /// `FASTERHENRY_TWOL_RADIUS_MULT` (default 2.0): unlike the
    /// cubel-global knob it defaults ABOVE the operator radius because
    /// the cross-layer correction is the point - at the operator
    /// radius the cross set is the blind gap that produced the 365
    /// -iteration row. Knob scales only the precond pattern (never the
    /// operator), so `A` is unchanged.
    fn two_level(system: &IterativeSystem, omega: f64) -> Result<Self, SolveError> {
        crate::memgate::set_stage(crate::memgate::STAGE_PRECOND);
        let radius_mult: f64 = std::env::var("FASTERHENRY_TWOL_RADIUS_MULT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(2.0);
        if !(0.0..=f64::INFINITY).contains(&radius_mult) || radius_mult <= 0.0 {
            eprintln!(
                "two-level: warning: FASTERHENRY_TWOL_RADIUS_MULT={radius_mult} invalid; using 2.0"
            );
        }
        let radius_mult = if radius_mult > 0.0 { radius_mult } else { 2.0 };
        let pattern = TwoLevel::build(
            &system.mesh,
            &system.filaments,
            &system.resistances,
            system.operator.near_radius() * radius_mult,
        )
        .map_err(SolveError::from)?;
        let factors = pattern.factor::<T>(omega);
        Ok(Self::TwoLevel(Box::new(TwoLevelState { pattern, factors })))
    }

    /// Applies the preconditioner to `v`. `op` is the system's own operator
    /// (`Z`); only the multiplicative two-level precond needs it, for the
    /// residual `v - Z B_b^-1 v` - the rest ignore it.
    fn run(&self, v: &[T], op: &dyn Fn(&[T]) -> Vec<T>) -> Vec<T> {
        match self {
            Self::Jacobi(d) => v.iter().zip(d.iter()).map(|(&v, &d)| v / d).collect(),
            Self::Identity => v.to_vec(),
            Self::Cubel(state) => state.pattern.apply_blocks(&state.factors, v),
            Self::CubelGlobal(state) => {
                CubelGlobal::apply_global(state.factors.as_ref(), state.global.n(), v)
            }
            Self::CubelGlobalSub(state) => {
                // Multiplicative two-level:
                //   z = B_b^-1 v + P A_c^-1 R (v - Z B_b^-1 v)
                // B_b is the block-local fine smoother (full rank); the coarse
                // step corrects the conductor part of the residual it leaves.
                let zb = state.local.pattern.apply_blocks(&state.local.factors, v);
                let w = op(&zb);
                let mut rc = vec![T::zero(); state.c_len];
                for j in 0..state.c_len {
                    rc[j] = v[state.r_off + j] - w[state.r_off + j];
                }
                let sc = CubelGlobal::apply_global(state.factors.as_ref(), state.c_len, &rc);
                let mut z = zb;
                for j in 0..state.c_len {
                    z[state.r_off + j] += sc[j];
                }
                z
            }
            Self::TwoLevel(state) => state.pattern.apply(&state.factors, v),
            Self::Conv(core) => conv_apply::<T>(core, v),
        }
    }
}

/// Index-only debug report: are the fine's conductor rows (each branch mapped
/// to its segment) the same loops as the coarse's rows? Prints a summary and,
/// on mismatches, dumps the first ones to `cubel_sub_nesting.txt`. This is the
/// "test current"-free structural check of the `R`/`P` premise (the loop sets
/// are nested); it is a build-time diagnostic, not an ongoing harness.
fn sub_nesting_report(system: &IterativeSystem, coarse: &CoarseSystem, r_off: usize, c_len: usize) {
    let per_segment = system.per_segment();
    let mut seg_of = vec![usize::MAX; system.mesh.branch_count()];
    let mut b = 0usize;
    for (s, &n) in per_segment.iter().enumerate() {
        for _ in 0..n {
            seg_of[b] = s;
            b += 1;
        }
    }
    let norm = |row: &[(usize, f64)]| -> Vec<(usize, i64)> {
        let mut v: Vec<(usize, i64)> = row.iter().map(|&(i, s)| (i, s as i64)).collect();
        v.sort_unstable();
        v
    };
    let mut mism = 0usize;
    let mut dump = String::new();
    for j in 0..c_len {
        let fine: Vec<(usize, f64)> = system
            .mesh
            .row(r_off + j)
            .iter()
            .map(|&(br, sg)| (seg_of[br], sg))
            .collect();
        let a = norm(&fine);
        let c = norm(coarse.mesh.row(j));
        if a != c {
            mism += 1;
            if mism <= 20 {
                dump.push_str(&format!("coarse loop {j}: fine {a:?} | coarse {c:?}\n"));
            }
        }
    }
    eprintln!(
        "cubel-global-sub: nesting: {c_len} coarse loops, {mism} mismatched vs the fine conductor rows"
    );
    if mism > 0 {
        let path = "cubel_sub_nesting.txt";
        if std::fs::write(path, &dump).is_ok() {
            eprintln!("cubel-global-sub: wrote {path} (first mismatches)");
        }
    }
}

/// The fine branch -> segment map (the structure of the aggregation `S`) from
/// the per-segment counts: branch `b` belongs to segment `s` iff `b` falls in
/// `s`'s consecutive range (the mesh's `first_branch` ordering).
fn fine_branch_segment(per_segment: &[usize]) -> Vec<usize> {
    let mut seg_of = Vec::with_capacity(per_segment.iter().sum());
    for (s, &n) in per_segment.iter().enumerate() {
        for _ in 0..n {
            seg_of.push(s);
        }
    }
    seg_of
}

/// The worst residual of the value-side identity used by `cubel-global-sub`:
/// the loop-space restriction `R` (the conductor-block selection) and the
/// branch aggregation `S` must agree, i.e. `M_c^T (R x) == S (M_f^T x)` for
/// every fine loop current `x`. Both sides are linear, so the unit vectors
/// (the incidence rows) are exhaustive; a general deterministic current is
/// checked as a cross-check.
fn sub_identity_worst(
    system: &IterativeSystem,
    coarse: &CoarseSystem,
    r_off: usize,
    c_len: usize,
) -> f64 {
    let seg_of = fine_branch_segment(system.per_segment());
    let bc = coarse.mesh.branch_count();
    let fl = system.mesh.internal_count();

    // i_c = S (M_f^T x): the fine branch currents aggregated per segment.
    let s_apply = |xf: &[f64]| -> Vec<f64> {
        let mut ic = vec![0.0; bc];
        for (e, &x) in xf.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            for &(b, sgn) in system.mesh.row(e) {
                ic[seg_of[b]] += x * sgn;
            }
        }
        ic
    };
    // x_c = R x: the fine's conductor block (e >= r_off) -> the coarse.
    let r_apply = |xf: &[f64]| -> Vec<f64> {
        let mut xc = vec![0.0; c_len];
        for e in r_off..r_off + c_len {
            xc[e - r_off] = xf[e];
        }
        xc
    };
    // i_c = M_c^T x_c.
    let mc_t_apply = |xc: &[f64]| -> Vec<f64> {
        let mut ic = vec![0.0; bc];
        for (j, &x) in xc.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            for &(b, sgn) in coarse.mesh.row(j) {
                ic[b] += x * sgn;
            }
        }
        ic
    };
    let worst_of = |xf: &[f64]| -> f64 {
        s_apply(xf)
            .iter()
            .zip(&mc_t_apply(&r_apply(xf)))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    };

    let mut worst = 0.0f64;
    for e in 0..fl {
        let mut xf = vec![0.0; fl];
        xf[e] = 1.0;
        worst = worst.max(worst_of(&xf));
    }
    let xf: Vec<f64> = (0..fl)
        .map(|i| ((i.wrapping_mul(2654435761)) % 1000) as f64 / 1000.0 - 0.5)
        .collect();
    worst.max(worst_of(&xf))
}

/// Value-side debug wrapper: runs [`sub_identity_worst`], prints it, and dumps
/// `S`/`R` and the result to `cubel_sub_check.txt`. Gated on
/// `FASTERHENRY_CUBEL_SUB_CHECK=1`.
fn sub_identity_check(system: &IterativeSystem, coarse: &CoarseSystem, r_off: usize, c_len: usize) {
    let seg_of = fine_branch_segment(system.per_segment());
    let bf = system.mesh.branch_count();
    let bc = coarse.mesh.branch_count();
    let fl = system.mesh.internal_count();
    let worst = sub_identity_worst(system, coarse, r_off, c_len);

    eprintln!(
        "cubel-global-sub: identity  M_c^T R x == S M_f^T x : worst |diff| = {worst:e}  (unit + general)"
    );

    let mut out = String::new();
    out.push_str("# cubel-global-sub debug dump\n");
    out.push_str(&format!(
        "# fine: {fl} internal loops, {bf} branches; coarse: {c_len} loops, {bc} branches\n"
    ));
    out.push_str(&format!("# r_off (first conductor loop) = {r_off}\n"));
    out.push_str("# S: fine branch -> segment (the aggregation structure)\n");
    for b in 0..bf {
        out.push_str(&format!("S fine branch {b} -> segment {}\n", seg_of[b]));
    }
    out.push_str("# R/P: fine conductor loop -> coarse loop (identity offset)\n");
    for j in 0..c_len {
        out.push_str(&format!("R fine loop {} -> coarse loop {j}\n", r_off + j));
    }
    out.push_str(&format!("# identity worst |diff| = {worst:e}\n"));
    if std::fs::write("cubel_sub_check.txt", &out).is_ok() {
        eprintln!("cubel-global-sub: wrote cubel_sub_check.txt");
    }
}

fn schur_complement_gmres<T>(
    ep: &DMatrix<T>,
    pp: &DMatrix<T>,
    apply_ee: impl Fn(&[T]) -> Vec<T>,
    precond: &Precond<T>,
    params: &GmresParams,
    frequency: f64,
) -> Result<(DMatrix<T>, Vec<GmresOutcome>), SolveError>
where
    T: ComplexField<RealField = f64> + Copy + Sync + Send + FaerComplexField,
    // The conv engine's run converts T <-> C64 at the boundary.
    T: crate::cubel::FieldParts,
{
    let internal = ep.nrows();
    if internal == 0 {
        return Ok((pp.clone(), Vec::new()));
    }
    let mut solved = DMatrix::zeros(internal, ep.ncols());
    let mut outcomes = Vec::with_capacity(ep.ncols());
    for port in 0..ep.ncols() {
        let rhs: Vec<T> = ep.column(port).iter().copied().collect();
        let mut x = vec![T::zero(); internal];
        let outcome = gmres(
            &apply_ee,
            |v: &[T]| precond.run(v, &apply_ee),
            &rhs,
            &mut x,
            params,
        );
        if !outcome.converged {
            return Err(SolveError::NotConverged {
                frequency,
                port,
                iterations: outcome.iterations,
                relative_residual: outcome.relative_residual,
            });
        }
        solved.column_mut(port).copy_from_slice(&x);
        outcomes.push(outcome);
    }
    Ok((pp - ep.transpose() * solved, outcomes))
}

/// `M_{0..loops} · v` for a branch vector `v`: the first `loops` loops.
fn gather<T: ComplexField<RealField = f64> + Copy>(
    mesh: &MeshMatrix,
    loops: usize,
    v: &[T],
) -> Vec<T> {
    (0..loops)
        .into_par_iter()
        .with_min_len(256)
        .map(|i| {
            mesh.row(i)
                .iter()
                .fold(T::zero(), |sum, &(b, sign)| sum + v[b].scale(sign))
        })
        .collect()
}

/// `M · v` over every loop, internal and port.
fn gather_all(mesh: &MeshMatrix, v: &[f64]) -> Vec<f64> {
    gather(mesh, mesh.loop_count(), v)
}

/// `Σₐ Σ_b sₐ s_b L_ab` over the branches of one loop — the loop's
/// diagonal entry of `M L Mᵀ` — or, for loops longer than
/// [`EXACT_DIAGONAL_MAX_LOOP_LENGTH`], the sum of the self terms alone.
fn loop_inductance(filaments: &[Filament], row: &[(usize, f64)]) -> Result<f64, KernelError> {
    let own: f64 = row
        .iter()
        .map(|&(branch, _)| self_inductance(&filaments[branch]))
        .sum();
    if row.len() > EXACT_DIAGONAL_MAX_LOOP_LENGTH {
        return Ok(own);
    }
    let mut mutual = 0.0;
    for (k, &(a, sign_a)) in row.iter().enumerate() {
        for &(b, sign_b) in &row[k + 1..] {
            let m = mutual_inductance(&filaments[a], &filaments[b]).map_err(|e| e.at(a, b))?;
            mutual += sign_a * sign_b * m;
        }
    }
    Ok(own + 2.0 * mutual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Node, NodeId, SegmentDef};

    const COPPER: f64 = 5.8e7;

    /// A unit square ring, uniformly subdivided `2x2` per segment, with a
    /// diagonal port and a stub - enough segments for several loop types.
    fn subdivided_loop() -> IterativeSystem {
        let mut geometry = Geometry::new();
        for point in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ] {
            geometry.add_node(Node::from(point)).unwrap();
        }
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)] {
            geometry
                .add_segment(SegmentDef::new(NodeId(a), NodeId(b), 1e-3, 1e-3, COPPER))
                .unwrap();
        }
        let ports = [Port::new(NodeId(1), NodeId(4))];
        IterativeSystem::assemble(
            &geometry,
            &ports,
            &Discretization::uniform(2, 2),
            &IterativeParams::default(),
        )
        .unwrap()
    }

    #[test]
    fn sub_identity_holds() {
        let system = subdivided_loop();
        let coarse = system
            .coarse
            .as_ref()
            .expect("subdivided: the coarse twin must exist");
        let r_off = system.intra_loop_count();
        let c_len = coarse.mesh.internal_count();
        assert_eq!(
            system.mesh.internal_count() - r_off,
            c_len,
            "fine conductor loops ({}) != coarse loops ({c_len})",
            system.mesh.internal_count() - r_off
        );
        let worst = sub_identity_worst(&system, coarse, r_off, c_len);
        assert!(
            worst < 1e-12,
            "M_c^T R x != S M_f^T x: worst |diff| = {worst:e}"
        );
    }

    #[test]
    fn cubel_global_sub_builds_on_subdivided() {
        let system = subdivided_loop();
        let precond: Precond<C64> = Precond::cubel_global_sub(&system, 1.0e7).unwrap();
        match precond {
            Precond::CubelGlobalSub(state) => {
                assert!(state.c_len > 0 && state.r_off > 0);
                assert!(state.factors.is_some());
            }
            _ => panic!("expected CubelGlobalSub"),
        }
    }
}
