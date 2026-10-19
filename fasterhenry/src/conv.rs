//! Conv (matrix-free) precond kernel core, v2 (D17 re-scope, NOTES
//! 2026-10-19/20): pour planes are quasi-uniform manhattan meshes, so
//! a layer-pair coupling operator is approximately a 2D Toeplitz
//! operator in the XY displacement - it depends on the displacement
//! only. The precond reuses that structure to approximately invert
//! the local couplings: sample the EXACT mutual kernel on a
//! displacement raster, FFT to spectral form, apply a DAMPED
//! spectral inverse. Precond only: the pFFT operator stays the
//! equation, so this machinery can never change the answer
//! (GMRES-referee contract); inexactness costs iterations only.
//!
//! Failure-mode guards (from the design session, NOTES 2026-10-19):
//! - no unbounded amplification: inversion is regularized
//!   Wiener-style, `h / (h^2 + eps)` with `eps = (damp * peak)^2`,
//!   so the spectral gain stays bounded by
//!   `1 / (2 * damp * peak)` at every frequency, including the
//!   kernel's sign-flip holes;
//! - wrap-around: the raster views the linear kernel as a CIRCULAR
//!   convolution; displacement support that exceeds the raster
//!   contaminates the far spectrum. That is a precond-quality effect
//!   (the raster sizing is ladder territory), never accuracy.

use num_complex::Complex64;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner};

/// Damped spectral inverse of one displacement-domain kernel.
pub struct ConvKernel {
    /// Raster side (odd; the zero displacement lives at cell
    /// `(n / 2, n / 2)`).
    n: usize,
    /// Raster cell pitch (metres).
    cell_m: f64,
    /// The kernel's wrapped spectrum, `FFT2(kernel)`; kept for tests
    /// and future diagnostics.
    kernel_hat: Vec<Complex64>,
    /// The displacement-domain samples before the FFT (the same
    /// wrapped layout). The real part is what a single-source probe
    /// reads back; complex-sample kernels keep the full value here
    /// and only the real `LayerPairOperator` kernels are introspected.
    kernel_time: Vec<Complex64>,
    /// The damped inverse spectrum - what the apply multiplies by.
    inverse_hat: Vec<Complex64>,
    /// The regularization floor used, `(damp * peak)^2`.
    epsilon: f64,
    /// Cached FFT plans: planning again per apply would dominate the
    /// engine's per-iteration cost (2026-10-20; the acceptance deck's
    /// debug run measured planning-heavy applies at ~100 s vs 7 s
    /// cached at release).
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
}

impl ConvKernel {
    /// Builds the damped spectral inverse. `sample(dispx, dispy)`
    /// supplies the EXACT kernel value at an XY displacement point in
    /// metres; it must be even in both axes for the wrapped spectrum
    /// to stay real (an odd component folds into sign-aliased noise;
    /// the spectral multiply only preserves magnitude then).
    /// `damp` in `(0, 1)` sets the regularization floor.
    pub fn build(n: usize, cell_m: f64, sample: impl Fn(f64, f64) -> f64, damp: f64) -> Self {
        Self::build_complex(
            n,
            cell_m,
            |dx, dy| Complex64::new(sample(dx, dy), 0.0),
            damp,
        )
    }

    /// Complex-sample build. The Wiener inverse becomes
    /// `conj(h) / (|h|^2 + eps)` - the Tikhonov-regularized pseudo
    /// inverse of the family operator, bounded by `1 / (2 * sqrt eps)`
    /// at every frequency for COMPLEX kernels too (the GMRES
    /// failmode-2 guard carries over verbatim). For real samples this
    /// reduces exactly to the re-signed real form.
    pub fn build_complex(
        n: usize,
        cell_m: f64,
        sample: impl Fn(f64, f64) -> Complex64,
        damp: f64,
    ) -> Self {
        assert!(n >= 3, "conv kernel raster too small: {n}");
        assert!(
            cell_m.is_finite() && cell_m > 0.0,
            "conv kernel pitch must be a positive metre value: {cell_m}"
        );
        let bound = "conv kernel damping must lie in (0, 1):";
        assert!(
            damp.is_finite() && damp > 0.0 && damp < 1.0,
            "{bound} {damp}"
        );
        // RAW wrapped layout: the zero displacement at plane index 0,
        // negative displacements wrapping to the top rows/columns.
        // The FFT convolution identity
        // `IFFT2(FFT2(y) * H)[j] = sum_i y[i] * k[(j - i) mod n^2]`
        // is expressed in exactly this layout; a CENTERED kernel
        // inserts a spurious circular shift between the operator and
        // every field read (caught by the layer-pair tripwire).
        let half = n as i64 / 2;
        let raw_displacement = |index: i64| -> f64 {
            (if index > half {
                index - n as i64
            } else {
                index
            }) as f64
                * cell_m
        };
        let mut plane: Vec<Complex64> = Vec::with_capacity(n * n);
        for row in 0..n as i64 {
            for col in 0..n as i64 {
                plane.push(sample(raw_displacement(row), raw_displacement(col)));
            }
        }
        let kernel_time: Vec<Complex64> = plane.clone();
        let fft = FftPlanner::new().plan_fft_forward(n);
        apply_fft2(&*fft, &mut plane);
        let mut peak = 0.0f64;
        for value in &plane {
            peak = peak.max(value.norm());
        }
        let mut inverse_hat = Vec::with_capacity(n * n);
        let fft: Arc<dyn Fft<f64>> = FftPlanner::new().plan_fft_forward(n);
        let ifft: Arc<dyn Fft<f64>> = FftPlanner::new().plan_fft_inverse(n);
        if peak == 0.0 {
            // A structurally zero kernel (perpendicular families, or a
            // cross layer at DC) has no coupling to invert: the Wiener
            // formula `conj(h) / (|h|^2 + eps)` would produce 0/0 = NaN
            // per band, so the inverse is defined as all-zero instead
            // (2026-10-20, accepted-deck diagnosis).
            for _ in 0..plane.len() {
                inverse_hat.push(Complex64::new(0.0, 0.0));
            }
            return ConvKernel {
                n,
                cell_m,
                kernel_hat: plane,
                kernel_time,
                inverse_hat,
                epsilon: 0.0,
                fft,
                ifft,
            };
        }
        let epsilon = (damp * peak).powi(2);
        for value in &plane {
            // Tikhonov-regularized pseudo inverse: `conj(h) / (|h|^2 +
            // eps)`. A real spectrum reduces to the re-signed real
            // form (the coupling sign of negative-mutual bands is
            // preserved); a complex kernel gets the least-squares
            // flavor, bounded at 1 / (2 * sqrt(eps)) either way.
            let denom = value.norm().powi(2) + epsilon;
            inverse_hat.push(value.conj() / denom);
        }
        ConvKernel {
            n,
            cell_m,
            kernel_hat: plane,
            kernel_time,
            inverse_hat,
            epsilon,
            fft,
            ifft,
        }
    }

    /// The apply: damped circular convolution with the inverse
    /// spectrum, in place. Layout: the field is a row-major `n x n`
    /// plane, `cell_m` strides, index `(row, col)` matching the
    /// kernel raster. Cost: two forward + two inverse FFT passes of
    /// side n and one spectral multiply.
    pub fn apply(&self, field: &mut [Complex64]) {
        assert_eq!(
            field.len(),
            self.n * self.n,
            "field size must equal the kernel raster"
        );
        let n = self.n;
        apply_fft2(&*self.fft, field);
        for (value, gain) in field.iter_mut().zip(&self.inverse_hat) {
            *value = *value * *gain;
        }
        apply_fft2(&*self.ifft, field);
        // The circular-convolution identity
        // `IFFT2_unnorm(FFT2_unnorm(y) * H) / n^2 = y (asterisk) h`
        // pins the normalization: rustfft leaves both directions
        // unnormalized, so the division cancels the double inverse
        // scale exactly.
        let norm = 1.0 / (n * n) as f64;
        for value in field.iter_mut() {
            *value = *value * norm;
        }
    }

    /// The forward-perusal apply: circular convolution with the raw
    /// kernel spectrum (no damping, no inversion). Test round-trips
    /// and diagnostics; the engine's preconditioned operator is
    /// `operator_apply * apply`, not this.
    pub fn apply_forward(&self, field: &mut [Complex64]) {
        assert_eq!(
            field.len(),
            self.n * self.n,
            "field size must equal the kernel raster"
        );
        let n = self.n;
        apply_fft2(&*self.fft, field);
        for (value, gain) in field.iter_mut().zip(&self.kernel_hat) {
            *value = *value * *gain;
        }
        apply_fft2(&*self.ifft, field);
        let norm = 1.0 / (n * n) as f64;
        for value in field.iter_mut() {
            *value = *value * norm;
        }
    }

    /// Raster side.
    pub fn raster(&self) -> usize {
        self.n
    }

    /// Cell pitch (metres).
    pub fn cell_m(&self) -> f64 {
        self.cell_m
    }

    /// The regularization floor recorded at build time.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Peak magnitude of the spectral inverse (the bounded-gain
    /// evidence for the GMRES failmode-2 guard).
    pub fn inverse_peak_gain(&self) -> f64 {
        self.inverse_hat
            .iter()
            .map(|v| v.norm())
            .fold(0.0f64, f64::max)
    }
}

/// Forward or inverse FFT pass over the row-major plane: rows, then
/// strided columns. Single-direction normalization is left to the
/// caller (rustfft is unnormalized both ways).
fn apply_fft2(fft: &dyn Fft<f64>, plane: &mut [Complex64]) {
    let n = fft.len();
    assert!(plane.len() % n == 0, "plane rows must each hold {n} points");
    for row in 0..plane.len() / n {
        fft.process(&mut plane[row * n..(row + 1) * n]);
    }
    let rows = plane.len() / n;
    let mut column = vec![Complex64::new(0.0, 0.0); n];
    for col in 0..n {
        for row in 0..rows {
            column[row] = plane[row * n + col];
        }
        fft.process(&mut column);
        for row in 0..rows {
            plane[row * n + col] = column[row];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: f64 = 2.0;

    #[test]
    fn structurally_zero_kernel_inverts_to_zero_without_nan() {
        // A structurally zero kernel (cross-axis families, cross-layer
        // at DC) has no spectral energy to invert: the naive Wiener
        // formula is 0/0 = NaN per band, and NaN spreads silently
        // through f64::max. The guard must produce an all-zero
        // inverse with no epsilon floor, a kernel_hat that stays
        // zero, and an apply that leaves finite-time zeros - pins the
        // 2026-10-20 summary's guard directly (audit F1).
        let n = 5;
        let cell = 0.1;
        let kernel = ConvKernel::build_complex(n, cell, |_, _| Complex64::new(0.0, 0.0), 0.01);
        assert_eq!(kernel.epsilon(), 0.0, "zero-peak takes no eps floor");
        assert_eq!(
            kernel.inverse_peak_gain(),
            0.0,
            "an all-zero inverse has no gain - and not NaN"
        );
        for value in &kernel.kernel_hat {
            assert_eq!(value.norm(), 0.0, "zero sample raster must stay zero");
        }
        let mut field = vec![Complex64::new(3.25, -1.5); n * n];
        kernel.apply(&mut field);
        for value in &field {
            assert!(value.is_finite(), "guard apply must not produce NaN");
            assert_eq!(
                value.norm(),
                0.0,
                "an all-zero inverse smears nothing: got {value}"
            );
        }
    }

    #[test]
    fn const_kernel_pins_the_normalization() {
        // A constant kernel's spectrum is a single spike at DC with
        // height m0 = C * n^2; applied to an all-ones field the
        // circular-convolution normalization must reproduce exactly
        // m0 / (m0^2 + eps) per cell.
        let n = 5;
        let cell = 0.1;
        let kernel = ConvKernel::build(n, cell, |_, _| C, 0.5);
        let m0 = C * (n * n) as f64;
        let eps = (0.5 * m0).powi(2);
        let expected = m0 / (m0 * m0 + eps);
        let mut field = vec![Complex64::new(1.0, 0.0); n * n];
        kernel.apply(&mut field);
        for value in field {
            assert!(
                (value.re - expected).abs() < 1e-8 && value.im.abs() < 1e-8,
                "got {value}, expected {expected}-ish"
            );
        }
    }

    #[test]
    fn spectral_gain_stays_bounded_through_sign_flips() {
        // The Lorentzian has spectrum sign flips near the raster
        // corners (fine-feature kernel); the Wiener floor must hold
        // the peak gain at 1 / (2 * damp * peak) - the failmode-2
        // guard, never an unbounded inverse.
        let kernel = ConvKernel::build(
            33,
            0.05,
            |dx, dy| {
                let d2 = dx * dx + dy * dy;
                1.0 / (1.0 + d2 / 0.04)
            },
            0.01,
        );
        let eps = kernel.epsilon();
        let bound = 0.5 / eps.sqrt();
        let peak = kernel.inverse_peak_gain();
        assert!(
            peak <= bound * (1.0 + 1e-9),
            "peak gain {peak} exceeded the Wiener bound {bound}"
        );
        // The bound is tight enough to be meaningfully reached.
        assert!(
            peak > 0.5 * bound,
            "gain {peak} never exercised the bound; eps={eps}"
        );
    }

    #[test]
    fn damped_inverse_roundtrips_low_frequency_fields() {
        // With the damping floor far below a low-frequency field's
        // spectrum, apply and re-wet with the raw kernel recover the
        // field: the inexactness is confined to the spectral holes.
        // Kernel chosen to satisfy the wrap premise: `1/(1+d^2/0.04)`
        // reaches only ~6% of its peak at the raster boundary, so the
        // circular view adds negligible contamination.
        let kernel = ConvKernel::build(
            33,
            0.05,
            |dx, dy| {
                let d2 = dx * dx + dy * dy;
                1.0 / (1.0 + d2 / 0.04)
            },
            0.01,
        );
        let n = kernel.raster();
        let cell = kernel.cell_m();
        let mut field = Vec::with_capacity(n * n);
        for row in 0..n as i64 {
            for col in 0..n as i64 {
                let dx = (row - n as i64 / 2) as f64 * cell;
                let dy = (col - n as i64 / 2) as f64 * cell;
                field.push(Complex64::new((-(dx * dx + dy * dy) / 0.5).exp(), 0.0));
            }
        }
        kernel.apply(&mut field);
        kernel.apply_forward(&mut field);
        for value in &field {
            assert!(value.im.abs() < 1e-4, "complex leakage {value}");
        }
        let mut again = Vec::with_capacity(n * n);
        for row in 0..n as i64 {
            for col in 0..n as i64 {
                let dx = (row - n as i64 / 2) as f64 * cell;
                let dy = (col - n as i64 / 2) as f64 * cell;
                again.push((-(dx * dx + dy * dy) / 0.5).exp());
            }
        }
        let mut top = 0.0f64;
        for truth in &again {
            top = top.max(truth.abs());
        }
        let mut worst = 0.0f64;
        // Absolute error against the field top: the preconditioner
        // claim is that the spectral holes only SKIP what is there -
        // a few-percent absolute attenuation of low-frequency
        // components, never a shape wreck. (Relative error is
        // meaningless at the field tails: a 3% absolute leak reads
        // 100% next to truth ~ 0.1 * top.)
        let mut worst_at = 0;
        for (index, truth) in again.iter().enumerate() {
            let error = (field[index].re - truth).abs() / top;
            if error > worst {
                worst = error;
                worst_at = index;
            }
        }
        let wr = worst_at / n;
        let wc = worst_at % n;
        let half = n as i64 / 2;
        eprintln!(
            "DBG worst at row {} col {} dx {} dy {} truth {:.5} got {:.5} (rel err {:.4})",
            wr,
            wc,
            (wr as i64 - half) * 5,
            (wc as i64 - half) * 5,
            again[worst_at],
            field[worst_at].re,
            (field[worst_at].re - again[worst_at]).abs() / again[worst_at].abs().max(1e-12)
        );
        assert!(
            worst < 0.05,
            "round-trip error {worst} beyond the damping scope"
        );
    }
}

// ---------------------------------------------------------------------------
// Layer-pair raster operator (v2 slice 2b, increment 1).
//
// The branch-space near operator's conv-representable part: every
// manhattan filament projects onto a shared XY raster (one cell per
// mesh pitch, the strict quasi-uniform posture - `CONV_GRADE = 0`
// default), layer-pair couplings ride per-family kernels sampled from
// the EXACT mutual inductance of representative filaments, and the
// apply is a circular convolution per (layer pair, orientation
// family). Sign conventions: a filament's direction sign folds into
// the raster write and the flux read, so antiparallel mesh
// conventions stay exact.
//
// Deliberate v2-default simplifications (each booked in NOTES):
// - ONE shared raster geometry for all layers (per-layer pitch is a
//   grade-knob relaxation);
// - filaments that are neither x- nor y-directed (via barrels,
//   diagonal traces) are not conv-covered - they ride the explicit
//   tab blocks / the branch-CSR spine;
// - the xy-family kernel rides the fork's `SampledFilaments` path
//   (~1e-3 method accuracy) - precond-only, never accuracy.
// ---------------------------------------------------------------------------

use crate::census::branch_layers;
use crate::filament::Filament;
use crate::geometry::{Node, Segment};
use crate::inductance::{self, KernelError};
use nalgebra::Vector3;

/// One copper layer's conv-covered population: per orientation axis,
/// the branch indices and their direction signs relative to the +axis
/// reference.
struct LayerPlane {
    /// Per axis (0 = x-directed, 1 = y-directed): `(branch, sign)`.
    filaments: [Vec<(usize, f64)>; 2],
}

/// A filament's raster placement (crate-internal type returned by
/// `LayerPairOperator::placement`): the positional projection plus the
/// filament's direction sign relative to the +axis reference (the
/// single source of truth for the write/read folding).
#[derive(Clone, Copy)]
pub(crate) struct Place {
    layer: usize,
    axis: usize,
    cell: usize,
    sign: f64,
}

/// The conv-representable part of the branch-space near operator:
/// per layer-pair, per orientation-family circular-convolution
/// kernels over the shared raster.
pub struct LayerPairOperator {
    n: usize,
    cell_m: f64,
    place: Vec<Option<Place>>,
    layers: Vec<LayerPlane>,
    /// `kernels[li][lj][src_axis][dst_axis]`: the exact-kernel
    /// spectrum for couplings from layer `li`'s `src_axis` population
    /// onto layer `lj`'s `dst_axis` population; `None` where either
    /// population is empty.
    kernels: Vec<Vec<[[Option<ConvKernel>; 2]; 2]>>,
}

/// The flow axis of a filament's direction: `Some((axis, sign))` for
/// x- or y-directed filaments, `None` for via barrels and diagonals.
fn flow_axis(direction: Vector3<f64>) -> Option<(usize, f64)> {
    let (x, y) = (direction.x.abs(), direction.y.abs());
    if x >= 0.5 && x >= y {
        Some((0, direction.x.signum()))
    } else if y >= 0.5 {
        Some((1, direction.y.signum()))
    } else {
        None
    }
}

/// The median of a non-empty slice (panics on empty: every caller
/// guards).
fn median(values: &mut Vec<f64>) -> f64 {
    assert!(!values.is_empty(), "median of an empty population");
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The modal pitch of a point cloud: the median gap between
/// consecutive DISTINCT coordinate values, over both projections.
/// Exact on uniform grids; a robust estimator on near-uniform ones
/// (the grade knob revisits irregular decks).
fn modal_pitch(points: &[[f64; 2]]) -> f64 {
    let mut gaps: Vec<f64> = Vec::new();
    for axis in 0..2 {
        let mut values: Vec<f64> = points.iter().map(|p| p[axis]).collect();
        values.sort_by(f64::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * a.abs().max(1.0));
        for pair in values.windows(2) {
            let gap = pair[1] - pair[0];
            if gap > 0.0 {
                gaps.push(gap);
            }
        }
    }
    assert!(
        !gaps.is_empty(),
        "raster pitch: a degenerate point cloud has no gaps"
    );
    median(&mut gaps)
}

impl LayerPairOperator {
    /// Builds the operator from the deck's filaments: z-clustered
    /// layers (`census::branch_layers`), one shared raster at the
    /// modal pitch, per-layer-pair per-family kernels sampled from
    /// the exact mutual kernel on each population's median geometry.
    pub fn build(filaments: &[Filament]) -> Result<Self, KernelError> {
        assert!(!filaments.is_empty(), "layer-pair operator: no filaments");
        let layer_of = branch_layers(filaments);
        let layer_count = layer_of.iter().max().map_or(1, |&m| m + 1);

        // Population pass: per (layer, axis) collect (branch, sign);
        // per member record the centre for the raster geometry.
        let mut points: Vec<[f64; 2]> = Vec::new();
        let mut members: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layer_count * 2];
        for (branch, filament) in filaments.iter().enumerate() {
            let Some((axis, sign)) = flow_axis(filament.direction()) else {
                continue;
            };
            let centre = filament.center();
            points.push([centre.x, centre.y]);
            members[layer_of[branch] * 2 + axis].push((branch, sign));
        }
        assert!(
            !points.is_empty(),
            "layer-pair operator: no manhattan filaments"
        );
        let cell_m = modal_pitch(&points);

        // Raster geometry: the covered centres' bbox plus a margin of
        // four cells per side (wrap hygiene), side capped loudly.
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for point in &points {
            for axis in 0..2 {
                lo[axis] = lo[axis].min(point[axis]);
                hi[axis] = hi[axis].max(point[axis]);
            }
        }
        const MARGIN_CELLS: i64 = 4;
        const MAX_SIDE: usize = 1024;
        let span_cells = |lo: f64, hi: f64| ((hi - lo) / cell_m).ceil() as i64 + 1;
        let side = span_cells(lo[0], hi[0]).max(span_cells(lo[1], hi[1])) + 2 * MARGIN_CELLS;
        assert!(
            side <= MAX_SIDE as i64,
            "layer-pair raster side {side} exceeds the {MAX_SIDE} cap - coarsen the pitch knob"
        );
        let n = (side as usize).max(3);
        // Cell indexing MUST match the kernel raster's convention:
        // the major axis (row) indexes the X displacement, the minor
        // (col) the Y displacement - the same assignment
        // `ConvKernel::build` uses when sampling. Getting this
        // backwards transposes every cross-family kernel silently.
        let origin_x = lo[0] - MARGIN_CELLS as f64 * cell_m;
        let origin_y = lo[1] - MARGIN_CELLS as f64 * cell_m;
        let cell_of = |x: f64, y: f64| -> usize {
            let row = (((x - origin_x) / cell_m).round().max(0.0)) as usize;
            let col = (((y - origin_y) / cell_m).round().max(0.0)) as usize;
            row.min(n - 1) * n + col.min(n - 1)
        };

        let mut place: Vec<Option<Place>> = vec![None; filaments.len()];
        let mut layers: Vec<LayerPlane> = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let mut by_axis = [Vec::new(), Vec::new()];
            for axis in 0..2 {
                for &(branch, sign) in &members[layer * 2 + axis] {
                    let centre = filaments[branch].center();
                    let cell = cell_of(centre.x, centre.y);
                    place[branch] = Some(Place {
                        layer,
                        axis,
                        cell,
                        sign,
                    });
                    by_axis[axis].push((branch, sign));
                }
            }
            layers.push(LayerPlane { filaments: by_axis });
        }

        // Kernel sampling: per layer pair, per (src, dst) axis family,
        // on the median geometry of each population at its own z.
        let mut kernels: Vec<Vec<[[Option<ConvKernel>; 2]; 2]>> = Vec::with_capacity(layer_count);
        for li in 0..layer_count {
            let mut row: Vec<[[Option<ConvKernel>; 2]; 2]> = Vec::with_capacity(layer_count);
            for lj in 0..layer_count {
                let mut families: [[Option<ConvKernel>; 2]; 2] = Default::default();
                for src_axis in 0..2 {
                    if members[li * 2 + src_axis].is_empty() {
                        continue;
                    }
                    for dst_axis in 0..2 {
                        if members[lj * 2 + dst_axis].is_empty() {
                            continue;
                        }
                        let kernel = sample_family_kernel(
                            filaments,
                            &members[li * 2 + src_axis],
                            &members[lj * 2 + dst_axis],
                            src_axis,
                            dst_axis,
                            li == lj,
                            n,
                            cell_m,
                        )?;
                        families[src_axis][dst_axis] = Some(kernel);
                    }
                }
                row.push(families);
            }
            kernels.push(row);
        }

        Ok(Self {
            n,
            cell_m,
            place,
            layers,
            kernels,
        })
    }

    /// The conv-representable near operator: branch currents in,
    /// branch fluxes out (`flux = L_conv * current`), in branch order.
    /// Uncovered branches read zero. Cost: one circular convolution
    /// per (source layer, source axis, victim layer, victim axis)
    /// family with a non-empty kernel; each source plane is lumped
    /// once and reused across every victim family.
    pub fn apply(&self, currents: &[f64], fluxes: &mut [f64]) {
        assert_eq!(
            currents.len(),
            self.place.len(),
            "current vector must cover every branch"
        );
        assert_eq!(
            fluxes.len(),
            self.place.len(),
            "flux vector must cover every branch"
        );
        let n2 = self.n * self.n;
        for value in fluxes.iter_mut() {
            *value = 0.0;
        }
        for (li, layer) in self.layers.iter().enumerate() {
            for (src_axis, population) in layer.filaments.iter().enumerate() {
                if population.is_empty() {
                    continue;
                }
                // Lump this population's signed currents onto the
                // raster once, then reuse across every victim family.
                let mut field = vec![Complex64::new(0.0, 0.0); n2];
                for &(branch, _) in population {
                    let placed = self.place[branch]
                        .as_ref()
                        .expect("populated branch placed");
                    field[placed.cell] += Complex64::new(placed.sign * currents[branch], 0.0);
                }
                for (lj, victim) in self.layers.iter().enumerate() {
                    for (dst_axis, victims) in victim.filaments.iter().enumerate() {
                        let Some(kernel) = &self.kernels[li][lj][src_axis][dst_axis] else {
                            continue;
                        };
                        let mut convolved = field.clone();
                        kernel.apply_forward(&mut convolved);
                        for &(branch, _) in victims {
                            let placed = self.place[branch]
                                .as_ref()
                                .expect("populated branch placed");
                            fluxes[branch] += placed.sign * convolved[placed.cell].re;
                        }
                    }
                }
            }
        }
    }

    /// The shared raster side (test-visible).
    pub fn raster(&self) -> usize {
        self.n
    }

    /// The shared raster pitch (test-visible).
    pub fn pitch(&self) -> f64 {
        self.cell_m
    }

    /// How many filaments the conv operator covers.
    pub fn covered(&self) -> usize {
        self.place.iter().filter(|p| p.is_some()).count()
    }

    /// One layer-axis population (the inverse-kernel builder needs
    /// the same membership pass).
    pub(crate) fn population(&self, layer: usize, axis: usize) -> &[(usize, f64)] {
        &self.layers[layer].filaments[axis]
    }

    /// The layer count (the populations' outer index bound).
    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The branch count every per-branch vector this operator binds.
    pub(crate) fn branch_count(&self) -> usize {
        self.place.len()
    }

    /// One filament's raster placement.
    pub(crate) fn placement(&self, branch: usize) -> Option<&Place> {
        self.place[branch].as_ref()
    }

    /// The conv operator value the apply produces for one ordered
    /// branch pair (test introspection: the kernel sample at the
    /// centre displacement). `None` for uncovered branches, empty
    /// families, or displacements beyond the wrapped half-extent
    /// (where a single-source probe would alias).
    pub(crate) fn pair_value(&self, source: usize, victim: usize) -> Option<f64> {
        let src = self.place[source].as_ref()?;
        let dst = self.place[victim].as_ref()?;
        let kernel = self.kernels[src.layer][dst.layer][src.axis][dst.axis].as_ref()?;
        let half = self.n as i64 / 2;
        let row = dst.cell as i64 / self.n as i64 - src.cell as i64 / self.n as i64;
        let col = dst.cell as i64 % self.n as i64 - src.cell as i64 % self.n as i64;
        if row.abs() > half || col.abs() > half {
            return None;
        }
        // RAW wrapped layout: negative displacements mod n.
        let raw = |d: i64| d.rem_euclid(self.n as i64) as usize;
        let index = raw(row) * self.n + raw(col);
        // The introspected kernels are the forward (real-sample)
        // ones: the operator direction is what the tripwire probes.
        Some(src.sign * dst.sign * kernel.kernel_time[index].re)
    }
}

/// `M^T . x` in C64: the first `loops` mesh rows' residual lifted to
/// branch-space flux-like values. Used by the v2 precond's Richardson
/// sweep (`ConvState` wiring in `iterative.rs`); kept beside the
/// kernels so the conv engine's data flow reads top to bottom.
pub(crate) fn mesh_scatter(
    mesh: &crate::mesh::MeshMatrix,
    loops: usize,
    x: &[Complex64],
) -> Vec<Complex64> {
    let mut branch = vec![Complex64::new(0.0, 0.0); mesh.branch_count()];
    for (i, &xi) in x.iter().enumerate().take(loops) {
        for &(b, sign) in mesh.row(i) {
            branch[b] += xi * sign;
        }
    }
    branch
}

/// `M . v` in C64 over the first `loops` mesh rows.
pub(crate) fn mesh_gather(
    mesh: &crate::mesh::MeshMatrix,
    loops: usize,
    v: &[Complex64],
) -> Vec<Complex64> {
    (0..loops)
        .map(|i| {
            mesh.row(i)
                .iter()
                .fold(Complex64::new(0.0, 0.0), |sum, &(b, sign)| {
                    sum + v[b] * sign
                })
        })
        .collect()
}

/// The population's representative segment: median
/// length/width/height/sigma, median z, +axis direction, explicit
/// width direction so the cross-section orientation is deterministic.
fn representative_segment(
    filaments: &[Filament],
    population: &[(usize, f64)],
    axis: usize,
) -> Segment {
    let mut lengths = Vec::with_capacity(population.len());
    let mut widths = Vec::with_capacity(population.len());
    let mut heights = Vec::with_capacity(population.len());
    let mut zs = Vec::with_capacity(population.len());
    let mut sigmas = Vec::with_capacity(population.len());
    for &(branch, _) in population {
        let f = &filaments[branch];
        lengths.push(f.length());
        widths.push(f.width());
        heights.push(f.height());
        zs.push(f.center().z);
        sigmas.push(f.sigma());
    }
    let half = median(&mut lengths) * 0.5;
    let z = median(&mut zs);
    let (a, b, width_dir) = if axis == 0 {
        ([-half, 0.0, z], [half, 0.0, z], [0.0, 1.0, 0.0])
    } else {
        ([0.0, -half, z], [0.0, half, z], [1.0, 0.0, 0.0])
    };
    Segment::new(
        Node::from(a),
        Node::from(b),
        median(&mut widths),
        median(&mut heights),
        median(&mut sigmas),
    )
    .with_width_dir(width_dir)
}

/// Samples one orientation-family kernel: two representative
/// segments (the median geometry of each population, at its own
/// layer's z, +axis direction), the source at the origin, the victim
/// displaced over the raster; the exact mutual inductance at every
/// displacement cell. The same-layer zero displacement is the
/// representative's self inductance (the mutual path rejects
/// coincident pairs).
fn sample_family_kernel(
    filaments: &[Filament],
    source_population: &[(usize, f64)],
    victim_population: &[(usize, f64)],
    src_axis: usize,
    dst_axis: usize,
    same_layer: bool,
    n: usize,
    cell_m: f64,
) -> Result<ConvKernel, KernelError> {
    let src_segment = representative_segment(filaments, source_population, src_axis);
    let dst_segment = representative_segment(filaments, victim_population, dst_axis);
    let src = Filament::new(&src_segment).expect("representative segment is well-formed");
    let dst_centre = dst_segment.center();
    let (dst_a, dst_b) = (dst_segment.a.position(), dst_segment.b.position());
    let (width, height, sigma, width_dir) = (
        dst_segment.width,
        dst_segment.height,
        dst_segment.sigma,
        dst_segment.width_dir,
    );
    // The damping floor is dead weight here: these kernels are read
    // only through `apply_forward` (the operator direction); the
    // inverse spectrum and its damping knob arrive with the precond
    // wiring. 0.01 is the nominal value the wiring will expose.
    Ok(ConvKernel::build(
        n,
        cell_m,
        |dx, dy| {
            if same_layer && src_axis == dst_axis && dx == 0.0 && dy == 0.0 {
                return inductance::self_inductance(&src);
            }
            let offset = Vector3::new(dx, dy, 0.0) - dst_centre;
            let displaced = Segment::new(
                Node::from([dst_a.x + offset.x, dst_a.y + offset.y, dst_a.z]),
                Node::from([dst_b.x + offset.x, dst_b.y + offset.y, dst_b.z]),
                width,
                height,
                sigma,
            )
            .with_width_dir(width_dir.expect("representative carries an explicit width direction"));
            let victim = Filament::new(&displaced).expect("displaced segment is well-formed");
            inductance::mutual_inductance(&src, &victim)
                .expect("kernel-sample pair has aligned cross-sections by construction")
        },
        0.01,
    ))
}

#[cfg(test)]
mod layer_pair_tests {
    use super::*;

    const PITCH: f64 = 1e-3;
    const LENGTH: f64 = 4e-3;
    const W: f64 = 0.2e-3;
    const H: f64 = 0.1e-3;
    const SIGMA: f64 = 5.8e7;

    fn bar(a: [f64; 3], b: [f64; 3], width_dir: [f64; 3]) -> Filament {
        let segment =
            Segment::new(Node::from(a), Node::from(b), W, H, SIGMA).with_width_dir(width_dir);
        Filament::new(&segment).expect("test bar is well-formed")
    }

    /// A uniform two-layer deck: five x-directed bars on z = 0
    /// (centres on the pitch grid), one reversed bar sharing a cell,
    /// five y-directed bars on z = 1 mm, one via barrel. Every
    /// centre lands on a raster cell, so the pitch-matched raster
    /// holds one filament per cell per axis and the conv operator
    /// value must equal the exact mutual to float noise.
    fn deck() -> Vec<Filament> {
        let mut filaments = Vec::new();
        for i in 0..5 {
            let y = i as f64 * PITCH;
            filaments.push(bar([0.0, y, 0.0], [LENGTH, y, 0.0], [0.0, 1.0, 0.0]));
        }
        // Branch 5: antiparallel, sharing branch 2's cell.
        filaments.push(bar(
            [LENGTH, 2.0 * PITCH, 0.0],
            [0.0, 2.0 * PITCH, 0.0],
            [0.0, 1.0, 0.0],
        ));
        for j in 0..5 {
            let x = j as f64 * PITCH;
            filaments.push(bar([x, 0.0, 1.0e-3], [x, LENGTH, 1.0e-3], [1.0, 0.0, 0.0]));
        }
        // Branch 11: a via barrel - uncovered, own z cluster.
        filaments.push(bar(
            [2.0 * PITCH, 2.0 * PITCH, 0.0],
            [2.0 * PITCH, 2.0 * PITCH, 1.0e-3],
            [1.0, 0.0, 0.0],
        ));
        filaments
    }

    #[test]
    fn tab_blocks_carry_exact_mutuals_and_solve_back() {
        // The census-patch tab blocks (2c) reproduce the exact
        // mutuals among the patch members (the via plus both landing
        // pours of the deck's via cell), and the factored block
        // solves back: (R + jw L) (A^-1 v) = v at float noise. Pins
        // the explicit composition's entries and the smoother stage.
        let deck = deck();
        let resistances = vec![1.0e-6; deck.len()];
        let omega = 2.0 * std::f64::consts::PI * 1.0e3;
        let blocks = ConvPrecond::build_tabs(
            &deck,
            &resistances,
            omega,
            TabComposition::Both,
            PITCH,
            true,
        )
        .expect("uniform deck builds tab blocks");
        assert_eq!(blocks.len(), 1, "one patch: the deck's via cell");
        let block = &blocks[0];
        assert_eq!(
            block.members,
            // Branch order: the via, the x-pour of its cell, the
            // antiparallel x-pour sharing it, the y-pour crossing.
            vec![11usize, 2, 5, 8],
            "via first (branch order), then the landing pours"
        );
        let count = 4usize;
        let at = |row: usize, col: usize| block.l[row * count + col];
        let test_err = |delta: f64| (delta.abs() < 1.0e-15);
        assert!(test_err(at(0, 0) - inductance::self_inductance(&deck[11])));
        assert!(test_err(at(1, 1) - inductance::self_inductance(&deck[2])));
        // The antiparallel twin sits on branch 2's exact line:
        // mutual is the negated self L.
        let twin = inductance::mutual_inductance(&deck[2], &deck[5])
            .expect("deck pair admits the exact kernel");
        assert!(test_err(twin + inductance::self_inductance(&deck[2])));
        assert!(test_err(at(1, 2) - twin));
        // The via crosses every Manhattan pour: exact zero mutual.
        assert!(test_err(
            at(0, 1)
                - inductance::mutual_inductance(&deck[11], &deck[2])
                    .expect("deck pair admits the exact kernel")
        ));
        assert_eq!(at(0, 1), 0.0);
        // The x/y pour pair in the cell crosses: exact zero.
        let pour_crossing = inductance::mutual_inductance(&deck[2], &deck[8])
            .expect("deck pair admits the exact kernel");
        assert_eq!(pour_crossing, 0.0);
        assert!(test_err(at(1, 3) - pour_crossing));
        assert_eq!(at(3, 1), pour_crossing, "the L block is symmetric");
        assert!(
            block.lu.is_some(),
            "the Both composition factors the smoother block"
        );
        // Roundtrip the exact local inverse through the block.
        let residual: Vec<C64> = vec![C64::new(1.0, -2.0); deck.len()];
        let mut correction = vec![C64::new(0.0, 0.0); deck.len()];
        block.solve(&residual, &mut correction);
        for row in 0..count {
            let mut sum = C64::new(resistances[block.members[row]], omega * at(row, row))
                * correction[block.members[row]];
            for col in 0..count {
                if col == row {
                    continue;
                }
                sum += C64::new(0.0, omega * at(row, col)) * correction[block.members[col]];
            }
            let back = sum - residual[block.members[row]];
            assert!(
                back.norm() < 1e-12 * residual[block.members[row]].norm(),
                "LU solve roundtrip failed: {back} at member {}",
                block.members[row]
            );
        }
    }

    #[test]
    fn layer_pair_operator_matches_exact_mutuals_on_a_uniform_deck() {
        let filaments = deck();
        let operator = LayerPairOperator::build(&filaments).expect("uniform deck builds");
        assert_eq!(
            operator.pitch(),
            PITCH,
            "modal pitch must recover the grid pitch"
        );
        assert_eq!(operator.covered(), 11, "the via barrel stays uncovered");

        // (source, victim) probes across every family and the sign
        // path; the diagonal probes pin the self-inductance sample.
        let probes: &[(usize, usize)] = &[
            (0, 2),
            (0, 4),
            (2, 0),
            (5, 0),
            (0, 5),
            (0, 6),
            (0, 8),
            (4, 6),
            (2, 10),
            (6, 0),
            (8, 2),
            (6, 8),
            (6, 10),
            (0, 0),
            (6, 6),
        ];
        for &(source, victim) in probes {
            let expected = if source == victim {
                inductance::self_inductance(&filaments[source])
            } else {
                inductance::mutual_inductance(&filaments[source], &filaments[victim])
                    .expect("probe pair has aligned cross-sections")
            };
            let got = operator
                .pair_value(source, victim)
                .unwrap_or_else(|| panic!("pair ({source},{victim}) aliased"));
            let scale = expected.abs().max(1e-12);
            assert!(
                (got - expected).abs() / scale < 1e-9,
                "pair ({source},{victim}): conv {got} vs exact {expected}"
            );
        }

        // The apply reproduces the same values from a single-source
        // probe, and the uncovered via reads zero.
        let mut currents = vec![0.0; filaments.len()];
        let mut fluxes = vec![0.0; filaments.len()];
        currents[0] = 1.0;
        operator.apply(&currents, &mut fluxes);
        for &(source, victim) in &[(0usize, 2usize), (0, 4), (0, 5), (0, 6), (0, 8)] {
            let expected = inductance::mutual_inductance(&filaments[source], &filaments[victim])
                .expect("probe pair has aligned cross-sections");
            assert!(
                (fluxes[victim] - expected).abs() / expected.abs().max(1e-12) < 1e-9,
                "apply pair ({source},{victim}): got {} vs exact {expected}",
                fluxes[victim]
            );
        }
        assert_eq!(fluxes[11], 0.0, "the uncovered via must read zero flux");
    }

    /// Same-layer perpendicular strokes sharing a cell: the zero
    /// displacement of the family is a CROSSING pair - exact zero
    /// mutual (Neumann's dot product vanishes pointwise) - never the
    /// self inductance. Pins the 2026-10-20 diagonal fix.
    #[test]
    fn same_layer_perpendicular_family_carries_no_diagonal_self() {
        let mut filaments = vec![
            bar([0.0, 0.0, 0.0], [LENGTH, 0.0, 0.0], [0.0, 1.0, 0.0]),
            bar(
                [2.0 * PITCH, -LENGTH / 2.0, 0.0],
                [2.0 * PITCH, LENGTH / 2.0, 0.0],
                [1.0, 0.0, 0.0],
            ),
        ];
        // A spectator bar off to the side: the coordinate cloud needs a
        // gap in some projection for the modal pitch (the crossing pair
        // alone is a single point).
        filaments.push(bar(
            [0.0, LENGTH, 0.0],
            [LENGTH, LENGTH, 0.0],
            [0.0, 1.0, 0.0],
        ));
        let operator = LayerPairOperator::build(&filaments).expect("three-bar deck builds");
        let got = operator.pair_value(0, 1).expect("crossing pair resolves");
        assert_eq!(
            got, 0.0,
            "perpendicular pair at zero displacement must read exact zero, not self L"
        );
        let got = operator
            .pair_value(1, 0)
            .expect("reverse crossing pair resolves");
        assert_eq!(got, 0.0, "reverse direction of the same crossing pair");
    }
}

// ---------------------------------------------------------------------------
// The v2 conv preconditioner core (slice 2b increment 2).
//
// The precond apply runs a fixed-count Richardson inner solve of the
// matrix-free near model `A_near = M (R + j omega L_conv) M^T`:
// - the matvec's L part is the layer-pair operator (per-family
//   forward convolutions; in-layer AND cross-layer pour couplings),
//   R rides the branch diagonal exactly, j omega the scaling;
// - the smoother is the per-family Tikhonov inverse
//   `conj(H) / (|H|^2 + eps)` over the complex kernel
//   `r_agg * delta(d) + j omega * M(d)` - bounded gain at every
//   frequency, so the inner solve cannot poison the outer GMRES
//   (failmode-2 guard);
// - the via-tab blocks and the branch-CSR spine are RESERVED slots,
//   empty in this increment (booked as 2c/2d): couplings they would
//   carry are simply absent from the model - the answer stays
//   operator-exact (GMRES referee), only precond quality is dimmed.
// ---------------------------------------------------------------------------

/// One (layer pair, axis family) population pair of the smoother.
pub struct ConvPrecond {
    operator: LayerPairOperator,
    /// `inverses[li][lj][src_axis][dst_axis]`: the damped spectral
    /// inverse of the family's complex kernel; `None` where either
    /// population is empty.
    inverses: Vec<Vec<[[Option<ConvKernel>; 2]; 2]>>,
    /// The explicit census-patch tab blocks (v2 slice 2c); empty
    /// under the `Off` composition.
    tabs: Vec<TabBlock>,
}

impl ConvPrecond {
    /// Builds the model: the layer-pair operator (omega-independent)
    /// plus this frequency's per-family inverse kernels. Per-family
    /// diagonal resistance rides the same-layer kernel's zero
    /// displacement as the population's mean branch resistance.
    ///
    /// Tab chassis (2c): `composition` selects the explicit
    /// per-census-patch blocks (`Both` default through the env
    /// knob); `tab_bin_m` bins the via census (the engine reads
    /// `FASTERHENRY_CENSUS_VIA_BIN_M` or falls back to
    /// `derived_radius`); `tab_pour` adds each patch's landing pours
    /// (`FASTERHENRY_CONV_TAB_POUR=1` default).
    pub fn build(
        filaments: &[Filament],
        resistances: &[f64],
        omega: f64,
        damp: f64,
        composition: TabComposition,
        tab_bin_m: f64,
        tab_pour: bool,
    ) -> Result<Self, KernelError> {
        assert_eq!(
            filaments.len(),
            resistances.len(),
            "one resistance per branch"
        );
        let operator = LayerPairOperator::build(filaments)?;
        let layer_count = operator.layer_count();
        let mut inverses: Vec<Vec<[[Option<ConvKernel>; 2]; 2]>> = Vec::with_capacity(layer_count);
        for li in 0..layer_count {
            let mut row: Vec<[[Option<ConvKernel>; 2]; 2]> = Vec::with_capacity(layer_count);
            for lj in 0..layer_count {
                let mut families: [[Option<ConvKernel>; 2]; 2] = Default::default();
                for src_axis in 0..2 {
                    let src_population = operator.population(li, src_axis);
                    if src_population.is_empty() {
                        continue;
                    }
                    for dst_axis in 0..2 {
                        let dst_population = operator.population(lj, dst_axis);
                        if dst_population.is_empty() {
                            continue;
                        }
                        let kernel = sample_inverse_kernel(
                            filaments,
                            resistances,
                            src_population,
                            dst_population,
                            src_axis,
                            dst_axis,
                            li == lj,
                            omega,
                            operator.raster(),
                            operator.pitch(),
                            damp,
                        )?;
                        families[src_axis][dst_axis] = Some(kernel);
                    }
                }
                row.push(families);
            }
            inverses.push(row);
        }
        let tabs = Self::build_tabs(
            filaments,
            resistances,
            omega,
            composition,
            tab_bin_m,
            tab_pour,
        )?;
        Ok(Self {
            operator,
            inverses,
            tabs,
        })
    }

    /// The explicit per-patch blocks: member branches from the
    /// census's flood fill, exact dense mutuals among them, LU
    /// factored when the smoother composition is live.
    fn build_tabs(
        filaments: &[Filament],
        resistances: &[f64],
        omega: f64,
        composition: TabComposition,
        tab_bin_m: f64,
        tab_pour: bool,
    ) -> Result<Vec<TabBlock>, KernelError> {
        if composition == TabComposition::Off {
            return Ok(Vec::new());
        }
        let membership = crate::census::via_patch_membership(filaments, tab_bin_m, tab_pour);
        let smoother = composition == TabComposition::Both;
        let mut blocks = Vec::new();
        for members in membership.members {
            let count = members.len();
            if count == 0 {
                continue;
            }
            let mut l = vec![0.0f64; count * count];
            for row in 0..count {
                let l_row = inductance::self_inductance(&filaments[members[row]]);
                for col in row + 1..count {
                    let mutual = inductance::mutual_inductance(
                        &filaments[members[row]],
                        &filaments[members[col]],
                    )?;
                    l[row * count + col] = mutual;
                    l[col * count + row] = mutual;
                }
                l[row * (count + 1)] = l_row;
            }
            blocks.push(TabBlock::build(members, l, resistances, omega, smoother));
        }
        Ok(blocks)
    }

    /// The near-model branch drops: `R * current + j omega * L_conv *
    /// current`, per branch (the caller lifts through the mesh).
    pub fn near_drops(
        &self,
        currents: &[Complex64],
        resistances: &[f64],
        omega: f64,
    ) -> Vec<Complex64> {
        let count = currents.len();
        let re: Vec<f64> = currents.iter().map(|c| c.re).collect();
        let im: Vec<f64> = currents.iter().map(|c| c.im).collect();
        let mut flux_re = vec![0.0; count];
        let mut flux_im = vec![0.0; count];
        self.operator.apply(&re, &mut flux_re);
        self.operator.apply(&im, &mut flux_im);
        // The explicit tab blocks: exact dense flux over member
        // branches; the unconv-representable couplings (vias, tabs)
        // live here, never in the kernels.
        for tab in &self.tabs {
            tab.flux(currents, &mut flux_re, &mut flux_im);
        }
        (0..count)
            .map(|b| {
                let r = resistances[b];
                Complex64::new(
                    r * re[b] - omega * flux_im[b],
                    r * im[b] + omega * flux_re[b],
                )
            })
            .collect()
    }

    /// The smoother: branch correction = un-raster of the per-family
    /// Tikhonov inverses applied to the rastered branch residual. The
    /// residual vector is flux-like (loop residual lifted by the
    /// mesh transpose at the caller).
    pub fn smooth(&self, residual: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(
            residual.len(),
            self.operator.branch_count(),
            "residual per branch"
        );
        let n = self.operator.raster();
        let n2 = n * n;
        let mut correction = vec![Complex64::new(0.0, 0.0); residual.len()];
        for lj in 0..self.operator.layer_count() {
            for dst_axis in 0..2 {
                let dst_population = self.operator.population(lj, dst_axis);
                if dst_population.is_empty() {
                    continue;
                }
                // Raster this destination population's signed
                // residual once, then reuse across source families.
                let mut field = vec![Complex64::new(0.0, 0.0); n2];
                for &(branch, _) in dst_population {
                    let placed = self
                        .operator
                        .placement(branch)
                        .expect("populated branch placed");
                    field[placed.cell] += Complex64::new(
                        placed.sign * residual[branch].re,
                        placed.sign * residual[branch].im,
                    );
                }
                for li in 0..self.operator.layer_count() {
                    for src_axis in 0..2 {
                        let Some(kernel) = &self.inverses[li][lj][src_axis][dst_axis] else {
                            continue;
                        };
                        let src_population = self.operator.population(li, src_axis);
                        let mut worked = field.clone();
                        kernel.apply(&mut worked);
                        for &(branch, _) in src_population {
                            let placed = self
                                .operator
                                .placement(branch)
                                .expect("populated branch placed");
                            correction[branch] += Complex64::new(
                                placed.sign * worked[placed.cell].re,
                                placed.sign * worked[placed.cell].im,
                            );
                        }
                    }
                }
            }
        }
        // The exact per-patch stage: inverse-block solves over each
        // patch's member residual, additive with the conv families'
        // corrections (the `Both` composition; a no-op when the LU
        // is absent).
        for tab in &self.tabs {
            tab.solve(residual, &mut correction);
        }
        correction
    }

    /// Test-facing: the raster side.
    pub fn raster(&self) -> usize {
        self.operator.raster()
    }
}

/// C64 alias for the tab-block internals (f64-real complex, the
/// runtime arithmetic of the conv core).
type C64 = Complex64;

/// How the explicit per-patch tab blocks (v2 slice 2c) compose into
/// the precond: `Both` feeds the near-model matvec AND an exact
/// per-patch solve stage in the smoother (default); `Matvec` keeps
/// the smoother conv-only (quality unchanged, machinery exercisable);
/// `Off` reproduces the increment-2 model exactly. Env
/// `FASTERHENRY_CONV_TABS=both|matvec|off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TabComposition {
    #[default]
    Both,
    Matvec,
    Off,
}

/// One census-patch explicit tab block: the member branches' exact
/// dense mutual block (the fork's `mutual_inductance`, never a
/// kernel approximation), living in branch space and composed into
/// the near-model matvec as flux plus, under the `Both` composition,
/// an exact `R + j omega L` LU solve as the smoother's per-patch
/// stage.
pub struct TabBlock {
    /// The member branch indices (the block's row/col order).
    members: Vec<usize>,
    /// Row-major member-order L (real, RL-only model).
    l: Vec<f64>,
    /// Row-major in-place LU of `diag(R_members) + j omega L` plus the
    /// pivot permutation, when the smoother composition is live
    /// (`None` under `Matvec`).
    lu: Option<(Vec<C64>, Vec<usize>)>,
}

impl TabBlock {
    /// Blocks are built once per frequency: the LU carries `omega`,
    /// the exact mutuals do not.
    fn build(
        members: Vec<usize>,
        l: Vec<f64>,
        resistances: &[f64],
        omega: f64,
        smoother: bool,
    ) -> Self {
        let count = members.len();
        assert_eq!(
            l.len(),
            count * count,
            "tab block: one row-major L per member pair"
        );
        let lu = if smoother {
            let mut factored: Vec<C64> = l
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    let (row, col) = (index / count, index % count);
                    C64::new(
                        if row == col {
                            resistances[members[row]]
                        } else {
                            0.0
                        },
                        omega * value,
                    )
                })
                .collect();
            let mut pivots: Vec<usize> = (0..count).collect();
            for pivot in 0..count {
                // Partial-pivot selection scans the PIVOT COLUMN: the
                // pivot value is the (row, pivot) entry, never the
                // diagonal (row, col=pivot search collapses to the
                // diagonal only at pivot time 0).
                let mut best = pivot;
                let mut best_norm = factored[pivot * count + pivot].norm();
                for row in pivot + 1..count {
                    let norm = factored[row * count + pivot].norm();
                    if norm > best_norm {
                        best = row;
                        best_norm = norm;
                    }
                }
                pivots.swap(pivot, best);
                for col in 0..count {
                    factored.swap(pivot * count + col, best * count + col);
                }
                let piv = factored[pivot * (count + 1)];
                let pivot_norm = piv.norm();
                assert!(
                    pivot_norm > 0.0,
                    "tab block: singular smooth matrix (zero resistance? zero mutual?); pivot_norm {pivot_norm}, count {count}, diag {r:?}",
                    r = (0..count)
                        .map(|i| factored[i * (count + 1)])
                        .collect::<Vec<_>>()
                );
                for row in pivot + 1..count {
                    // In-place Doolittle: the L factor lives at
                    // (row, pivot), not on the diagonal.
                    let factor = factored[row * count + pivot] / piv;
                    factored[row * count + pivot] = factor;
                    for col in pivot + 1..count {
                        let contribution = factor * factored[pivot * count + col];
                        factored[row * count + col] -= contribution;
                    }
                }
            }
            Some((factored, pivots))
        } else {
            None
        };
        Self { members, l, lu }
    }

    /// The block's flux contribution: `flux[i] += L[i][j] *
    /// current[j]` over member pairs (real L, so the caller's real /
    /// imaginary flux accumulators both take it).
    fn flux(&self, currents: &[C64], flux_re: &mut [f64], flux_im: &mut [f64]) {
        let count = self.members.len();
        for (i, &dst) in self.members.iter().enumerate() {
            let row = &self.l[i * count..(i + 1) * count];
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for (j, &src) in self.members.iter().enumerate() {
                sum_re += row[j] * currents[src].re;
                sum_im += row[j] * currents[src].im;
            }
            flux_re[dst] += sum_re;
            flux_im[dst] += sum_im;
        }
    }

    /// The exact per-patch smoother stage:
    /// `correction[members] += (diag(R) + j omega L)^-1 residual[members]`.
    fn solve(&self, residual: &[C64], correction: &mut [C64]) {
        let Some((factored, pivots)) = &self.lu else {
            return;
        };
        let count = self.members.len();
        let mut z: Vec<C64> = pivots
            .iter()
            .map(|&row| residual[self.members[row]])
            .collect();
        // Forward substitution (unit diagonal L).
        for row in 0..count {
            for col in 0..row {
                let contribution = factored[row * count + col] * z[col];
                z[row] -= contribution;
            }
        }
        // Back substitution.
        for row in (0..count).rev() {
            let mut value = z[row];
            for col in row + 1..count {
                value -= factored[row * count + col] * z[col];
            }
            z[row] = value / factored[row * (count + 1)];
        }
        for (slot, &member) in self.members.iter().enumerate() {
            correction[member] += z[slot];
        }
    }
}

/// Samples one family's INVERSE kernel: the complex kernel
/// `r_agg * delta(d) + j omega * M(d)` on the median-geometry
/// representatives, damped-spectral-inverted. The diagonal
/// resistance rides the same-layer zero displacement as the source
/// population's mean branch resistance.
#[allow(clippy::too_many_arguments)]
fn sample_inverse_kernel(
    filaments: &[Filament],
    resistances: &[f64],
    source_population: &[(usize, f64)],
    victim_population: &[(usize, f64)],
    src_axis: usize,
    dst_axis: usize,
    same_layer: bool,
    omega: f64,
    n: usize,
    cell_m: f64,
    damp: f64,
) -> Result<ConvKernel, KernelError> {
    let r_agg = source_population
        .iter()
        .map(|&(b, _)| resistances[b])
        .sum::<f64>()
        / source_population.len() as f64;
    let src_segment = representative_segment(filaments, source_population, src_axis);
    let dst_segment = representative_segment(filaments, victim_population, dst_axis);
    let src = Filament::new(&src_segment).expect("representative segment is well-formed");
    let dst_centre = dst_segment.center();
    let (dst_a, dst_b) = (dst_segment.a.position(), dst_segment.b.position());
    let (width, height, sigma, width_dir) = (
        dst_segment.width,
        dst_segment.height,
        dst_segment.sigma,
        dst_segment.width_dir,
    );
    Ok(ConvKernel::build_complex(
        n,
        cell_m,
        |dx, dy| {
            if same_layer && src_axis == dst_axis && dx == 0.0 && dy == 0.0 {
                return Complex64::new(r_agg, omega * inductance::self_inductance(&src));
            }
            let offset = Vector3::new(dx, dy, 0.0) - dst_centre;
            let displaced = Segment::new(
                Node::from([dst_a.x + offset.x, dst_a.y + offset.y, dst_a.z]),
                Node::from([dst_b.x + offset.x, dst_b.y + offset.y, dst_b.z]),
                width,
                height,
                sigma,
            )
            .with_width_dir(width_dir.expect("representative carries an explicit width direction"));
            let victim = Filament::new(&displaced).expect("displaced segment is well-formed");
            let mutual = inductance::mutual_inductance(&src, &victim)
                .expect("kernel-sample pair has aligned cross-sections by construction");
            Complex64::new(0.0, omega * mutual)
        },
        damp,
    ))
}
