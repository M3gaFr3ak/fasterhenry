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
        let cell_m = match std::env::var("FASTERHENRY_CONV_RASTER_PITCH") {
            Ok(text) => match text.trim().parse::<f64>() {
                Ok(value) if value.is_finite() && value > 0.0 => value,
                _ => {
                    eprintln!(
                        "conv: warning: FASTERHENRY_CONV_RASTER_PITCH={text:?} is not a positive finite number of meters - falling through to the modal-pitch estimate (a raster pitch is precond cost, never the answer)",
                    );
                    modal_pitch(&points)
                }
            },
            Err(_) => modal_pitch(&points),
        };

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
            "layer-pair raster side {side} exceeds the {MAX_SIDE} cap - set FASTERHENRY_CONV_RASTER_PITCH (meters) to coarsen the cell",
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

    pub(crate) const PITCH: f64 = 1e-3;
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
    pub(crate) fn deck() -> Vec<Filament> {
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
            None,
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
    fn spine_excludes_the_blocking_landing_pours() {
        // The increment-3 deck's single patch carries its landing
        // pours ([11, 2, 5, 8] members): those near partners must ride
        // the BLOCK exactly once and stay SPINE-ABSENT, while the
        // bin-adjacent pours outside the patch's cells ride the spine
        // (conv-inc4 audit F1: with via-only patch ids the block
        // members slipped into the spine as well and were counted
        // twice).
        let deck = deck();
        let resistances = vec![1.0e-6; deck.len()];
        let omega = std::f64::consts::TAU * 1.0e3;
        let spine = ConvPrecond::build_spine(
            &deck,
            &resistances,
            omega,
            TabComposition::Both,
            PITCH,
            true,
            None,
        )
        .expect("deck builds a spine");
        let spine = spine.expect("a deck with a via carries a spine");
        let lo = spine.row_ptr[0] as usize;
        let hi = spine.row_ptr[1] as usize;
        // The via's spine row must EXCLUDE the block members (2, 5,
        // 8 cross or share its cells) and carry the bin-adjacent
        // pours whose centres sit OUTSIDE the patch's cells: 1 (the
        // y-apart x-pour), 3, 7 and 9 (the 1 mm-away y-pours).
        assert_eq!(
            spine.col[lo..hi].to_vec(),
            vec![1u32, 3, 7, 9],
            "the via's spine row: only the block-free partners"
        );
        for (index, &partner) in spine.col[lo..hi].iter().enumerate() {
            let expected = inductance::mutual_inductance(&deck[11], &deck[partner as usize])
                .expect("spine pair admits the exact kernel");
            assert!(
                (spine.values[lo + index] - expected).abs() < 1.0e-15,
                "spine entry (11,{partner}) must be the exact mutual"
            );
        }
    }

    #[test]
    fn spine_carries_block_free_via_pairs_only() {
        // Two separate patches: vias at the (0,0) and (3,0) census
        // cells (4-connected flood fill keeps them apart at a 1 mm
        // bin), pours scattered around them. The spine's slot rows
        // must carry only the block-free pairs - via-to-pour partners
        // whose centre cell does not belong to the via's own patch;
        // same-patch pairs live in the block, spills stay conv-borne.
        let mut filaments = Vec::new();
        for i in 0..6 {
            filaments.push(bar(
                [0.0, i as f64 * PITCH, 0.0],
                [LENGTH, i as f64 * PITCH, 0.0],
                [0.0, 1.0, 0.0],
            ));
        }
        // Branch 6: a y-pour whose line passes through via A's cell
        // but whose centre is centred at (0.5, 2.5) mm.
        filaments.push(bar(
            [0.5 * PITCH, 0.5 * PITCH, 0.0],
            [0.5 * PITCH, LENGTH, 0.0],
            [1.0, 0.0, 0.0],
        ));
        // Branch 7: an x-pour passing below via B too, centre
        // (2.0, 0.5) mm.
        filaments.push(bar(
            [0.0, 0.5 * PITCH, 0.0],
            [LENGTH, 0.5 * PITCH, 0.0],
            [0.0, 1.0, 0.0],
        ));
        // Branches 8 and 9: the two vias.
        filaments.push(bar(
            [0.5 * PITCH, 0.5 * PITCH, 0.0],
            [0.5 * PITCH, 0.5 * PITCH, 1.5e-3],
            [1.0, 0.0, 0.0],
        ));
        filaments.push(bar(
            [3.5 * PITCH, 0.5 * PITCH, 0.0],
            [3.5 * PITCH, 0.5 * PITCH, 1.5e-3],
            [1.0, 0.0, 0.0],
        ));
        let resistances = vec![1.0e-6; filaments.len()];
        let omega = std::f64::consts::TAU * 1.0e3;
        let spine = ConvPrecond::build_spine(
            &filaments,
            &resistances,
            omega,
            TabComposition::Both,
            PITCH,
            true,
            None,
        )
        .expect("two-patch deck builds a spine");
        let spine = spine.expect("a deck with vias carries a spine");
        assert_eq!(spine.vias, vec![8usize, 9]);
        assert_eq!(
            spine.jacobi.len(),
            2,
            "the Both composition carries the Jacobi denominators"
        );
        let partners = |slot: usize| {
            let lo = spine.row_ptr[slot] as usize;
            let hi = spine.row_ptr[slot + 1] as usize;
            spine.col[lo..hi].to_vec()
        };
        // Slot 0 (via A): the pours of its own patch's cells are
        // blocked... none - br 6 and 7 have centres OFF the patch's
        // cells, br 0/1's are at (2,0)/(2,1): all four partners.
        assert_eq!(partners(0), vec![0u32, 1, 6, 7]);
        // Slot 1 (via B): 0/1 (dx zero within the pour line), 7
        // (passes through); not 6 (3 mm away) and not a via-via pair
        // (3 mm exceeds the bin radius).
        assert_eq!(partners(1), vec![0u32, 1, 7]);
        for (slot, &via) in spine.vias.iter().enumerate() {
            let lo = spine.row_ptr[slot] as usize;
            let hi = spine.row_ptr[slot + 1] as usize;
            for (index, &partner) in spine.col[lo..hi].iter().enumerate() {
                let expected =
                    inductance::mutual_inductance(&filaments[via], &filaments[partner as usize])
                        .expect("spine pair admits the exact kernel");
                assert!(
                    (spine.values[lo + index] - expected).abs() < 1.0e-15,
                    "spine entry ({via},{partner}) must be the exact mutual"
                );
            }
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
// - the explicit tab blocks AND the cross spine both COMPOSED here:
//   tabs since increment 3 (fork 0e27478, census-patch membership +
//   exact LU blocks; env FASTERHENRY_CONV_TABS), the spine since
//   increment 4 (fork 0e53ff0, block-free via pairs + damped Jacobi;
//   env FASTERHENRY_CONV_SPINE) - both composition-selectable with
//   BOTH default; no remaining reserved slots in the near model.
//   **[corrected 2026-10-20, conv-inc4 audit F2 - this header's
//   spine half was left stale by the very commit that shipped the
//   spine (the inc3 audit's F1 pattern repeating)]**
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The conv-matvec operator core (v2 slice 2b increment 5). The ENGINE's
// equation is no longer the pair-list precorrect: the near band is owned
// by a displacement-domain DIFFERENCE model - see the increment-5
// decision in the peec books (answer exactness = discretization + pFFT
// far knobs + the conv near model). Header honesty: this composition is
// `conv-matvec`, not `precorrect-FFT - near-correct`.
// ---------------------------------------------------------------------------

/// A displacement-domain DIFFERENCE kernel: `Kdiff(u) =
/// L_exact(synthetic median pair at u) - L_grid(same pair)` over the
/// near band, zero elsewhere, FFT-prepared once for forward circular
/// convolution. The displacement convention matches [`ConvKernel`]'s
/// raw wrapped layout (plane index 0 = displacement 0, negatives
/// wrapping to the top rows/columns), so the same raster geometry
/// hosts source cells and victim reads.
pub struct DiffKernel {
    /// Raster side.
    n: usize,
    /// Raster cell pitch (metres).
    cell_m: f64,
    /// The difference table's half-width, in cells. Zero outside.
    band_cells: usize,
    /// `FFT2(Kdiff)` in the raw wrapped layout; the apply's multiply.
    kernel_hat: Vec<Complex64>,
    /// The REAL signed-displacement table `[2r+1]^2` (the substitution
    /// stages read it back per pair). Empty when `r == 0`.
    time_window: Vec<f64>,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
}

impl DiffKernel {
    /// Builds the difference kernel. `sample(dx, dy)` receives SIGNED
    /// displacement cells (`|dx|, |dy| <= band_cells`); it is called
    /// exactly that many times, never outside the band.
    pub fn build_diff(
        n: usize,
        cell_m: f64,
        band_cells: usize,
        sample: impl Fn(i64, i64) -> f64,
    ) -> Self {
        assert!(n >= 3, "diff kernel raster too small: {n}");
        assert!(
            cell_m.is_finite() && cell_m > 0.0,
            "diff kernel pitch must be a positive metre value: {cell_m}"
        );
        let reach = band_cells as i64;
        let dims = 2 * band_cells + 1;
        let mut time_window = vec![0.0; dims * dims];
        for (index, value) in time_window.iter_mut().enumerate() {
            let dx = index as i64 / dims as i64 - reach;
            let dy = index as i64 % dims as i64 - reach;
            *value = sample(dx, dy);
        }
        // The plane: window entries at their wrapped positions, zero
        // elsewhere (never sampled beyond the band).
        let wrap = |index: i64| -> i64 {
            let half = n as i64 / 2;
            if index > half {
                index - n as i64
            } else {
                index
            }
        };
        let mut plane = vec![Complex64::new(0.0, 0.0); n * n];
        for row in 0..n as i64 {
            for col in 0..n as i64 {
                let (dx, dy) = (wrap(row), wrap(col));
                if dx.abs() <= reach && dy.abs() <= reach {
                    plane[(row * n as i64 + col) as usize] = Complex64::new(
                        time_window[((dx + reach) * dims as i64 + dy + reach) as usize],
                        0.0,
                    );
                }
            }
        }
        let fft: Arc<dyn Fft<f64>> = FftPlanner::new().plan_fft_forward(n);
        let mut kernel_hat = plane;
        apply_fft2(&*fft, &mut kernel_hat);
        let ifft: Arc<dyn Fft<f64>> = FftPlanner::new().plan_fft_inverse(n);
        Self {
            n,
            cell_m,
            band_cells,
            kernel_hat,
            time_window,
            fft,
            ifft,
        }
    }

    /// The forward circular-convolution apply, in place (same layout
    /// and normalization contract as [`ConvKernel::apply_forward`]).
    ///
    /// # Panics
    ///
    /// If `field.len()` differs from the raster side squared.
    pub fn apply_forward(&self, field: &mut [Complex64]) {
        assert_eq!(
            field.len(),
            self.n * self.n,
            "field size must equal the diff kernel raster"
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

    /// The kernel's cell pitch (metres), test-visible.
    pub fn cell_m(&self) -> f64 {
        self.cell_m
    }

    /// Raster side.
    pub fn raster(&self) -> usize {
        self.n
    }

    /// The kernel's cell pitch (metres), test-visible.
    pub fn diff_band(&self) -> usize {
        self.band_cells
    }

    /// The substitution stages' lookup: the model's difference value at
    /// signed displacement cells, zero outside the band.
    pub fn time_at(&self, dx: i64, dy: i64) -> f64 {
        let reach = self.band_cells as i64;
        if dx.abs() > reach || dy.abs() > reach {
            return 0.0;
        }
        let dims = 2 * self.band_cells + 1;
        self.time_window[((dx + reach) * dims as i64 + dy + reach) as usize]
    }
}

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
    /// The branch-CSR cross spine (v2 slice 2d); `None` under `Off`.
    spine: Option<SpineCsr>,
    /// The spine's Jacobi damping (the `Both` composition's stage;
    /// ignored when the LU stage list is empty).
    spine_damp: f64,
}

impl ConvPrecond {
    /// Builds the model: the layer-pair operator (omega-independent)
    /// plus this frequency's per-family inverse kernels. Per-family
    /// diagonal resistance rides the same-layer kernel's zero
    /// displacement as the population's mean branch resistance.
    ///
    /// Tab chassis (2c): `composition` selects the explicit
    /// per-census-patch blocks (`Both` default through the env
    /// knob); `tab_bin_m` bins the via census (the bin chain mirrors
    /// the census CLI's `census_for` fallback exactly:
    /// `FASTERHENRY_CENSUS_VIA_BIN_M`, then
    /// `FASTERHENRY_CENSUS_RADIUS_M`, then `census::derived_radius`
    /// - F3 doc fold 2026-10-20); `tab_pour` adds each patch's
    /// landing pours (`FASTERHENRY_CONV_TAB_POUR=1` default).
    pub fn build(
        filaments: &[Filament],
        resistances: &[f64],
        omega: f64,
        damp: f64,
        composition: TabComposition,
        tab_bin_m: f64,
        tab_pour: bool,
        spine_composition: TabComposition,
        spine_damp: f64,
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
            None,
        )?;
        let spine = Self::build_spine(
            filaments,
            resistances,
            omega,
            spine_composition,
            tab_bin_m,
            tab_pour,
            None,
        )?;
        Ok(Self {
            operator,
            inverses,
            tabs,
            spine,
            spine_damp,
        })
    }
    /// The explicit per-patch blocks: member branches from the
    /// census's flood fill, exact dense mutuals among them, LU
    /// factored when the smoother composition is live.
    ///
    /// `corr_fn` (the conv-matvec substitution, increment 5): when
    /// present, each block additionally carries per-pair substitution
    /// values `corr = L_exact - L_grid(pair) - Kdiff(u)` over the REAL
    /// member branch indices, swapped into the flux by `flux()`; the
    /// smoother-side builds pass `None`.
    fn build_tabs(
        filaments: &[Filament],
        resistances: &[f64],
        omega: f64,
        composition: TabComposition,
        tab_bin_m: f64,
        tab_pour: bool,
        corr_fn: Option<&dyn Fn(usize, usize) -> f64>,
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
            let corr = corr_fn.map(|f| {
                let indices = &members;
                (0..count)
                    .flat_map(|row| (0..count).map(move |col| f(indices[row], indices[col])))
                    .collect::<Vec<f64>>()
            });
            blocks.push(TabBlock::build(
                members,
                l,
                resistances,
                omega,
                smoother,
                corr,
            ));
        }
        Ok(blocks)
    }

    /// The branch-CSR cross spine (2d): the near-field couplings no
    /// other near-model stage carries - pairs with a via/tab-member
    /// endpoint and NOT both inside one patch's block. Pour-pour stays
    /// conv-borne even across patches; same-patch pairs stay in the
    /// block. CSR rows over branch indices, entries exact mutual L;
    /// the damped Jacobi denominators ride only the via rows.
    fn build_spine(
        filaments: &[Filament],
        resistances: &[f64],
        omega: f64,
        composition: TabComposition,
        tab_bin_m: f64,
        tab_pour: bool,
        corr_fn: Option<&dyn Fn(usize, usize) -> f64>,
    ) -> Result<Option<SpineCsr>, KernelError> {
        if composition == TabComposition::Off {
            return Ok(None);
        }
        let membership = crate::census::via_patch_membership(filaments, tab_bin_m, tab_pour);
        let vias: Vec<usize> = membership
            .members
            .iter()
            .flatten()
            .copied()
            .filter(|b| {
                membership.patch_of_branch[*b].is_some()
                    && filaments[*b].direction().z.abs() >= crate::census::VIA_IS_Z
            })
            .collect();
        if vias.is_empty() {
            return Ok(None);
        }
        // Deterministic slot order: the member lists ride the census
        // HashMap's discovery order, so sort the slots into deck
        // order (the wanted scan zips its output to this order).
        let mut vias = vias;
        vias.sort_unstable();
        // Spine radius = the via census bin, falling to the deck's
        // derived first guess when the bin is degenerate: the engine
        // lever (a CONV_SPINE radius knob) prices at the first ekart
        // row.
        let radius = if tab_bin_m.is_finite() && tab_bin_m > 0.0 {
            tab_bin_m
        } else {
            crate::census::derived_radius(filaments)
        };
        let all = crate::pfft::near_pairs_wanted(filaments, radius, &vias);
        // Slot-indexed entries: for via slot s, its partner list (any
        // deck branch unless same-patch). The flux apply scatters BOTH
        // directions per pair (mutuals are symmetric), so only the via
        // side is stored.
        let mut slot_rows: Vec<Vec<(u32, f64)>> = Vec::with_capacity(vias.len());
        let mut row_corr: Vec<Vec<f64>> = Vec::with_capacity(vias.len());
        for (&via, list) in vias.iter().zip(&all) {
            let mut row: Vec<(u32, f64)> = Vec::new();
            for &partner in list {
                let partner = partner as usize;
                if partner == via {
                    continue;
                }
                let src_patch = membership.patch_of_branch[via];
                if src_patch.is_some() && src_patch == membership.patch_of_branch[partner] {
                    continue;
                }
                // Canonical per pair: a partner that is itself a via
                // (in some patch) emits from its own slot when it is
                // the lower endpoint; a pour partner has no slot, so
                // the pair only exists here.
                if membership.patch_of_branch[partner].is_some() && partner < via {
                    continue;
                }
                let mutual = inductance::mutual_inductance(&filaments[via], &filaments[partner])?;
                row.push((partner as u32, mutual));
            }
            row.sort_by_key(|&(p, _)| p);
            row_corr.push(
                row.iter()
                    .map(|&(p, _)| corr_fn.map_or(0.0, |f| f(via, p as usize)))
                    .collect::<Vec<f64>>(),
            );
            slot_rows.push(row);
        }
        let mut row_ptr: Vec<u32> = Vec::with_capacity(vias.len() + 1);
        row_ptr.push(0u32);
        let mut col: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        let mut corr: Option<Vec<f64>> = corr_fn.map(|_| Vec::new());
        for (row, corr_row) in slot_rows.iter().zip(&row_corr) {
            for (&(partner, mutual), sub) in row.iter().zip(corr_row) {
                col.push(partner);
                values.push(mutual);
                if let Some(corr_values) = corr.as_mut() {
                    corr_values.push(*sub);
                }
            }
            row_ptr.push(col.len() as u32);
        }
        let jacobi = if composition == TabComposition::Both {
            slot_rows
                .iter()
                .zip(&vias)
                .map(|(row, &b)| {
                    let row_sum: f64 = row.iter().map(|&(_, m)| m.abs()).sum();
                    C64::new(
                        resistances[b],
                        omega * (row_sum + inductance::self_inductance(&filaments[b])),
                    )
                })
                .collect::<Vec<C64>>()
        } else {
            Vec::new()
        };
        Ok(Some(SpineCsr {
            row_ptr,
            col,
            values,
            jacobi,
            vias,
            corr,
        }))
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
        // The cross spine: both directions of every block-free via
        // pair; same no-kernels discipline as the tab blocks.
        if let Some(spine) = &self.spine {
            spine.flux(currents, &mut flux_re, &mut flux_im);
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
        // The spine's damped Jacobi over the via rows (a no-op when
        // no denominators exist).
        if let Some(spine) = &self.spine {
            spine.smooth_jacobi(residual, &mut correction, self.spine_damp);
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
    /// The conv-matvec substitution values (increment 5): per member
    /// pair the amount to swap for the exact mutual so the composition
    /// (grid far approximation + difference kernels + this) counts
    /// every pair exactly once: `corr = L_exact - L_grid - Kdiff(u)`.
    /// `None` for the smoother-side blocks.
    corr: Option<Vec<f64>>,
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
        corr: Option<Vec<f64>>,
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
        Self {
            members,
            l,
            lu,
            corr,
        }
    }

    /// The block's flux contribution: `flux[i] += L[i][j] *
    /// current[j]` over member pairs (real L, so the caller's real /
    /// imaginary flux accumulators both take it).
    fn flux(&self, currents: &[C64], flux_re: &mut [f64], flux_im: &mut [f64]) {
        let count = self.members.len();
        // The substitution values replace the exact mutuals in the
        // matvec composition (the exact amount is carried by the
        // surrounding stages - see the increment-5 decision); the
        // smoother side keeps the exact L.
        let values: &Vec<f64> = self.corr.as_ref().unwrap_or(&self.l);
        for (i, &dst) in self.members.iter().enumerate() {
            let row = &values[i * count..(i + 1) * count];
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

/// The branch-CSR cross spine (2d): the near-field couplings no other
/// near-model stage carries (pairs with a via endpoint, partners
/// outside their patch block). Only the via side of each pair is
/// stored (slot-indexed CSR); the flux apply scatters both directions
/// since mutuals are symmetric. The `Both` composition's smoother
/// stage is a damped Jacobi pass over the via rows, denominators the
/// branch's near-model local scale `R + j omega (L_self + row-sum
/// |L|)` (real parts in the re side, bounded gain, fixed per
/// frequency).
pub struct SpineCsr {
    /// Slot (via) row boundaries over `col`/`values`, length
    /// `vias.len() + 1`.
    row_ptr: Vec<u32>,
    col: Vec<u32>,
    values: Vec<f64>,
    /// Per slot the Jacobi denominator; empty under `Matvec`.
    jacobi: Vec<C64>,
    /// The slot's branch index (ascending deck order).
    vias: Vec<usize>,
    /// The conv-matvec substitution values (increment 5), aligned to
    /// `values`: per stored pair the swap amount
    /// `corr = L_exact - L_grid - Kdiff(u)`. `None` for the smoother
    /// side.
    corr: Option<Vec<f64>>,
}

impl SpineCsr {
    /// The spine's flux contribution: for via slot `s`, both directions
    /// of every stored pair (via's flux takes the partner's current and
    /// vice versa).
    fn flux(&self, currents: &[C64], flux_re: &mut [f64], flux_im: &mut [f64]) {
        if self.vias.is_empty() {
            return;
        }
        // Substitution values (matvec composition) replace the exact
        // mutuals - the exact amount is carried by the surrounding
        // stages.
        let values = self.corr.as_deref().unwrap_or(&self.values);
        for (slot, &via) in self.vias.iter().enumerate() {
            let lo = self.row_ptr[slot] as usize;
            let hi = self.row_ptr[slot + 1] as usize;
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for (index, &partner) in self.col[lo..hi].iter().enumerate() {
                let mutual = values[lo + index];
                sum_re += mutual * currents[partner as usize].re;
                sum_im += mutual * currents[partner as usize].im;
                // The transpose side (the via's current drops on the
                // partner).
                let transpose_re = mutual * currents[via].re;
                let transpose_im = mutual * currents[via].im;
                flux_re[partner as usize] += transpose_re;
                flux_im[partner as usize] += transpose_im;
            }
            flux_re[via] += sum_re;
            flux_im[via] += sum_im;
        }
    }

    /// The damped Jacobi smoother stage over the via rows:
    /// `correction[vias[s]] += residual[vias[s]] / denominators[s]`.
    fn smooth_jacobi(&self, residual: &[C64], correction: &mut [C64], damp: f64) {
        if self.jacobi.is_empty() {
            return;
        }
        for (slot, &via) in self.vias.iter().enumerate() {
            correction[via] += residual[via] * damp / self.jacobi[slot];
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
// ---------------------------------------------------------------------------
// The conv-matvec operator core (increment 5): the engine's equation
// becomes `grid far + conv near model` - see the increment-5 decision in
// the peec books. `MatvecCore` builds ONCE (all stages are real-L, omega
// folds in the caller's AC hop) and its `apply` is a matrix product,
// never an inverse.
// ---------------------------------------------------------------------------

/// Builds one family's difference kernel over the real far operator:
/// synthetic median source + displaceable victim, sample =
/// `mutual_inductance - far.pair_grid_estimate` (the per-pair closed
/// form of the grid's own estimate), over the near band only.
#[allow(clippy::too_many_arguments)]
fn sample_difference_kernel(
    far: &crate::pfft::PfftOperator,
    filaments: &[Filament],
    source_population: &[(usize, f64)],
    victim_population: &[(usize, f64)],
    src_axis: usize,
    dst_axis: usize,
    n: usize,
    cell_m: f64,
    band_cells: usize,
    self_family: bool,
) -> Result<DiffKernel, KernelError> {
    let src_segment = representative_segment(filaments, source_population, src_axis);
    let dst_segment = representative_segment(filaments, victim_population, dst_axis);
    // The synthetic pair must sit inside the grid's valid box: the
    // negative displacements would otherwise push the victim's
    // projection stencil past the grid edge, silently truncating the
    // double sum (measured on the accepted deck: the -u side returned
    // garbage ~1e-7 while +u decayed cleanly). Translate the pair so
    // its midpoint sits at the box centre; exact mutuals are
    // translation-invariant, so only the grid term's absolute phase
    // moves - the booked median-phase approximation of the difference
    // table.
    let translation = far
        .grid_box()
        .map(|(lo, hi)| (lo + hi) * 0.5 - (src_segment.center() + dst_segment.center()) * 0.5)
        .unwrap_or_else(Vector3::zeros);
    let translate = |segment: &Segment| -> Segment {
        let a = segment.a.position() + translation;
        let b = segment.b.position() + translation;
        Segment::new(
            Node::from([a.x, a.y, a.z]),
            Node::from([b.x, b.y, b.z]),
            segment.width,
            segment.height,
            segment.sigma,
        )
        .with_width_dir(
            segment
                .width_dir
                .expect("representative carries an explicit width direction"),
        )
    };
    let src =
        Filament::new(&translate(&src_segment)).expect("representative segment is well-formed");
    let (dst_a, dst_b) = (dst_segment.a.position(), dst_segment.b.position());
    let (width, height, sigma, width_dir) = (
        dst_segment.width,
        dst_segment.height,
        dst_segment.sigma,
        dst_segment.width_dir,
    );
    Ok(DiffKernel::build_diff(n, cell_m, band_cells, |dx, dy| {
        // The victim is displaced by the RELATIVE delta from its own
        // median position, then the pair rides the box-centre
        // translation: the table's (dx, dy) is the pair's true
        // displacement, so (0, 0) is the unshifted median pair and the
        // plane's displacement domain is centered.
        let offset = Vector3::new(dx as f64 * cell_m, dy as f64 * cell_m, 0.0) + translation;
        let displaced = Segment::new(
            Node::from([dst_a.x + offset.x, dst_a.y + offset.y, dst_a.z + offset.z]),
            Node::from([dst_b.x + offset.x, dst_b.y + offset.y, dst_b.z + offset.z]),
            width,
            height,
            sigma,
        )
        .with_width_dir(width_dir.expect("representative carries an explicit width direction"));
        let victim = Filament::new(&displaced).expect("displaced segment is well-formed");
        let exact = if self_family && dx == 0 && dy == 0 {
            // A true self family's zero displacement is the filament
            // with itself. Cross families (cross axis, cross layer)
            // have a coincident-but-distinct median pair here: the
            // exact mutual is finite (perpendicular pairs: exactly
            // zero) and the self shortcut would leak the self-L into
            // that family's table (measured: the accepted deck's via
            // cell leaked 3e-9 down its y->x family).
            inductance::self_inductance(&src)
        } else {
            inductance::mutual_inductance(&src, &victim)
                .expect("kernel-sample pair has aligned cross-sections by construction")
        };
        exact - far.pair_grid_estimate(&src, &victim)
    }))
}

/// The conv-matvec operator's near half: the same layer-pair population
/// structure as the smoother, but the family kernels are the FORWARD
/// difference kernels ([`DiffKernel`]) and the tab/spine stages carry
/// per-pair SUBSTITUTION values so the composition
/// `grid + kernels + tabs + spine` counts every member pair exactly
/// once. Unlike [`ConvPrecond`] this is the equation (part of it), so
/// every stage is a forward product; nothing here inverts anything.
pub struct MatvecCore {
    operator: LayerPairOperator,
    /// `kernels[li][lj][src_axis][dst_axis]`: the family's difference
    /// kernel; `None` where either population is empty.
    kernels: Vec<Vec<[[Option<DiffKernel>; 2]; 2]>>,
    /// The census-patch blocks in substitution form (`Off` composition
    /// produces none).
    tabs: Vec<TabBlock>,
    /// The branch-CSR cross spine in substitution form; `None` under
    /// the `Off` composition.
    spine: Option<SpineCsr>,
}

impl MatvecCore {
    /// Builds the near half alongside the FAR-only pFFT operator
    /// `far` (which owns the grid estimate every stage subtracts).
    ///
    /// Emits an honest header line about the composition (the
    /// increment-5 decision's labeling clause).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        far: &crate::pfft::PfftOperator,
        filaments: &[Filament],
        tab_composition: TabComposition,
        tab_bin_m: f64,
        tab_pour: bool,
        spine_composition: TabComposition,
    ) -> Result<Self, KernelError> {
        let build_start = std::time::Instant::now();
        let operator = LayerPairOperator::build(filaments)?;
        eprintln!(
            "matvec: layer operator: {} layers, raster {} cells, pitch {:.3e} m (t {:.1} s)",
            operator.layers.len(),
            operator.raster(),
            operator.pitch(),
            build_start.elapsed().as_secs_f64(),
        );
        let layer_count = operator.layer_count();
        // Near-band half width, in cells: the diff table's support.
        // Three caps stack: the pFFT's own near radius (the posture's
        // k * h) sets the physics; the raster never sees wrap-around
        // support; and the victim's full projection stencil must stay
        // inside the grid's valid box (the synthetic pair roams
        // +-band*pitch around the box centre), else the -u side samps
        // truncated stencils - silent garbage (measured on the
        // accepted deck). Past the band the difference model drops to
        // the grid alone: the per-pair error is the pFFT far error
        // class - the booked answer budget, not a correction failure.
        let radius_band = if far.near_radius() > 0.0 {
            (far.near_radius() / operator.pitch()).ceil() as usize + 1
        } else {
            1
        };
        let box_band = far.grid_box().map_or(radius_band, |(lo, hi)| {
            let centre = (lo + hi) * 0.5;
            let pitch = operator.pitch();
            let room_x = (centre.x - lo.x).min(hi.x - centre.x);
            let room_y = (centre.y - lo.y).min(hi.y - centre.y);
            ((room_x / pitch).floor() as usize).min((room_y / pitch).floor() as usize)
        });
        let band_cells = radius_band
            .min(box_band)
            .min(operator.raster() / 2 - 1)
            .max(1);
        let mut kernels: Vec<Vec<[[Option<DiffKernel>; 2]; 2]>> = Vec::with_capacity(layer_count);
        for li in 0..layer_count {
            let mut row: Vec<[[Option<DiffKernel>; 2]; 2]> = Vec::with_capacity(layer_count);
            for lj in 0..layer_count {
                let mut families: [[Option<DiffKernel>; 2]; 2] = Default::default();
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
                        let kernel = sample_difference_kernel(
                            far,
                            filaments,
                            src_population,
                            dst_population,
                            src_axis,
                            dst_axis,
                            operator.raster(),
                            operator.pitch(),
                            band_cells,
                            li == lj && src_axis == dst_axis,
                        )?;
                        eprintln!(
                            "matvec: kernel layer pair [{},{}] axis [{},{}] sampled (t {:.1} s)",
                            li,
                            lj,
                            src_axis,
                            dst_axis,
                            build_start.elapsed().as_secs_f64(),
                        );
                        families[src_axis][dst_axis] = Some(kernel);
                    }
                }
                row.push(families);
            }
            kernels.push(row);
        }
        // The substitution closure: per pair (i, j) the amount the
        // composition must swap in over the stage fluxes. The pair's
        // family lookup keys on the RECEIVER displacement
        // (centre_i - centre_j) in the (lj, li, aj, ai) family's
        // difference table; manhattan-only (vias have no family, so
        // their lookup is zero and the substitution is purely the
        // exact-vs-grid difference).
        let layer_of = branch_layers(filaments);
        let pitch = operator.pitch();
        // Scoped so the kernels borrow ends before the core moves them.
        let (tabs, spine) = {
            let kernels_ref = &kernels;
            let corr_fn = move |i: usize, j: usize| -> f64 {
                let (fi, fj) = (&filaments[i], &filaments[j]);
                // The substitution: the stage's flux REPLACES its exact
                // `l` with this value, so corr must be the full swap
                // `L_exact - L_grid(pair) - Kdiff(u)`: composed with the
                // far grid and the difference kernel the pair nets out
                // to the exact mutual exactly once.
                let mut sub = if i == j {
                    inductance::self_inductance(fi)
                } else {
                    inductance::mutual_inductance(fi, fj)
                        .expect("census patch pair has aligned geometry")
                };
                sub -= far.pair_grid_estimate(fi, fj);
                if let (Some((ai, si)), Some((aj, sj))) =
                    (flow_axis(fi.direction()), flow_axis(fj.direction()))
                {
                    let li = layer_of[i] as usize;
                    let lj = layer_of[j] as usize;
                    let displacement = fi.center() - fj.center();
                    if let Some(kernel) = &kernels_ref[lj][li][aj][ai] {
                        let dx = (displacement.x / pitch).round() as i64;
                        let dy = (displacement.y / pitch).round() as i64;
                        // The conv's per-pair coefficient rides the
                        // sign product (the antiparallel twin's entry
                        // is -h(u)), so the substitution amount does
                        // too.
                        sub -= kernel.time_at(dx, dy) * (si * sj);
                    }
                }
                sub
            };
            let tabs = Self::build_tabs(
                filaments,
                &[],
                tab_composition,
                tab_bin_m,
                tab_pour,
                &corr_fn,
            )?;
            eprintln!(
                "matvec: tabs built: {} blocks (t {:.1} s)",
                tabs.len(),
                build_start.elapsed().as_secs_f64(),
            );
            let spine = Self::build_spine(
                filaments,
                &[],
                spine_composition,
                tab_bin_m,
                tab_pour,
                &corr_fn,
            )?;
            eprintln!(
                "matvec: spine built (t {:.1} s)",
                build_start.elapsed().as_secs_f64(),
            );
            (tabs, spine)
        };
        eprintln!(
            "matvec: conv-matvec operator: {} layer pair families, band {} cells at pitch {:.3e} m, {} tab blocks, spine {} (t {:.1} s)",
            operator.layers.len(),
            band_cells,
            operator.pitch(),
            tabs.len(),
            spine.as_ref().map_or(0, |s| s.vias.len()),
            build_start.elapsed().as_secs_f64(),
        );
        Ok(Self {
            operator,
            kernels,
            tabs,
            spine,
        })
    }

    /// Builds the substitution-form tab blocks: same census as the
    /// smoother side, but each pair's table value is the SUBSTITUTION:
    /// `corr = L_exact - L_grid(pair) - Kdiff(u)` keyed on the REAL
    /// member branch indices via `corr_fn`.
    fn build_tabs(
        filaments: &[Filament],
        _resistances: &[f64],
        composition: TabComposition,
        tab_bin_m: f64,
        tab_pour: bool,
        corr_fn: &dyn Fn(usize, usize) -> f64,
    ) -> Result<Vec<TabBlock>, KernelError> {
        if composition == TabComposition::Off {
            return Ok(Vec::new());
        }
        let membership = crate::census::via_patch_membership(filaments, tab_bin_m, tab_pour);
        let block_total = membership.members.len();
        let tabs_start = std::time::Instant::now();
        let mut next_report_at = std::time::Duration::from_secs(60);
        let mut pairs_done: usize = 0;
        let mut blocks = Vec::new();
        for (block_id, members) in membership.members.into_iter().enumerate() {
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
            // Smooth-self composition: no LU on the operator side.
            let corr = {
                let indices = &members;
                (0..count)
                    .flat_map(|row| (0..count).map(move |col| corr_fn(indices[row], indices[col])))
                    .collect::<Vec<f64>>()
            };
            pairs_done += count * count;
            if tabs_start.elapsed() >= next_report_at {
                eprintln!(
                    "matvec: tabs: block {}/{}, member pairs {} (t {:.1} s)",
                    block_id + 1,
                    block_total,
                    pairs_done,
                    tabs_start.elapsed().as_secs_f64(),
                );
                next_report_at += std::time::Duration::from_secs(60);
            }
            blocks.push(TabBlock::build(members, l, &[], 0.0, false, Some(corr)));
        }
        Ok(blocks)
    }

    /// Builds the substitution-form spine (mirrors the smoother's
    /// construction; the `Both` Jacobi stage is smoother-only, empty
    /// here).
    fn build_spine(
        filaments: &[Filament],
        _resistances: &[f64],
        composition: TabComposition,
        tab_bin_m: f64,
        tab_pour: bool,
        corr_fn: &dyn Fn(usize, usize) -> f64,
    ) -> Result<Option<SpineCsr>, KernelError> {
        if composition == TabComposition::Off {
            return Ok(None);
        }
        let membership = crate::census::via_patch_membership(filaments, tab_bin_m, tab_pour);
        let vias: Vec<usize> = membership
            .members
            .iter()
            .flatten()
            .copied()
            .filter(|b| {
                membership.patch_of_branch[*b].is_some()
                    && filaments[*b].direction().z.abs() >= crate::census::VIA_IS_Z
            })
            .collect();
        if vias.is_empty() {
            return Ok(None);
        }
        let mut vias = vias;
        vias.sort_unstable();
        let mut radius = match std::env::var("FASTERHENRY_SPINE_RADIUS_M")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some(text) => match text.parse::<f64>() {
                Ok(value) if value.is_finite() && value > 0.0 => value,
                _ => {
                    eprintln!(
                        "conv: warning: FASTERHENRY_SPINE_RADIUS_M={text:?} not a positive number; falling back to the census bin"
                    );
                    0.0
                }
            },
            None => 0.0,
        };
        radius = if radius > 0.0 {
            radius
        } else if tab_bin_m.is_finite() && tab_bin_m > 0.0 {
            tab_bin_m
        } else {
            crate::census::derived_radius(filaments)
        };
        let spine_start = std::time::Instant::now();
        let all = crate::pfft::near_pairs_wanted(filaments, radius, &vias);
        let wanted_pairs: usize = all.iter().map(Vec::len).sum();
        eprintln!(
            "matvec: spine: {} via rows at partner radius {:.3e} m, partner pairs {}, scan done (t {:.1} s)",
            vias.len(),
            radius,
            wanted_pairs,
            spine_start.elapsed().as_secs_f64(),
        );
        let mut slot_rows: Vec<Vec<(u32, f64)>> = Vec::with_capacity(vias.len());
        let mut row_corr: Vec<Vec<f64>> = Vec::with_capacity(vias.len());
        let row_total = vias.len();
        let mut next_report_at = std::time::Duration::from_secs(60);
        let mut pairs_done: usize = 0;
        for (row_id, (&via, list)) in vias.iter().zip(all).enumerate() {
            let mut row: Vec<(u32, f64)> = Vec::new();
            for partner in list {
                let partner = partner as usize;
                if partner == via {
                    continue;
                }
                let src_patch = membership.patch_of_branch[via];
                if src_patch.is_some() && src_patch == membership.patch_of_branch[partner] {
                    continue;
                }
                if membership.patch_of_branch[partner].is_some() && partner < via {
                    continue;
                }
                let mutual = inductance::mutual_inductance(&filaments[via], &filaments[partner])?;
                row.push((partner as u32, mutual));
            }
            row.sort_by_key(|&(p, _)| p);
            row_corr.push(
                row.iter()
                    .map(|&(p, _)| corr_fn(via, p as usize))
                    .collect::<Vec<f64>>(),
            );
            pairs_done += row.len();
            if spine_start.elapsed() >= next_report_at {
                eprintln!(
                    "matvec: spine: row {}/{}, partner pairs {} (t {:.1} s)",
                    row_id + 1,
                    row_total,
                    pairs_done,
                    spine_start.elapsed().as_secs_f64(),
                );
                next_report_at += std::time::Duration::from_secs(60);
            }
            slot_rows.push(row);
        }
        let mut row_ptr: Vec<u32> = Vec::with_capacity(vias.len() + 1);
        row_ptr.push(0u32);
        let mut col: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        let mut corr: Vec<f64> = Vec::new();
        for (row, corr_row) in slot_rows.iter().zip(&row_corr) {
            for (&(partner, mutual), sub) in row.iter().zip(corr_row) {
                col.push(partner);
                values.push(mutual);
                corr.push(*sub);
            }
            row_ptr.push(col.len() as u32);
        }
        Ok(Some(SpineCsr {
            row_ptr,
            col,
            values,
            jacobi: Vec::new(),
            vias,
            corr: Some(corr),
        }))
    }

    /// The near half's flux: difference-kernel convolutions over the
    /// family populations (the model's estimate) plus the tab/spine
    /// substitution fluxes.
    pub fn apply(&self, currents: &[f64], fluxes: &mut [f64]) {
        assert_eq!(
            currents.len(),
            self.operator.place.len(),
            "current vector must cover every branch"
        );
        assert_eq!(
            fluxes.len(),
            self.operator.place.len(),
            "flux vector must cover every branch"
        );
        let n2 = self.operator.n * self.operator.n;
        for value in fluxes.iter_mut() {
            *value = 0.0;
        }
        let currents_c: Vec<Complex64> = currents.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        for (li, layer) in self.operator.layers.iter().enumerate() {
            for (src_axis, population) in layer.filaments.iter().enumerate() {
                if population.is_empty() {
                    continue;
                }
                let mut field = vec![Complex64::new(0.0, 0.0); n2];
                for &(branch, _) in population {
                    let placed = self.operator.place[branch]
                        .as_ref()
                        .expect("populated branch placed");
                    field[placed.cell] += Complex64::new(placed.sign * currents[branch], 0.0);
                }
                for (lj, victim) in self.operator.layers.iter().enumerate() {
                    for (dst_axis, victims) in victim.filaments.iter().enumerate() {
                        let Some(kernel) = &self.kernels[li][lj][src_axis][dst_axis] else {
                            continue;
                        };
                        let mut convolved = field.clone();
                        kernel.apply_forward(&mut convolved);
                        for &(branch, _) in victims {
                            let placed = self.operator.place[branch]
                                .as_ref()
                                .expect("populated branch placed");
                            fluxes[branch] += placed.sign * convolved[placed.cell].re;
                        }
                    }
                }
            }
        }
        // Substitution stages in complex form (the tab/spine fluxes
        // already read the corr values; zero omega on the hop since
        // corr is real and the caller's AC lambda folds omega in).
        let mut flux_re = vec![0.0f64; currents.len()];
        let mut flux_im = vec![0.0f64; currents.len()];
        for tab in &self.tabs {
            tab.flux(&currents_c, &mut flux_re, &mut flux_im);
        }
        if let Some(spine) = &self.spine {
            spine.flux(&currents_c, &mut flux_re, &mut flux_im);
        }
        for (dst, re) in fluxes.iter_mut().zip(&flux_re) {
            *dst += *re;
        }
    }
}

impl std::fmt::Debug for MatvecCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatvecCore")
            .field("layers", &self.operator.layers.len())
            .field("tabs", &self.tabs.len())
            .field(
                "spine_slots",
                &self.spine.as_ref().map_or(0, |s| s.vias.len()),
            )
            .finish()
    }
}

#[cfg(test)]
mod matvec_tests {
    use super::*;

    // The uniform-deck fixture lives in `layer_pair_tests` below; reuse
    // it here by path.
    use crate::conv::layer_pair_tests::{deck, PITCH};

    /// The composed conv-matvec operator - far pFFT + difference-kernel
    /// conv + tab/spine substitution - must reproduce every mutual the
    /// precorrect operator computes, per pair, on the all-identical
    /// uniform deck: the grid term of the difference kernel and the real
    /// pairs share their pitch-aligned phase there, so the residual is
    /// the synthetic-pair phase class (~1e-10 absolute, measured) and
    /// the substitution stages (tab: the shared via cell incl. the
    /// antiparallel twin; spine: the via) net to float noise. The
    /// kernel stage must be linear per column (measured); the
    /// composition is asserted column-wise against the precorrect
    /// reference, plus one multi-source vector against the dense path.
    #[test]
    fn matvec_composition_matches_the_precorrect_per_pair() {
        let deck = deck();
        let params = crate::pfft::PfftParams::default();
        let full = crate::pfft::PfftOperator::new(&deck, &params)
            .expect("deck builds a precorrect operator");
        let mut far_params = params;
        far_params.precorrect_near = false;
        let far =
            crate::pfft::PfftOperator::new(&deck, &far_params).expect("deck builds a far operator");

        let core = MatvecCore::build(
            &far,
            &deck,
            TabComposition::Both,
            PITCH,
            true,
            TabComposition::Both,
        )
        .expect("deck builds a matvec core");

        // The reference (the precorrect operator) is the exact answer
        // here, so the per-pair ceiling is the measured phase class.
        let ceiling = 5e-10;
        for j in 0..deck.len() {
            let mut unit = vec![0.0f64; deck.len()];
            unit[j] = 1.0;
            let reference = full.apply(&unit);
            let grid = far.apply(&unit);
            let mut flux = vec![0.0f64; deck.len()];
            core.apply(&unit, &mut flux);
            for (i, total) in grid.iter().zip(&flux).map(|(g, f)| g + f).enumerate() {
                assert!(
                    (total - reference[i]).abs() <= ceiling,
                    "composed pair ({i} <- {j}): total {total:e} exact {:e} \
                     off the measured phase class",
                    reference[i]
                );
            }
        }

        // Multi-source: signed currents across all branches.
        let x: Vec<f64> = (0..deck.len())
            .map(|i| {
                if i % 3 == 0 {
                    1.0
                } else {
                    -0.5 - 0.1 * i as f64
                }
            })
            .collect();
        let reference = full.apply(&x);
        let grid = far.apply(&x);
        let mut flux = vec![0.0f64; deck.len()];
        core.apply(&x, &mut flux);
        let mut worst = (0usize, 0.0f64);
        for (i, total) in grid.iter().zip(&flux).map(|(g, f)| g + f).enumerate() {
            let residual = (total - reference[i]).abs();
            if residual > worst.1 {
                worst = (i, residual);
            }
        }
        // Twelve unit-ish sources, each pair inside the ceiling: the
        // measured multi-source worst is ~2e-10 absolute.
        assert!(
            worst.1 <= 2.5e-9,
            "multi-source composition residual {:e} at branch {} beyond the measured class",
            worst.1,
            worst.0
        );
        // PfftOperator::new records its phase on the global memgate
        // governor; reset it so parallel in-crate tests (the memgate
        // stage-name fixture) see the init stage.
        crate::memgate::set_stage(crate::memgate::STAGE_INIT);
    }
}
