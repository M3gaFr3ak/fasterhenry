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
    /// The damped inverse spectrum - what the apply multiplies by.
    inverse_hat: Vec<Complex64>,
    /// The regularization floor used, `(damp * peak)^2`.
    epsilon: f64,
}

impl ConvKernel {
    /// Builds the damped spectral inverse. `sample(dispx, dispy)`
    /// supplies the EXACT kernel value at an XY displacement point in
    /// metres; it must be even in both axes for the wrapped spectrum
    /// to stay real (an odd component folds into sign-aliased noise;
    /// the spectral multiply only preserves magnitude then).
    /// `damp` in `(0, 1)` sets the regularization floor.
    pub fn build(n: usize, cell_m: f64, sample: impl Fn(f64, f64) -> f64, damp: f64) -> Self {
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
        let half = n as i64 / 2;
        let mut plane: Vec<Complex64> = Vec::with_capacity(n * n);
        for row in 0..n as i64 {
            for col in 0..n as i64 {
                let dx = (row - half) as f64 * cell_m;
                let dy = (col - half) as f64 * cell_m;
                plane.push(Complex64::new(sample(dx, dy), 0.0));
            }
        }
        let fft = FftPlanner::new().plan_fft_forward(n);
        apply_fft2(&*fft, &mut plane);
        let mut peak = 0.0f64;
        for value in &plane {
            peak = peak.max(value.re.abs().max(value.im.abs()));
        }
        let epsilon = (damp * peak).powi(2);
        let mut inverse_hat = Vec::with_capacity(n * n);
        for value in &plane {
            // An even real kernel has a real spectrum up to FFT
            // rounding: the re-signed inversion preserves each band's
            // coupling sign (negative-mutual bands correct with the
            // right polarity).
            let h = value.re;
            inverse_hat.push(Complex64::new(h / (h * h + epsilon), 0.0));
        }
        ConvKernel {
            n,
            cell_m,
            kernel_hat: plane,
            inverse_hat,
            epsilon,
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
        let fft = FftPlanner::new().plan_fft_forward(n);
        let ifft = FftPlanner::new().plan_fft_inverse(n);
        apply_fft2(&*fft, field);
        for (value, gain) in field.iter_mut().zip(&self.inverse_hat) {
            *value = *value * *gain;
        }
        apply_fft2(&*ifft, field);
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
        let fft = FftPlanner::new().plan_fft_forward(n);
        let ifft = FftPlanner::new().plan_fft_inverse(n);
        apply_fft2(&*fft, field);
        for (value, gain) in field.iter_mut().zip(&self.kernel_hat) {
            *value = *value * *gain;
        }
        apply_fft2(&*ifft, field);
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
