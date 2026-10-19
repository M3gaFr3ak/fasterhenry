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
    /// The displacement-domain samples before the FFT (the same
    /// wrapped layout): the value a single-source probe reads back.
    kernel_time: Vec<f64>,
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
                plane.push(Complex64::new(
                    sample(raw_displacement(row), raw_displacement(col)),
                    0.0,
                ));
            }
        }
        let kernel_time: Vec<f64> = plane.iter().map(|v| v.re).collect();
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
            kernel_time,
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

/// A filament's raster placement: the positional projection plus the
/// filament's direction sign relative to the +axis reference (the
/// single source of truth for the write/read folding).
#[derive(Clone, Copy)]
struct Place {
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
        Some(src.sign * dst.sign * kernel.kernel_time[index])
    }
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
    /// The population's representative segment: median
    /// length/width/height/sigma, median z, +axis direction, explicit
    /// width direction so the cross-section orientation is
    /// deterministic.
    fn representative(filaments: &[Filament], population: &[(usize, f64)], axis: usize) -> Segment {
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
    let src_segment = representative(filaments, source_population, src_axis);
    let dst_segment = representative(filaments, victim_population, dst_axis);
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
            if same_layer && dx == 0.0 && dy == 0.0 {
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
}
