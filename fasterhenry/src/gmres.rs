//! Restarted GMRES for matrix-free linear systems, real or complex.
//!
//! [`gmres`] solves `A·x = b` given only the product `v ↦ A·v`, by the
//! generalized minimal residual method of Saad & Schultz (*SIAM J. Sci. Stat.
//! Comput.* 7, 1986), restarted every [`GmresParams::restart`] iterations and
//! right-preconditioned: it iterates on `A·M` with `M ≈ A⁻¹` a
//! caller-supplied preconditioner and returns `x = M·u`. With right
//! preconditioning the residual GMRES minimizes is the true residual
//! `b − A·x`, so the stopping test is on the quantity the caller cares about
//! whatever the preconditioner.
//!
//! The Krylov basis is orthonormalized by modified Gram–Schmidt applied
//! twice (one full re-orthogonalization pass, which keeps the basis
//! orthogonal to working precision where a single pass can lose it); the
//! small Hessenberg least-squares problem is reduced by complex Givens
//! rotations, whose running product gives the residual norm of every
//! iterate for free. Inner products are the Hermitian ones, `⟨u, v⟩ =
//! Σ conj(uᵢ)·vᵢ`: GMRES minimizes the Euclidean norm and needs no symmetry
//! of `A`, which is why it suits the complex-*symmetric* (not Hermitian)
//! mesh impedance matrix.
//!
//! The scalar type is any `nalgebra` [`ComplexField`] over `f64`, so the same
//! code runs in real arithmetic (`f64`) — every rotation is then real and no
//! imaginary part ever appears — and in complex arithmetic
//! (`Complex<f64>`). Every operation is in a fixed order, so the result is a
//! deterministic function of the inputs and of the two callbacks.

use nalgebra::ComplexField;

/// Stopping and restart parameters of [`gmres`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmresParams {
    /// Converged once `‖b − A·x‖₂ ≤ tolerance · ‖b‖₂`. Default:
    /// [`DEFAULT_TOLERANCE`].
    pub tolerance: f64,
    /// Krylov dimension between restarts; memory is `restart + 1` vectors.
    /// Default: [`DEFAULT_RESTART`].
    pub restart: usize,
    /// Cap on the total number of products with `A`, over all restarts.
    /// Default: [`DEFAULT_MAX_ITERATIONS`].
    pub max_iterations: usize,
}

/// Default [`GmresParams::tolerance`].
pub const DEFAULT_TOLERANCE: f64 = 1e-10;
/// Default [`GmresParams::restart`].
pub const DEFAULT_RESTART: usize = 100;
/// Default [`GmresParams::max_iterations`].
pub const DEFAULT_MAX_ITERATIONS: usize = 5000;

impl Default for GmresParams {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            restart: DEFAULT_RESTART,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

/// How a [`gmres`] run ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmresOutcome {
    /// Products with `A` spent in the Arnoldi iterations (the one extra
    /// product per restart that recomputes the true residual is not
    /// counted).
    pub iterations: usize,
    /// The true relative residual `‖b − A·x‖₂ / ‖b‖₂` of the returned `x`
    /// (`0` when `b = 0`).
    pub relative_residual: f64,
    /// Whether `relative_residual` met [`GmresParams::tolerance`].
    pub converged: bool,
}

/// Solves `A·x = b` by restarted, right-preconditioned GMRES; see the
/// [module documentation](self).
///
/// `apply(v)` must return `A·v` and `precondition(v)` must return `M·v`, for
/// a fixed nonsingular `M` (pass `|v| v.to_vec()` for none). `x` holds the
/// initial guess on entry and the solution on return. When `b = 0` the
/// solution is `x = 0` without a single product.
///
/// Iteration stops when the tolerance is met, when
/// [`GmresParams::max_iterations`] products have been spent, or when the
/// residual stops being finite; the [`GmresOutcome`] says which.
///
/// # Panics
///
/// If `x` and `b` differ in length, or a callback returns a vector of the
/// wrong length.
pub fn gmres<T, A, P>(
    mut apply: A,
    mut precondition: P,
    b: &[T],
    x: &mut [T],
    params: &GmresParams,
) -> GmresOutcome
where
    T: ComplexField<RealField = f64> + Copy,
    A: FnMut(&[T]) -> Vec<T>,
    P: FnMut(&[T]) -> Vec<T>,
{
    let n = b.len();
    assert_eq!(x.len(), n, "gmres: x and b differ in length");
    let b_norm = norm(b);
    if b_norm == 0.0 {
        x.fill(T::zero());
        return GmresOutcome {
            iterations: 0,
            relative_residual: 0.0,
            converged: true,
        };
    }
    let target = params.tolerance * b_norm;
    let restart = params.restart.clamp(1, n);

    let mut iterations = 0;
    let mut residual = residual_of(&mut apply, b, x);
    let mut r_norm = norm(&residual);
    loop {
        let outcome = |converged| GmresOutcome {
            iterations,
            relative_residual: r_norm / b_norm,
            converged,
        };
        if r_norm <= target {
            return outcome(true);
        }
        if iterations >= params.max_iterations || !r_norm.is_finite() {
            return outcome(false);
        }

        // One Arnoldi cycle from the current residual.
        let mut basis: Vec<Vec<T>> = Vec::with_capacity(restart + 1);
        basis.push(residual.iter().map(|&r| r.unscale(r_norm)).collect());
        // Columns of the rotated Hessenberg matrix (upper triangular).
        let mut hessenberg: Vec<Vec<T>> = Vec::with_capacity(restart);
        let mut rotations: Vec<(f64, T)> = Vec::with_capacity(restart);
        let mut g = vec![T::zero(); restart + 1];
        g[0] = T::from_real(r_norm);
        while hessenberg.len() < restart && iterations < params.max_iterations {
            let j = hessenberg.len();
            let mut w = apply(&precondition(&basis[j]));
            assert_eq!(w.len(), n, "gmres: apply returned the wrong length");
            iterations += 1;

            let mut column = vec![T::zero(); j + 2];
            for _pass in 0..2 {
                for (v, h) in basis.iter().zip(column.iter_mut()) {
                    let projection = dot(v, &w);
                    *h += projection;
                    axpy(-projection, v, &mut w);
                }
            }
            let next_norm = norm(&w);
            column[j + 1] = T::from_real(next_norm);

            for (i, &(c, s)) in rotations.iter().enumerate() {
                let (a, b) = (column[i], column[i + 1]);
                column[i] = a.scale(c) + s * b;
                column[i + 1] = b.scale(c) - s.conjugate() * a;
            }
            let (c, s, r) = givens(column[j], column[j + 1]);
            column[j] = r;
            column[j + 1] = T::zero();
            rotations.push((c, s));
            let gj = g[j];
            g[j] = gj.scale(c);
            g[j + 1] = -(s.conjugate() * gj);
            hessenberg.push(column);
            eprintln!("gmres: iter {iterations}, |r| ~ {:.3e}", g[j + 1].modulus());

            // `next_norm == 0` is a lucky breakdown: the Krylov space is
            // invariant and the least-squares solution is exact.
            if next_norm == 0.0 || g[j + 1].modulus() <= target {
                break;
            }
            basis.push(w.iter().map(|&wi| wi.unscale(next_norm)).collect());
        }

        // Back substitution for the cycle's coefficients, then x += M·(V·y).
        let steps = hessenberg.len();
        let mut y = vec![T::zero(); steps];
        for i in (0..steps).rev() {
            let mut sum = g[i];
            for k in i + 1..steps {
                sum -= hessenberg[k][i] * y[k];
            }
            y[i] = sum / hessenberg[i][i];
        }
        let mut update = vec![T::zero(); n];
        for (v, &yk) in basis.iter().zip(&y) {
            axpy(yk, v, &mut update);
        }
        let correction = precondition(&update);
        assert_eq!(
            correction.len(),
            n,
            "gmres: precondition returned the wrong length"
        );
        for (xi, ci) in x.iter_mut().zip(correction) {
            *xi += ci;
        }
        residual = residual_of(&mut apply, b, x);
        r_norm = norm(&residual);
        // True residual at the cycle boundary (the restart point), printed
        // next to the estimate `g[steps]` the inner loop last reported - the
        // estimate-vs-true drift is what decides whether this restarts.
        eprintln!(
            "gmres: cycle end at iter {iterations}: true |r| = {r_norm:.4e}, est |r| = {:.4e}",
            g[steps].modulus(),
        );
    }
}

/// `b − A·x`.
fn residual_of<T, A>(apply: &mut A, b: &[T], x: &[T]) -> Vec<T>
where
    T: ComplexField<RealField = f64> + Copy,
    A: FnMut(&[T]) -> Vec<T>,
{
    let ax = apply(x);
    assert_eq!(ax.len(), b.len(), "gmres: apply returned the wrong length");
    b.iter().zip(ax).map(|(&bi, ai)| bi - ai).collect()
}

/// The Hermitian inner product `Σ conj(uᵢ)·vᵢ`.
fn dot<T: ComplexField<RealField = f64> + Copy>(u: &[T], v: &[T]) -> T {
    u.iter()
        .zip(v)
        .fold(T::zero(), |sum, (&ui, &vi)| sum + ui.conjugate() * vi)
}

/// `y += a·x`.
fn axpy<T: ComplexField<RealField = f64> + Copy>(a: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// Euclidean norm, scaled against overflow and underflow.
fn norm<T: ComplexField<RealField = f64> + Copy>(v: &[T]) -> f64 {
    let scale = v.iter().fold(0.0_f64, |m, e| m.max(e.modulus()));
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let sum: f64 = v.iter().map(|e| e.unscale(scale).modulus_squared()).sum();
    scale * sum.sqrt()
}

/// The rotation `G = [c s; −s̄ c]`, `c` real, with `G·[a; b] = [r; 0]`.
fn givens<T: ComplexField<RealField = f64> + Copy>(a: T, b: T) -> (f64, T, T) {
    let (abs_a, abs_b) = (a.modulus(), b.modulus());
    if abs_b == 0.0 {
        return (1.0, T::zero(), a);
    }
    if abs_a == 0.0 {
        return (0.0, T::one(), b);
    }
    let length = abs_a.hypot(abs_b);
    let phase = a.unscale(abs_a);
    let c = abs_a / length;
    let s = phase * b.conjugate().unscale(length);
    (c, s, phase.scale(length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use num_complex::Complex;

    type C = Complex<f64>;

    /// A deterministic pseudo-random stream in `[-1, 1)`.
    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    fn complex_symmetric(n: usize, seed: u64) -> DMatrix<C> {
        let mut next = stream(seed);
        let mut a = DMatrix::from_fn(n, n, |_, _| C::new(next(), next()) * 0.3);
        a = &a + a.transpose();
        for i in 0..n {
            a[(i, i)] += C::new(n as f64, 0.5 * n as f64);
        }
        a
    }

    fn apply_matrix<'a, T: ComplexField<RealField = f64> + Copy>(
        a: &'a DMatrix<T>,
    ) -> impl FnMut(&[T]) -> Vec<T> + 'a {
        move |v| (a * DVector::from_column_slice(v)).as_slice().to_vec()
    }

    #[test]
    fn complex_symmetric_system_converges_to_the_direct_solution() {
        let n = 60;
        let a = complex_symmetric(n, 3);
        let mut next = stream(7);
        let b: Vec<C> = (0..n).map(|_| C::new(next(), next())).collect();
        let mut x = vec![C::new(0.0, 0.0); n];
        let params = GmresParams {
            restart: 8,
            ..GmresParams::default()
        };
        let outcome = gmres(apply_matrix(&a), |v| v.to_vec(), &b, &mut x, &params);
        assert!(outcome.converged, "{outcome:?}");
        assert!(outcome.relative_residual <= 1e-10);
        let direct = a.clone().lu().solve(&DVector::from_vec(b)).unwrap();
        let error = (DVector::from_vec(x) - &direct).norm() / direct.norm();
        assert!(error < 1e-9, "error {error}");
    }

    #[test]
    fn real_system_stays_real_and_matches_the_direct_solution() {
        let n = 40;
        let mut next = stream(11);
        let mut a = DMatrix::from_fn(n, n, |_, _| next());
        for i in 0..n {
            a[(i, i)] += 2.0 * n as f64;
        }
        let b: Vec<f64> = (0..n).map(|_| next()).collect();
        let mut x = vec![0.0; n];
        let outcome = gmres(
            apply_matrix(&a),
            |v| v.to_vec(),
            &b,
            &mut x,
            &GmresParams::default(),
        );
        assert!(outcome.converged, "{outcome:?}");
        let direct = a.lu().solve(&DVector::from_vec(b)).unwrap();
        assert!((DVector::from_vec(x) - &direct).norm() < 1e-9 * direct.norm());
    }

    #[test]
    fn full_krylov_space_solves_exactly_in_n_steps() {
        // Without restarts GMRES terminates in at most n iterations.
        let n = 12;
        let a = complex_symmetric(n, 5);
        let b: Vec<C> = (0..n).map(|i| C::new(i as f64, 1.0)).collect();
        let mut x = vec![C::new(0.0, 0.0); n];
        let params = GmresParams {
            tolerance: 1e-13,
            restart: n,
            max_iterations: n,
        };
        let outcome = gmres(apply_matrix(&a), |v| v.to_vec(), &b, &mut x, &params);
        assert!(outcome.converged, "{outcome:?}");
        assert!(outcome.iterations <= n);
    }

    #[test]
    fn exact_preconditioner_converges_in_one_iteration() {
        let n = 20;
        let a = complex_symmetric(n, 9);
        let inverse = a.clone().try_inverse().unwrap();
        let b: Vec<C> = (0..n).map(|i| C::new(1.0, -(i as f64))).collect();
        let mut x = vec![C::new(0.0, 0.0); n];
        let outcome = gmres(
            apply_matrix(&a),
            apply_matrix(&inverse),
            &b,
            &mut x,
            &GmresParams::default(),
        );
        assert!(outcome.converged, "{outcome:?}");
        assert_eq!(outcome.iterations, 1);
    }

    #[test]
    fn zero_right_hand_side_is_solved_without_a_product() {
        let mut x = vec![C::new(3.0, 1.0); 4];
        let outcome = gmres(
            |_: &[C]| -> Vec<C> { unreachable!() },
            |v: &[C]| v.to_vec(),
            &[C::new(0.0, 0.0); 4],
            &mut x,
            &GmresParams::default(),
        );
        assert_eq!(outcome.iterations, 0);
        assert!(outcome.converged);
        assert!(x.iter().all(|e| *e == C::new(0.0, 0.0)));
    }

    #[test]
    fn iteration_cap_reports_non_convergence() {
        let n = 30;
        let a = complex_symmetric(n, 13);
        let b = vec![C::new(1.0, 0.0); n];
        let mut x = vec![C::new(0.0, 0.0); n];
        let params = GmresParams {
            tolerance: 1e-15,
            restart: 2,
            max_iterations: 3,
        };
        let outcome = gmres(apply_matrix(&a), |v| v.to_vec(), &b, &mut x, &params);
        assert!(!outcome.converged);
        assert_eq!(outcome.iterations, 3);
        assert!(outcome.relative_residual < 1.0);
    }

    #[test]
    fn givens_zeroes_the_second_component() {
        let (a, b) = (C::new(0.3, -1.2), C::new(-2.0, 0.7));
        let (c, s, r) = givens(a, b);
        let top = a * c + s * b;
        let bottom = b * c - s.conj() * a;
        assert!((top - r).norm() < 1e-15);
        assert!(bottom.norm() < 1e-15);
        assert!((r.norm() - a.norm().hypot(b.norm())).abs() < 1e-15);
    }
}
