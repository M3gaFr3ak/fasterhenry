//! Validation of the partial-inductance kernels through the public API.
//!
//! No tabulated literature values are hard-coded here. Every kernel is
//! checked against an *independent numerical evaluation of its defining
//! integral*, written in this file and sharing no code with the production
//! paths, and against analytic limits:
//!
//! * [`reference_self`] — the sixfold self integral reduced (by the
//!   distribution of coordinate differences) to a threefold one whose `1/r`
//!   corner singularity is removed exactly by a pyramidal Duffy map;
//! * [`reference_mutual`] — brute-force composite Gauss–Legendre quadrature
//!   of the sixfold Neumann volume integral, for separated bars;
//! * [`thin_bar_self`] — the Grover/Rosa long-conductor expansion built on
//!   Maxwell's exact geometric mean distance of a rectangle;
//! * far-field, scaling and symmetry identities.

use fasterhenry::inductance::{
    closed_form, mutual_batch_detailed_with, mutual_batch_with, mutual_inductance_detailed,
    Execution, Method,
};
use fasterhenry::{
    discretize, mutual_batch, mutual_batch_detailed, mutual_inductance, partial_inductance_matrix,
    partial_inductance_matrix_detailed, partial_inductance_method_counts, self_inductance,
    Filament, KernelError, MutualBatch, Node, Segment, MU0,
};
use nalgebra::{DMatrix, Vector3};

const COPPER: f64 = 5.8e7;

// ---------------------------------------------------------------------------
// Independent references
// ---------------------------------------------------------------------------

/// Gauss–Legendre nodes and weights on `[0, 1]`, by Newton iteration —
/// deliberately a separate implementation from the crate's.
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|k| {
            let mut x = (std::f64::consts::PI * (k as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut dp = 0.0;
            for _ in 0..60 {
                let (mut p0, mut p1) = (1.0, x);
                for j in 2..=n {
                    let jf = j as f64;
                    (p0, p1) = (p1, ((2.0 * jf - 1.0) * x * p1 - (jf - 1.0) * p0) / jf);
                }
                dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                x -= p1 / dp;
            }
            (0.5 * (x + 1.0), 1.0 / ((1.0 - x * x) * dp * dp))
        })
        .collect()
}

/// Partial self-inductance of an `l × w × h` bar from its definition.
///
/// `∫∫ dV dV'/|r − r'|` over one box equals
/// `8 ∫₀ˡ∫₀ʷ∫₀ʰ (l−x)(w−y)(h−z)/√(x²+y²+z²)`, since the differences of two
/// uniform coordinates have a triangular density. The box is split into
/// three pyramids with apex at the origin; on each, `x = l·s`, `y = w·s·t₁`,
/// `z = h·s·t₂` (and permutations) has Jacobian `lwh·s²`, which cancels the
/// `1/s` of `1/r` and leaves a smooth integrand.
fn reference_self(l: f64, w: f64, h: f64) -> f64 {
    let radial = gauss_legendre(4);
    let angular = gauss_legendre(56);
    let e = [l, w, h];
    let mut sum = 0.0;
    for apex_axis in 0..3 {
        let (i, j, k) = (apex_axis, (apex_axis + 1) % 3, (apex_axis + 2) % 3);
        for &(s, ws) in &radial {
            for &(t1, w1) in &angular {
                for &(t2, w2) in &angular {
                    let mut p = [0.0; 3];
                    p[i] = e[i] * s;
                    p[j] = e[j] * s * t1;
                    p[k] = e[k] * s * t2;
                    let r_over_s =
                        (e[i] * e[i] + e[j] * e[j] * t1 * t1 + e[k] * e[k] * t2 * t2).sqrt();
                    let density = (l - p[0]) * (w - p[1]) * (h - p[2]);
                    sum += ws * w1 * w2 * density * s / r_over_s;
                }
            }
        }
    }
    let integral = 8.0 * sum * l * w * h;
    MU0 / (4.0 * std::f64::consts::PI) * integral / (w * h * w * h)
}

/// Brute-force sixfold Neumann integral between two filaments' volumes:
/// `length_panels` panels of 8 points along each length, `cross` points
/// across each width and height. Only meaningful for separated bars.
fn reference_mutual(a: &Filament, b: &Filament, length_panels: usize, cross: usize) -> f64 {
    let cloud = |f: &Filament| {
        let mut points = Vec::new();
        for panel in 0..length_panels {
            for &(x, wx) in &gauss_legendre(8) {
                let s = (panel as f64 + x) / length_panels as f64 - 0.5;
                for &(y, wy) in &gauss_legendre(cross) {
                    for &(z, wz) in &gauss_legendre(cross) {
                        let p = f.center()
                            + f.direction() * (s * f.length())
                            + f.width_dir() * ((y - 0.5) * f.width())
                            + f.height_dir() * ((z - 0.5) * f.height());
                        points.push((p, wx * wy * wz * f.length() / length_panels as f64));
                    }
                }
            }
        }
        points
    };
    let (pa, pb) = (cloud(a), cloud(b));
    let mut sum = 0.0;
    for (p, wp) in &pa {
        let inner: f64 = pb.iter().map(|(q, wq)| wq / (p - q).norm()).sum();
        sum += wp * inner;
    }
    MU0 / (4.0 * std::f64::consts::PI) * a.direction().dot(&b.direction()) * sum
}

/// `ln` of the geometric mean distance of a `w × h` rectangle from itself
/// (Maxwell, *Treatise* §692; Rosa, *Bull. Bur. Stand.* 3 (1907); Grover
/// ch. 3). Tends to `ln w − 3/2` for a line.
fn ln_gmd_rectangle(w: f64, h: f64) -> f64 {
    let (a, b) = (w / h, h / w);
    0.5 * (w * w + h * h).ln() - a * a / 12.0 * (b * b).ln_1p() - b * b / 12.0 * (a * a).ln_1p()
        + 2.0 / 3.0 * (a * b.atan() + b * a.atan())
        - 25.0 / 12.0
}

/// Mean distance between two uniformly random points of a `w × h`
/// rectangle (a classical result, 0.5214… for the unit square), arranged to
/// avoid cancellation for thin rectangles.
fn mean_distance_rectangle(w: f64, h: f64) -> f64 {
    let (long, t) = (w.max(h), w.min(h) / w.max(h));
    let s = t.hypot(1.0);
    long / 15.0
        * (t * t * t - 1.0 / (1.0 + s)
            + s * (3.0 - t * t)
            + 2.5 * (t * t * ((1.0 + s) / t).ln() + t.asinh() / t))
}

/// Partial self-inductance of a long bar: averaging Grover's equal parallel
/// filament formula `2l·[ln(2l/ρ) − 1 + ρ/l − ρ²/(4l²) + O(ρ⁴/l⁴)]` over all
/// pairs of points of the cross-section, which needs exactly the geometric
/// mean, arithmetic mean and mean-square distances. This is the Grover/Rosa
/// thin-conductor formula with its usual `0.2235·(w + h)` approximations
/// replaced by their exact values.
fn thin_bar_self(l: f64, w: f64, h: f64) -> f64 {
    MU0 * l / (2.0 * std::f64::consts::PI)
        * ((2.0 * l).ln() - ln_gmd_rectangle(w, h) - 1.0 + mean_distance_rectangle(w, h) / l
            - (w * w + h * h) / (24.0 * l * l))
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn filament(a: [f64; 3], b: [f64; 3], width: f64, height: f64) -> Filament {
    Filament::new(&Segment::new(
        Node::from(a),
        Node::from(b),
        width,
        height,
        COPPER,
    ))
    .unwrap()
}

fn filament_with(a: [f64; 3], b: [f64; 3], width: f64, height: f64, dir: [f64; 3]) -> Filament {
    let segment =
        Segment::new(Node::from(a), Node::from(b), width, height, COPPER).with_width_dir(dir);
    Filament::new(&segment).unwrap()
}

/// A bar of the given size along +x with its width along +y.
fn bar_x(l: f64, w: f64, h: f64) -> Filament {
    filament_with([0.0; 3], [l, 0.0, 0.0], w, h, [0.0, 1.0, 0.0])
}

fn rel(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs()
}

// ---------------------------------------------------------------------------
// Self term
// ---------------------------------------------------------------------------

#[test]
fn reference_helpers_reproduce_known_constants() {
    // GMD of a square of side a is 0.44705·a; mean distance 0.52140·a.
    assert!((ln_gmd_rectangle(1.0, 1.0).exp() - 0.447_05).abs() < 1e-5);
    assert!((mean_distance_rectangle(1.0, 1.0) - 0.521_405).abs() < 1e-6);
    assert!((ln_gmd_rectangle(2.0, 1e-9) - (2.0_f64.ln() - 1.5)).abs() < 1e-8);
    assert!((mean_distance_rectangle(1e-9, 3.0) - 1.0).abs() < 1e-8);
    // The unit cube's self-potential ∫∫ dV dV'/r = 1.8823126443896602.
    let cube = reference_self(1.0, 1.0, 1.0) / (MU0 / (4.0 * std::f64::consts::PI));
    assert!((cube - 1.882_312_644_389_660_2).abs() < 1e-11, "{cube}");
}

#[test]
fn self_closed_form_matches_direct_integration_of_the_definition() {
    for (l, w, h) in [
        (1.0, 1.0, 1.0),
        (1.0, 0.5, 0.25),
        (4.0, 1.0, 1.0),
        (1.0, 3.0, 2.0),
        (2e-3, 1e-3, 0.5e-3),
        (5.0, 1.0, 2.0),
    ] {
        let want = reference_self(l, w, h);
        let exact = closed_form::rectangular_bar_self(l, w, h);
        assert!(exact.relative_error < 1e-11, "({l}, {w}, {h}): {exact:?}");
        assert!(
            rel(exact.value, want) < 1e-9,
            "({l}, {w}, {h}): closed form {} vs reference {want}",
            exact.value
        );
        assert!(rel(self_inductance(&bar_x(l, w, h)), want) < 1e-9);
    }
}

#[test]
fn self_term_matches_the_grover_rosa_long_conductor_formula() {
    // Truncation error of the expansion is O((w/l)⁴)/32.
    for (l, w, h, tolerance) in [
        (1.0, 1e-2, 1e-2, 1e-8),
        (1.0, 1e-2, 2e-3, 1e-8),
        (10e-3, 1e-3, 35e-6, 5e-6),
        (0.1, 1e-4, 1e-4, 1e-9),
        (1.0, 1e-5, 3e-6, 1e-9),
        (1.0, 1e-7, 1e-7, 1e-9),
        (1.0, 1e-3, 1e-7, 1e-9),
    ] {
        let got = self_inductance(&bar_x(l, w, h));
        let want = thin_bar_self(l, w, h);
        assert!(
            rel(got, want) < tolerance,
            "({l}, {w}, {h}): {got} vs {want}, {:e}",
            rel(got, want)
        );
    }
}

#[test]
fn self_term_is_continuous_across_the_method_switch() {
    // Sweeping the slenderness moves the evaluation from the closed form to
    // the quadrature; L·(1/l) must vary smoothly, without a jump at the
    // switch. Compare both methods directly wherever the closed form still
    // has digits to spare.
    let mut methods = std::collections::HashSet::new();
    for k in 0..40 {
        let w = 10f64.powf(-0.1 * k as f64);
        let f = bar_x(1.0, w, 0.5 * w);
        let exact = closed_form::rectangular_bar_self(1.0, w, 0.5 * w);
        let got = mutual_inductance_detailed(&f, &f).unwrap();
        methods.insert(got.method);
        assert_eq!(got.value, self_inductance(&f));
        if exact.relative_error < 1e-6 {
            assert!(
                rel(got.value, exact.value) < 2e-9 + exact.relative_error,
                "w = {w}: {} vs {exact:?}",
                got.value
            );
        }
    }
    assert!(methods.contains(&Method::BarClosedForm));
    assert!(methods.contains(&Method::AlignedQuadrature));
}

#[test]
fn self_term_handles_extreme_aspect_ratios() {
    // Short fat plates and tapes against the definition…
    for (l, w, h) in [(0.2, 1.0, 1.0), (1.0, 5.0, 0.2), (1.0, 1.0, 0.02)] {
        let got = self_inductance(&bar_x(l, w, h));
        assert!(rel(got, reference_self(l, w, h)) < 1e-7, "({l}, {w}, {h})");
    }
    // …and everything from a disc to a hair is finite, positive and
    // monotonically increasing in slenderness per unit length.
    let mut previous = 0.0;
    for k in -3..=7 {
        let l = 10f64.powi(k);
        let per_length = self_inductance(&bar_x(l, 1.0, 1.0)) / l;
        assert!(per_length.is_finite() && per_length > previous, "l = {l}");
        previous = per_length;
    }
}

#[test]
fn inductance_scales_linearly_with_size() {
    let build = |scale: f64| {
        let a = filament(
            [0.0; 3].map(|x: f64| x * scale),
            [3.0 * scale, 0.5 * scale, 0.0],
            0.2 * scale,
            0.1 * scale,
        );
        let b = filament(
            [0.5 * scale, 1.0 * scale, 0.7 * scale],
            [2.0 * scale, 2.5 * scale, 0.2 * scale],
            0.3 * scale,
            0.1 * scale,
        );
        (a, b)
    };
    let (a1, b1) = build(1.0);
    for scale in [1e-6, 1e-3, 40.0] {
        let (a, b) = build(scale);
        assert!(rel(self_inductance(&a), scale * self_inductance(&a1)) < 1e-9);
        let (m, m1) = (
            mutual_inductance(&a, &b).unwrap(),
            mutual_inductance(&a1, &b1).unwrap(),
        );
        assert!(rel(m, scale * m1) < 1e-9, "scale {scale}");
    }
}

// ---------------------------------------------------------------------------
// Parallel mutual term
// ---------------------------------------------------------------------------

#[test]
fn parallel_bar_closed_form_matches_numerical_integration() {
    // Equal-length parallel bars with a lateral offset (and a few with axial
    // offset and unequal sizes), against the brute-force sixfold integral.
    let cases = [
        ([1.0, 1.0], [0.2, 0.2], [0.1, 0.1], [0.0, 0.6, 0.0]),
        ([1.0, 1.0], [0.2, 0.2], [0.1, 0.1], [0.0, 0.0, 0.5]),
        ([1.0, 1.0], [0.2, 0.2], [0.1, 0.1], [0.0, 0.7, 0.4]),
        ([1.0, 0.6], [0.2, 0.1], [0.1, 0.3], [0.4, 0.8, -0.3]),
        ([1.0, 1.0], [0.2, 0.2], [0.1, 0.1], [2.5, 0.3, 0.0]),
    ];
    for (length, width, height, offset) in cases {
        let exact = closed_form::rectangular_bars(length, width, height, offset);
        assert!(exact.relative_error < 1e-8, "{exact:?}");
        let a = bar_x(length[0], width[0], height[0]);
        let start = [
            offset[0] + 0.5 * (length[0] - length[1]),
            offset[1],
            offset[2],
        ];
        let b = filament_with(
            start,
            [start[0] + length[1], start[1], start[2]],
            width[1],
            height[1],
            [0.0, 1.0, 0.0],
        );
        let want = reference_mutual(&a, &b, 4, 6);
        assert!(
            rel(exact.value, want) < 1e-7,
            "{offset:?}: closed form {} vs integration {want}",
            exact.value
        );
        assert!(rel(mutual_inductance(&a, &b).unwrap(), want) < 1e-7);
    }
}

#[test]
fn grover_parallel_filament_formula_matches_numerical_integration() {
    // Line filaments: compare with bars thin enough (w/d = 1e-3) that their
    // cross-section changes the integral by O((w/d)²).
    for (l, d) in [(1.0, 0.1), (1.0, 1.0), (0.3, 2.0)] {
        let a = bar_x(l, 1e-3 * d, 1e-3 * d);
        let b = filament_with(
            [0.0, d, 0.0],
            [l, d, 0.0],
            1e-3 * d,
            1e-3 * d,
            [0.0, 1.0, 0.0],
        );
        let formula = closed_form::parallel_filaments(l, d);
        assert!(rel(mutual_inductance(&a, &b).unwrap(), formula) < 1e-6);
        // And with a direct one-dimensional-pair quadrature of ∫∫ ds ds'/r.
        let mut sum = 0.0;
        let nodes = gauss_legendre(40);
        for panel_a in 0..8 {
            for panel_b in 0..8 {
                for &(s, ws) in &nodes {
                    for &(t, wt) in &nodes {
                        let x = (panel_a as f64 + s - panel_b as f64 - t) * l / 8.0;
                        sum += ws * wt / x.hypot(d);
                    }
                }
            }
        }
        let want = MU0 / (4.0 * std::f64::consts::PI) * sum * (l / 8.0) * (l / 8.0);
        assert!(rel(formula, want) < 1e-9, "l = {l}, d = {d}");
        // The offset formula reduces to it.
        let general = closed_form::parallel_filaments_offset(l, l, 0.0, d).unwrap();
        assert!(rel(general, formula) < 1e-13);
    }
}

#[test]
fn bundle_neighbours_agree_between_closed_form_and_quadrature() {
    // Touching filaments of one bundle, from fat to very slender. Where the
    // closed form is still accurate the two methods must agree; beyond, the
    // long-filament expansion with the exact GMD takes over as reference.
    for slenderness in [5.0, 20.0, 100.0] {
        let segment = Segment::new(
            Node::new(0.0, 0.0, 0.0),
            Node::new(slenderness, 0.0, 0.0),
            3.0,
            2.0,
            COPPER,
        );
        let filaments = discretize(&segment, 3, 2).unwrap();
        for (i, j) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)] {
            let (a, b) = (&filaments[i], &filaments[j]);
            let delta = b.center() - a.center();
            let exact = closed_form::rectangular_bars(
                [slenderness; 2],
                [1.0; 2],
                [1.0; 2],
                [0.0, delta.dot(&a.width_dir()), delta.dot(&a.height_dir())],
            );
            let got = mutual_inductance(a, b).unwrap();
            assert!(exact.relative_error < 1e-7);
            assert!(
                rel(got, exact.value) < 2e-9 + exact.relative_error,
                "slenderness {slenderness}, pair ({i}, {j})"
            );
        }
    }
    // Two 1 µm × 1 µm filaments, 10 mm long, sharing a face: the closed form
    // has no digits left, the production path must not care. Reference: the
    // long-filament expansion, using GMD/AMD of the *pair* obtained from
    // those of the rectangles by the additivity of ∫∫ over a union:
    // (2A)²·⟨f⟩_union = 2A²·⟨f⟩_self + 2A²·⟨f⟩_pair.
    let (l, w) = (10e-3, 1e-6);
    let a = bar_x(l, w, w);
    let b = filament_with([0.0, w, 0.0], [l, w, 0.0], w, w, [0.0, 1.0, 0.0]);
    let pair_mean = |union: f64, single: f64| 2.0 * union - single;
    let ln_gmd = pair_mean(ln_gmd_rectangle(2.0 * w, w), ln_gmd_rectangle(w, w));
    let amd = pair_mean(
        mean_distance_rectangle(2.0 * w, w),
        mean_distance_rectangle(w, w),
    );
    let msd = pair_mean((4.0 * w * w + w * w) / 6.0, (w * w + w * w) / 6.0);
    let want = MU0 * l / (2.0 * std::f64::consts::PI)
        * ((2.0 * l).ln() - ln_gmd - 1.0 + amd / l - msd / (4.0 * l * l));
    let got = mutual_inductance_detailed(&a, &b).unwrap();
    assert_eq!(got.method, Method::AlignedQuadrature);
    assert!(rel(got.value, want) < 1e-9, "{} vs {want}", got.value);
    let useless = closed_form::rectangular_bars([l; 2], [w; 2], [w; 2], [0.0, w, 0.0]);
    assert!(useless.relative_error > 1e-3, "{useless:?}");
}

#[test]
fn collinear_and_overlapping_bars_are_handled() {
    // Consecutive segments of one straight trace share an end face.
    let a = bar_x(1.0, 0.2, 0.1);
    let b = filament_with([1.0, 0.0, 0.0], [1.8, 0.0, 0.0], 0.2, 0.1, [0.0, 1.0, 0.0]);
    let m = mutual_inductance(&a, &b).unwrap();
    // Additivity of the volume integral: L(a ∪ b) = L(a) + L(b) + 2·M(a, b).
    let whole = self_inductance(&bar_x(1.8, 0.2, 0.1));
    let parts = self_inductance(&a) + self_inductance(&b) + 2.0 * m;
    assert!(rel(parts, whole) < 1e-9, "{parts} vs {whole}");

    // The same identity for very slender bars, where only the quadrature
    // path is usable.
    let a = bar_x(1.0, 2e-5, 1e-5);
    let b = filament_with(
        [1.0, 0.0, 0.0],
        [1.8, 0.0, 0.0],
        2e-5,
        1e-5,
        [0.0, 1.0, 0.0],
    );
    let m = mutual_inductance(&a, &b).unwrap();
    let whole = self_inductance(&bar_x(1.8, 2e-5, 1e-5));
    let parts = self_inductance(&a) + self_inductance(&b) + 2.0 * m;
    assert!(rel(parts, whole) < 1e-9, "{parts} vs {whole}");

    // A bar split lengthwise into two halves side by side: the current
    // divides by area, so L(whole) = ¼·(L₁ + L₂ + 2M).
    let halves = discretize(
        &Segment::new(
            Node::new(0.0, 0.0, 0.0),
            Node::new(1.0, 0.0, 0.0),
            0.2,
            0.1,
            COPPER,
        ),
        2,
        1,
    )
    .unwrap();
    let m = mutual_inductance(&halves[0], &halves[1]).unwrap();
    let combined = 0.25 * (self_inductance(&halves[0]) + self_inductance(&halves[1]) + 2.0 * m);
    assert!(rel(combined, self_inductance(&bar_x(1.0, 0.2, 0.1))) < 1e-9);

    // Identical and partially overlapping bars are finite and ordered:
    // touching < overlapping < identical.
    let shifted = |y: f64| filament_with([0.0, y, 0.02], [1.0, y, 0.02], 0.2, 0.1, [0.0, 1.0, 0.0]);
    let overlapping = mutual_inductance(&a_fat(), &shifted(0.05)).unwrap();
    let touching = mutual_inductance(&a_fat(), &shifted(0.2)).unwrap();
    assert!(touching < overlapping && overlapping < self_inductance(&a_fat()));
    assert_eq!(
        mutual_inductance(&a_fat(), &a_fat()).unwrap(),
        self_inductance(&a_fat())
    );
}

fn a_fat() -> Filament {
    bar_x(1.0, 0.2, 0.1)
}

#[test]
fn antiparallel_filaments_have_negative_mutual_inductance() {
    let a = bar_x(1.0, 0.1, 0.1);
    let forward = filament_with([0.0, 0.3, 0.0], [1.0, 0.3, 0.0], 0.1, 0.1, [0.0, 1.0, 0.0]);
    let backward = filament_with([1.0, 0.3, 0.0], [0.0, 0.3, 0.0], 0.1, 0.1, [0.0, 1.0, 0.0]);
    let (mf, mb) = (
        mutual_inductance(&a, &forward).unwrap(),
        mutual_inductance(&a, &backward).unwrap(),
    );
    assert!(mf > 0.0);
    assert!(rel(mb, -mf) < 1e-12);
    // A cross-section rotated by 90° is still aligned: swap width and height.
    let rotated = filament_with([0.0, 0.3, 0.0], [1.0, 0.3, 0.0], 0.1, 0.1, [0.0, 0.0, 1.0]);
    assert!(rel(mutual_inductance(&a, &rotated).unwrap(), mf) < 1e-12);
}

// ---------------------------------------------------------------------------
// General mutual term
// ---------------------------------------------------------------------------

#[test]
fn point_quadrature_matches_the_parallel_closed_form_in_the_parallel_limit() {
    // Exactly parallel, far enough apart for the point rule to be chosen.
    let a = bar_x(1.0, 0.3, 0.2);
    for offset in [[0.0, 2.5, 0.0], [0.3, 1.5, 2.0], [4.0, 0.0, 0.5]] {
        let b = filament_with(
            offset,
            [offset[0] + 1.0, offset[1], offset[2]],
            0.3,
            0.2,
            [0.0, 1.0, 0.0],
        );
        let got = mutual_inductance_detailed(&a, &b).unwrap();
        assert_eq!(got.method, Method::PointQuadrature, "{offset:?}");
        let exact = closed_form::rectangular_bars([1.0; 2], [0.3; 2], [0.2; 2], offset);
        assert!(exact.relative_error < 1e-8, "{exact:?}");
        assert!(rel(got.value, exact.value) < 1e-7, "{offset:?}");
    }
    // Tilting one bar by a vanishing angle must approach the parallel value
    // continuously (the change is first order in the angle).
    let parallel = closed_form::rectangular_bars([1.0; 2], [0.3; 2], [0.2; 2], [0.0, 2.5, 0.0]);
    for angle in [1e-2_f64, 1e-3, 1e-4] {
        let (s, c) = angle.sin_cos();
        let tilted = filament_with(
            [0.5 - 0.5 * c, 2.5 - 0.5 * s, 0.0],
            [0.5 + 0.5 * c, 2.5 + 0.5 * s, 0.0],
            0.3,
            0.2,
            [-s, c, 0.0],
        );
        let got = mutual_inductance(&a, &tilted).unwrap();
        assert!(rel(got, parallel.value) < 2.0 * angle, "angle {angle}");
    }
}

#[test]
fn general_orientation_matches_a_fine_quadrature_reference() {
    let a = filament([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.10, 0.05);
    let cases = [
        // Far: point quadrature.
        (
            filament([0.5, 2.0, 1.0], [1.5, 2.8, 0.6], 0.08, 0.04),
            Method::PointQuadrature,
        ),
        (
            filament([3.0, 1.0, -2.0], [2.0, 2.0, -1.0], 0.2, 0.1),
            Method::PointQuadrature,
        ),
        // Near, skew: sampled filaments.
        (
            filament([0.2, 0.25, 0.1], [0.9, 0.5, 0.3], 0.06, 0.03),
            Method::SampledFilaments,
        ),
        (
            filament([0.1, -0.2, 0.2], [0.8, 0.3, 0.25], 0.05, 0.05),
            Method::SampledFilaments,
        ),
        // Near, parallel, cross-section rotated by 30°: sampled filaments.
        (
            filament_with(
                [0.1, 0.0, 0.3],
                [0.9, 0.0, 0.3],
                0.08,
                0.04,
                [0.0, 0.866, 0.5],
            ),
            Method::SampledFilaments,
        ),
    ];
    for (b, method) in cases {
        let got = mutual_inductance_detailed(&a, &b).unwrap();
        assert_eq!(got.method, method);
        assert!(got.resolved);
        let want = reference_mutual(&a, &b, 6, 4);
        let tolerance = if method == Method::PointQuadrature {
            1e-7
        } else {
            2e-6
        };
        assert!(
            rel(got.value, want) < tolerance,
            "{method:?}: {} vs {want} ({:e})",
            got.value,
            rel(got.value, want)
        );
    }
}

#[test]
fn far_field_tends_to_the_dipole_limit() {
    // M → μ0/(4π) · l₁ l₂ cos ε / d.
    let a = filament([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.1, 0.1);
    for d in [1e2, 1e4, 1e6] {
        let b = filament([0.0, d, 0.0], [0.6, d, 0.8], 0.1, 0.1);
        let want = 1e-7 * 1.0 * 1.0 * 0.6 / d;
        let got = mutual_inductance(&a, &b).unwrap();
        assert!(rel(got, want) < 2.0 / d, "d = {d}: {got} vs {want}");
    }
}

#[test]
fn orthogonal_filaments_do_not_couple_even_when_touching() {
    let a = filament([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.1, 0.1);
    let corner = filament([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.1, 0.1);
    let via = filament([0.5, 0.0, -0.5], [0.5, 0.0, 0.5], 0.1, 0.1);
    for b in [corner, via] {
        let got = mutual_inductance_detailed(&a, &b).unwrap();
        assert_eq!((got.value, got.method), (0.0, Method::Orthogonal));
    }
}

#[test]
fn touching_and_overlapping_skew_bars_are_close_to_a_fine_reference() {
    // Segments of a bend overlap near their common node, and crossing bars
    // may interpenetrate: the cross-section integrand is then weakly
    // singular and the sampling order is capped — the documented
    // "unresolved" regime. The reference samples the same exact line-to-line
    // formula on a cross-section grid twice as fine in each of the four
    // dimensions (itself converged to a few 1e-5). This is a self-referential
    // sampling-convergence check, not an independent reference for this regime.
    let a = filament([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.1, 0.05);
    let cases = [
        (
            "45° bend",
            filament([1.0, 0.0, 0.0], [1.7, 0.7, 0.0], 0.1, 0.05),
        ),
        (
            "135° bend",
            filament([1.0, 0.0, 0.0], [0.3, 0.7, 0.0], 0.1, 0.05),
        ),
        (
            "10° bend",
            filament([1.0, 0.0, 0.0], [1.98, 0.17, 0.0], 0.1, 0.05),
        ),
        (
            "interpenetrating crossing",
            filament([0.2, -0.3, 0.0], [0.8, 0.3, 0.0], 0.1, 0.05),
        ),
    ];
    let (nodes_a, nodes_b) = (gauss_legendre(16), gauss_legendre(17));
    let sample = |f: &Filament, y: f64, z: f64| {
        let offset =
            f.width_dir() * ((y - 0.5) * f.width()) + f.height_dir() * ((z - 0.5) * f.height());
        let (s, e) = (f.start() + offset, f.end() + offset);
        ([s.x, s.y, s.z], [e.x, e.y, e.z])
    };
    for (name, b) in cases {
        let got = mutual_inductance_detailed(&a, &b).unwrap();
        assert_eq!(
            (got.method, got.resolved),
            (Method::SampledFilaments, false),
            "{name}"
        );
        let mut want = 0.0;
        for &(ya, wya) in &nodes_a {
            for &(za, wza) in &nodes_a {
                let (a0, a1) = sample(&a, ya, za);
                for &(yb, wyb) in &nodes_b {
                    for &(zb, wzb) in &nodes_b {
                        let (b0, b1) = sample(&b, yb, zb);
                        let line = closed_form::inclined_filaments(a0, a1, b0, b1).unwrap();
                        want += wya * wza * wyb * wzb * line;
                    }
                }
            }
        }
        assert!(got.value.is_finite());
        assert!(
            rel(got.value, want) < 1e-3,
            "{name}: {} vs {want} ({:e})",
            got.value,
            rel(got.value, want)
        );
    }
}

#[test]
fn sampled_filaments_match_the_parallel_closed_form_in_the_parallel_limit() {
    // Close, slender bars: too near for point quadrature. A cross-section
    // rotated by 45° is not aligned, which forces the sampled-filament path;
    // for a square section that rotation only matters at O((w/d)⁴).
    let a = bar_x(1.0, 0.01, 0.01);
    let exact = closed_form::rectangular_bars([1.0; 2], [0.01; 2], [0.01; 2], [0.0, 0.1, 0.0]);
    assert!(exact.relative_error < 1e-7, "{exact:?}");
    let rotated = filament_with(
        [0.0, 0.1, 0.0],
        [1.0, 0.1, 0.0],
        0.01,
        0.01,
        [0.0, 1.0, 1.0],
    );
    let got = mutual_inductance_detailed(&a, &rotated).unwrap();
    assert_eq!((got.method, got.resolved), (Method::SampledFilaments, true));
    assert!(
        rel(got.value, exact.value) < 1e-5,
        "{} vs {exact:?}",
        got.value
    );

    // Tilting by a vanishing angle switches from the parallel to the skew
    // line formula (whose common-perpendicular feet run off to infinity) and
    // must still converge to the parallel value.
    for angle in [1e-2_f64, 1e-3, 1e-4, 1e-5, 2e-6] {
        let (s, c) = angle.sin_cos();
        let tilted = filament_with(
            [0.5 - 0.5 * c, 0.1 - 0.5 * s, 0.0],
            [0.5 + 0.5 * c, 0.1 + 0.5 * s, 0.0],
            0.01,
            0.01,
            [-s, c, 0.0],
        );
        let got = mutual_inductance_detailed(&a, &tilted).unwrap();
        assert_eq!(got.method, Method::SampledFilaments, "angle {angle}");
        assert!(
            rel(got.value, exact.value) < 20.0 * angle * angle + 1e-5,
            "angle {angle}: {} vs {}",
            got.value,
            exact.value
        );
    }
}

// ---------------------------------------------------------------------------
// Batched API
// ---------------------------------------------------------------------------

/// Two consecutive segments of a trace, its return conductor, and a skew
/// stub: 2·(3×2) + 2×2 + 1 = 17 filaments.
fn fixture() -> Vec<Filament> {
    let node = Node::new;
    let mut filaments = Vec::new();
    let trace = [
        Segment::new(
            node(0.0, 0.0, 0.0),
            node(5e-3, 0.0, 0.0),
            1e-3,
            35e-6,
            COPPER,
        ),
        Segment::new(
            node(5e-3, 0.0, 0.0),
            node(10e-3, 0.0, 0.0),
            1e-3,
            35e-6,
            COPPER,
        ),
    ];
    for segment in &trace {
        filaments.extend(discretize(segment, 3, 2).unwrap());
    }
    let ret = Segment::new(
        node(10e-3, 0.0, -0.2e-3),
        node(0.0, 0.0, -0.2e-3),
        2e-3,
        35e-6,
        COPPER,
    );
    filaments.extend(discretize(&ret, 2, 2).unwrap());
    let stub = Segment::new(
        node(12e-3, 1e-3, 0.0),
        node(15e-3, 4e-3, 1e-3),
        0.5e-3,
        35e-6,
        COPPER,
    );
    filaments.extend(discretize(&stub, 1, 1).unwrap());
    filaments
}

#[test]
fn batch_matches_pairwise_evaluation_in_every_execution_mode() {
    let filaments = fixture();
    let (rows, cols) = (&filaments[..7], &filaments[3..]);
    let parallel = mutual_batch(rows, cols).unwrap();
    assert_eq!(parallel.shape(), (rows.len(), cols.len()));
    for (i, a) in rows.iter().enumerate() {
        for (j, b) in cols.iter().enumerate() {
            assert_eq!(parallel[(i, j)], mutual_inductance(a, b).unwrap());
        }
    }
    let simd = mutual_batch_with(rows, cols, Execution::Simd).unwrap();
    let scalar = mutual_batch_with(rows, cols, Execution::Scalar).unwrap();
    assert_eq!(parallel, simd);
    let worst = (&scalar - &simd).component_div(&simd).abs().max();
    assert!(worst < 1e-13, "scalar vs SIMD: {worst:e}");
    assert_eq!(mutual_batch(&[], cols).unwrap().shape(), (0, cols.len()));
    assert_eq!(mutual_batch(rows, &[]).unwrap().shape(), (rows.len(), 0));
}

/// Compare every entry and its accuracy flag with the public single-pair API.
fn assert_detailed_batch(report: &MutualBatch, rows: &[Filament], cols: &[Filament]) {
    assert_eq!(report.values.shape(), (rows.len(), cols.len()));
    let mut expected = Vec::new();
    for (i, a) in rows.iter().enumerate() {
        for (j, b) in cols.iter().enumerate() {
            let pair = mutual_inductance_detailed(a, b).unwrap();
            let value = report.values[(i, j)];
            assert!(
                (value - pair.value).abs() <= 1e-13 * pair.value.abs(),
                "value at ({i}, {j}): {value} vs {}",
                pair.value
            );
            if !pair.resolved {
                expected.push((i, j));
            }
        }
    }
    assert_eq!(report.unresolved_pairs, expected);
}

#[test]
fn detailed_batch_reports_mixed_bends_crossings_and_separated_pairs() {
    let rows = [
        filament([0.0; 3], [1.0, 0.0, 0.0], 0.1, 0.05),
        filament([0.0, 10.0, 0.0], [1.0, 10.0, 0.0], 0.1, 0.05),
    ];
    let mut cols = Vec::new();
    for degrees in [10.0_f64, 45.0, 135.0] {
        let (s, c) = degrees.to_radians().sin_cos();
        cols.push(filament([1.0, 0.0, 0.0], [1.0 + c, s, 0.0], 0.1, 0.05));
    }
    let (s, c) = 60.0_f64.to_radians().sin_cos();
    cols.push(filament(
        [0.5 - 0.5 * c, -0.5 * s, 0.0],
        [0.5 + 0.5 * c, 0.5 * s, 0.0],
        0.1,
        0.05,
    ));
    cols.push(rows[0].clone()); // Self term is resolved.
    cols.push(filament([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.1, 0.05));
    for execution in [Execution::Scalar, Execution::Simd, Execution::Parallel] {
        let report = mutual_batch_detailed_with(&rows, &cols, execution).unwrap();
        assert_detailed_batch(&report, &rows, &cols);
        assert_eq!(report.unresolved_pairs, [(0, 0), (0, 1), (0, 2), (0, 3)]);
        assert_eq!(
            report.values,
            mutual_batch_with(&rows, &cols, execution).unwrap()
        );
    }
    let report = mutual_batch_detailed(&rows, &cols).unwrap();
    assert_eq!(report.values, mutual_batch(&rows, &cols).unwrap());
    assert!(mutual_batch_detailed(&rows[1..], &cols)
        .unwrap()
        .unresolved_pairs
        .is_empty());
    // Parallel scheduling must not change the report's ordering or values.
    for threads in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(
            pool.install(|| mutual_batch_detailed(&rows, &cols).unwrap()),
            report
        );
    }
}

#[test]
fn detailed_matrix_mirrors_unresolved_entries_and_preserves_self_terms() {
    let filaments = [
        filament([0.0; 3], [10e-3, 0.0, 0.0], 1e-3, 35e-6),
        filament([10e-3, 0.0, 0.0], [17e-3, 7e-3, 0.0], 1e-3, 35e-6),
        filament([0.0, 1.0, 0.0], [10e-3, 1.0, 0.0], 1e-3, 35e-6),
    ];
    let report = partial_inductance_matrix_detailed(&filaments).unwrap();
    assert_detailed_batch(&report, &filaments, &filaments);
    assert_eq!(report.unresolved_pairs, [(0, 1), (1, 0)]);
    assert_eq!(
        report,
        mutual_batch_detailed(&filaments, &filaments).unwrap()
    );
    assert_eq!(
        report.values,
        partial_inductance_matrix(&filaments).unwrap()
    );
    for (i, filament) in filaments.iter().enumerate() {
        assert_eq!(report.values[(i, i)], self_inductance(filament));
        for j in 0..filaments.len() {
            assert_eq!(report.values[(i, j)], report.values[(j, i)]);
        }
    }
    let separated = [filaments[0].clone(), filaments[2].clone()];
    assert!(partial_inductance_matrix_detailed(&separated)
        .unwrap()
        .unresolved_pairs
        .is_empty());
}

#[test]
fn detailed_batches_preserve_empty_shapes() {
    let one = [bar_x(1.0, 0.1, 0.05)];
    for (rows, cols) in [(&[][..], &one[..]), (&one[..], &[][..]), (&[][..], &[][..])] {
        for execution in [Execution::Scalar, Execution::Simd, Execution::Parallel] {
            let report = mutual_batch_detailed_with(rows, cols, execution).unwrap();
            assert_eq!(report.values.shape(), (rows.len(), cols.len()));
            assert!(report.unresolved_pairs.is_empty());
        }
        assert_detailed_batch(&mutual_batch_detailed(rows, cols).unwrap(), rows, cols);
    }
    assert_detailed_batch(&partial_inductance_matrix_detailed(&[]).unwrap(), &[], &[]);
    assert_detailed_batch(
        &partial_inductance_matrix_detailed(&one).unwrap(),
        &one,
        &one,
    );
}

#[test]
fn detailed_batches_preserve_kernel_errors_and_pair_indices() {
    // Finite dimensions outside the kernel's validated range underflow the
    // area squared, giving a deterministic NotFinite error for the self term.
    // The ordinary bars are orthogonal to this one, so all other pairs succeed.
    let ordinary = filament([0.0; 3], [0.0, 1.0, 0.0], 0.1, 0.05);
    let extreme = bar_x(1.0, 1e-100, 1e-100);
    assert!(matches!(
        mutual_inductance_detailed(&extreme, &extreme),
        Err(KernelError::NotFinite { .. })
    ));
    let rows = [ordinary.clone(), extreme.clone()];
    let cols = [ordinary.clone(), ordinary, extreme];
    for execution in [Execution::Scalar, Execution::Simd, Execution::Parallel] {
        let error = mutual_batch_with(&rows, &cols, execution).unwrap_err();
        assert_eq!(
            error,
            KernelError::NotFinite {
                first: 1,
                second: 2
            }
        );
        assert_eq!(
            mutual_batch_detailed_with(&rows, &cols, execution).unwrap_err(),
            error
        );
    }
    assert_eq!(
        mutual_batch_detailed(&rows, &cols).unwrap_err(),
        mutual_batch(&rows, &cols).unwrap_err()
    );
    let error = partial_inductance_matrix(&rows).unwrap_err();
    assert_eq!(
        error,
        KernelError::NotFinite {
            first: 1,
            second: 1
        }
    );
    assert_eq!(
        partial_inductance_matrix_detailed(&rows).unwrap_err(),
        error
    );
}

#[test]
fn inductance_matrix_is_symmetric_and_positive_definite() {
    let filaments = fixture();
    let l: DMatrix<f64> = mutual_batch(&filaments, &filaments).unwrap();
    let n = filaments.len();
    for i in 0..n {
        assert_eq!(l[(i, i)], self_inductance(&filaments[i]));
        for j in 0..n {
            let scale = l[(i, j)].abs().max(f64::MIN_POSITIVE);
            assert!(
                (l[(i, j)] - l[(j, i)]).abs() <= 1e-12 * scale,
                "asymmetry at ({i}, {j})"
            );
        }
    }
    assert_eq!(l, partial_inductance_matrix(&filaments).unwrap());

    // The return conductor runs the other way: its coupling to the trace is
    // negative, within it positive.
    assert!(l[(0, 12)] < 0.0 && l[(0, 1)] > 0.0 && l[(12, 13)] > 0.0);

    // Positive definite: Cholesky succeeds and the smallest eigenvalue is
    // well clear of rounding noise.
    assert!(l.clone().cholesky().is_some(), "L is not positive definite");
    let eigenvalues = l.clone().symmetric_eigenvalues();
    let (min, max) = (eigenvalues.min(), eigenvalues.max());
    assert!(min > 1e-6 * max, "eigenvalues span [{min:e}, {max:e}]");

    // Energy of any current pattern is positive.
    let currents = Vector3::new(1.0, -2.0, 0.5);
    let pattern = DMatrix::from_fn(n, 1, |i, _| currents[i % 3]);
    assert!((pattern.transpose() * &l * &pattern)[(0, 0)] > 0.0);
}

// ---------------------------------------------------------------------------
// Robustness
// ---------------------------------------------------------------------------

#[test]
fn random_pairs_are_finite_symmetric_and_obey_cauchy_schwarz() {
    // The Neumann kernel is positive definite, so M² ≤ L₁·L₂ for any two
    // conductors, however they touch or overlap. Log-uniform sizes over four
    // decades, random orientations, and separations from overlapping to far.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut methods = std::collections::HashMap::new();
    for case in 0..400 {
        let mut make = |origin: [f64; 3], reach: f64| {
            let length = 10f64.powf(-2.0 + 2.0 * uniform());
            let width = length * 10f64.powf(-3.0 + 3.0 * uniform());
            let height = width * 10f64.powf(-2.0 + 2.0 * uniform());
            let start = origin.map(|c| c + reach * (uniform() - 0.5));
            let dir = Vector3::new(uniform() - 0.5, uniform() - 0.5, uniform() - 0.5).normalize();
            // Every fourth filament is axis-parallel, so that aligned and
            // orthogonal pairs occur too.
            let dir = if case % 4 == 0 { Vector3::x() } else { dir };
            let end = Vector3::from(start) + dir * length;
            filament(start, [end.x, end.y, end.z], width, height)
        };
        let a = make([0.0; 3], 0.0);
        let reach = 10f64.powf(-3.0 + 4.0 * (case % 97) as f64 / 97.0);
        let b = make([0.0; 3], reach);
        let ab = mutual_inductance_detailed(&a, &b).unwrap();
        let ba = mutual_inductance_detailed(&b, &a).unwrap();
        assert_eq!(ab, ba, "case {case}");
        assert!(ab.value.is_finite(), "case {case}");
        let bound = (self_inductance(&a) * self_inductance(&b)).sqrt();
        assert!(bound.is_finite() && bound > 0.0, "case {case}");
        assert!(
            ab.value.abs() <= bound * (1.0 + 1e-9),
            "case {case} ({:?}): |M| = {:e} > √(L₁L₂) = {bound:e}",
            ab.method,
            ab.value.abs()
        );
        *methods.entry(ab.method).or_insert(0) += 1;
    }
    // The sweep exercises every evaluation path.
    for method in [
        Method::PointQuadrature,
        Method::SampledFilaments,
        Method::AlignedQuadrature,
    ] {
        assert!(
            methods.contains_key(&method),
            "{method:?} never used: {methods:?}"
        );
    }
}

#[test]
fn method_counts_tally_the_upper_triangle() {
    // Two parallel +x bars and one +y bar (orthogonal to both). The
    // histogram must cover exactly the upper-triangle unordered pair count
    // `n·(n+1)/2`, and the method split must be honest: the two crossing
    // pairs are exactly zero, and every self term takes an aligned method.
    let bar = |a: [f64; 3], b: [f64; 3]| {
        let seg = Segment::new(
            Node::new(a[0], a[1], a[2]),
            Node::new(b[0], b[1], b[2]),
            1e-3,
            1e-3,
            COPPER,
        );
        discretize(&seg, 1, 1).unwrap().pop().unwrap()
    };
    let fils = vec![
        bar([0.0; 3], [10e-3, 0.0, 0.0]),
        bar([0.0, 5e-3, 0.0], [10e-3, 5e-3, 0.0]),
        bar([5e-3, -1e-3, 0.0], [5e-3, 1e-3, 0.0]),
    ];
    let counts = partial_inductance_method_counts(&fils).unwrap();
    let n = fils.len();
    assert_eq!(counts.total(), n * (n + 1) / 2, "{counts:?}");
    // The two crossing pairs (each +x bar vs the +y bar) are orthogonal.
    assert_eq!(counts.orthogonal, 2, "{counts:?}");
    // The 3 self terms take the Hoer-Love closed form at a 10:1:1 aspect.
    assert!(
        counts.bar_closed_form >= 3,
        "self terms should be BarClosedForm: {counts:?}"
    );
}
