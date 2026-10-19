//! Acceptance for the two-level layer precond (`FASTERHENRY_PRECOND
//! =two-level`, DECISIONS D17 first rung): the iterative answer must
//! match the dense path on a deck whose segments span two copper
//! layers, i.e. one where the cross-layer CSR actually carries entries
//! and the block Gauss-Seidel applies them.
//!
//! This file is its OWN test binary on purpose: `FASTERHENRY_PRECOND`
//! is process-wide env state, and the other solve tests assert on the
//! DEFAULT precond. Keeping the env flip here (with set/remove) leaves
//! them untouched.

use std::f64::consts::TAU;

use fasterhenry::{
    Discretization, Geometry, IterativeParams, IterativeSystem, MeshSystem, Node, NodeId, Port,
    SegmentDef, Subdivision,
};

const COPPER: f64 = 5.8e7;

fn geometry(points: &[[f64; 3]], segments: &[(usize, usize, f64, f64)]) -> Geometry {
    let mut geometry = Geometry::new();
    for &point in points {
        geometry.add_node(Node::from(point)).unwrap();
    }
    for &(a, b, width, height) in segments {
        geometry
            .add_segment(SegmentDef::new(NodeId(a), NodeId(b), width, height, COPPER))
            .unwrap();
    }
    geometry
}

fn port(positive: usize, negative: usize) -> Port {
    Port::new(NodeId(positive), NodeId(negative))
}

/// The dense-path `Z` in textbook form (same construction as
/// tests/solve.rs): full `M (R + j omega L) M^T`, one port driven at a
/// time, nalgebra LU, invert the reduced admittance.

fn two_layer_structure() -> (Geometry, Vec<Port>, Discretization) {
    // Trace A at z = 0, its return at z = 0 (same layer), a second
    // trace pair 0.3 mm above (cross-layer couplings deck-wide), plus
    // a floating bar on the lower layer.
    let geometry = geometry(
        &[
            [0.0, 0.0, 0.0],
            [4e-3, 0.0, 0.0],
            [4e-3, 2e-3, 0.0],
            [0.5e-3, 0.0, 0.3e-3],
            [3.5e-3, 0.0, 0.3e-3],
            [1e-3, 1e-3, 0.0],
            [3e-3, 1e-3, 0.0],
        ],
        &[
            (0, 1, 0.4e-3, 0.1e-3),
            (1, 2, 0.4e-3, 0.1e-3),
            (3, 4, 0.2e-3, 0.05e-3),
            (5, 6, 0.2e-3, 0.1e-3),
        ],
    );
    let ports = vec![port(0, 2).named("A"), port(4, 3).named("B")];
    let mut subdivisions = vec![Subdivision::new(2, 2); 4];
    subdivisions[0] = Subdivision::new(4, 3);
    subdivisions[1] = Subdivision::new(3, 3);
    subdivisions[2] = Subdivision::new(3, 1);
    (geometry, ports, Discretization::PerSegment(subdivisions))
}

#[test]
fn two_level_precond_matches_the_dense_path() {
    std::env::set_var("FASTERHENRY_PRECOND", "two-level");
    std::env::set_var("FASTERHENRY_TWOL_RADIUS_MULT", "3.0");
    let (geometry, ports, discretization) = two_layer_structure();
    let dense = MeshSystem::assemble(&geometry, &ports, &discretization).unwrap();
    let iterative = IterativeSystem::assemble(
        &geometry,
        &ports,
        &discretization,
        &IterativeParams::default(),
    )
    .unwrap();
    assert_eq!(iterative.counts(), dense.counts());
    for frequency in [1e3, 1e6, 1e9] {
        let expected = dense.impedance(frequency).unwrap();
        let solution = iterative.solve(frequency).unwrap();
        let difference = (solution.impedance.clone() - &expected).norm() / expected.norm();
        let iterations: Vec<usize> = solution.gmres.iter().map(|o| o.iterations).collect();
        println!(
            "two-level at {frequency:e} Hz: difference {difference:e}, GMRES iterations {iterations:?}"
        );
        assert!(
            difference < 1e-4,
            "two-level at {frequency} Hz: iterative and dense differ by {difference:e}"
        );
        for (port, outcome) in solution.gmres.iter().enumerate() {
            assert!(outcome.converged, "two-level port {port}: {outcome:?}");
        }
    }
    // DC: real matrix, the two-level blocks factor over the resistive
    // network.
    let expected = dense.impedance(0.0).unwrap();
    let solution = iterative.solve(0.0).unwrap();
    let difference = (solution.impedance.clone() - &expected).norm() / expected.norm();
    println!("two-level at DC: difference {difference:e}");
    assert!(
        difference < 1e-4,
        "two-level at DC differs by {difference:e}"
    );
    std::env::remove_var("FASTERHENRY_PRECOND");
    std::env::remove_var("FASTERHENRY_TWOL_RADIUS_MULT");
}
