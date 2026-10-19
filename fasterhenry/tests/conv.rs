//! Acceptance for the v2 conv engine (`FASTERHENRY_PRECOND=conv`,
//! slice 2b increment 2): the iterative answer must match the dense
//! path on a deck of three pour-net planes plus one via barrel.
//! Correctness contract: right-preconditioned GMRES monitors the TRUE
//! residual, so the conv precond can only cost time or fail loudly -
//! never answer wrong. What this test pins is the plumbing (scatter to
//! `LayerPairOperator`/`ConvPrecond`, Richardson smoother, T lift)
//! plus convergence; iteration counts (precond quality) are logged,
//! not asserted - the tab blocks compose explicitly since increment 3
//! (0e27478; the parity tripwire below across all three
//! FASTERHENRY_CONV_TABS compositions) while the cross-CSR spine is
//! still a reserved slot (increment 2d).
//! **[corrected 2026-10-20, conv-inc3 audit F1: the tab half was
//! stale, pre-0e27478 wording]**
//!
//! Deck shape matters: the loop space must hold PLANE-scale cycles
//! (junctions between strokes), or the residuals collapse into
//! per-cell self-cancellations the shared raster cannot see - the
//! smoother then reads zero and the precond is vacuous (the first
//! all-parallel-bars deck rejected itself this way, 2026-10-20).
//!
//! This file is its OWN test binary on purpose: `FASTERHENRY_PRECOND`
//! is process-wide env state, and the other solve tests assert on the
//! DEFAULT precond.

use fasterhenry::{
    Discretization, Geometry, IterativeParams, IterativeSystem, MeshSystem, Node, NodeId, Port,
    SegmentDef, Subdivision,
};

const COPPER: f64 = 5.8e7;

fn port(positive: usize, negative: usize) -> Port {
    Port::new(NodeId(positive), NodeId(negative))
}

/// One plane: a ladder net - two windows formed by three horizontal
/// rails (y = 1, 2, 3 mm over a 5 mm span) and two vertical
/// connectors that JOINT at the mid rail's ends (T junctions: the
/// connector strokes are split at y = 2 mm), so the loop basis holds
/// two plane-scale cycles per plane. Returns the mid-rail node ids.
fn add_plane(geometry: &mut Geometry, z: f64, base: usize) -> (usize, usize) {
    for point in [
        [0.0, 1.0e-3, z],
        [5.0e-3, 1.0e-3, z],
        [0.0, 2.0e-3, z],
        [5.0e-3, 2.0e-3, z],
        [0.0, 3.0e-3, z],
        [5.0e-3, 3.0e-3, z],
    ] {
        geometry.add_node(Node::from(point)).unwrap();
    }
    for &(a, b) in &[
        (base, base + 1),
        (base + 2, base + 3),
        (base + 4, base + 5),
        (base, base + 2),
        (base + 2, base + 4),
        (base + 1, base + 3),
        (base + 3, base + 5),
    ] {
        geometry
            .add_segment(SegmentDef::new(
                NodeId(a),
                NodeId(b),
                0.2e-3,
                0.1e-3,
                COPPER,
            ))
            .unwrap();
    }
    (base + 2, base + 3)
}

fn three_layer_structure() -> (Geometry, Vec<Port>, Discretization) {
    let mut geometry = Geometry::new();
    let (a_pos, a_neg) = add_plane(&mut geometry, 0.0, 0);
    let (b_pos, b_neg) = add_plane(&mut geometry, 0.3e-3, 6);
    let (_c_pos, _c_neg) = add_plane(&mut geometry, 0.6e-3, 12);
    // the via barrel: plane 0's mid-rail right end up to plane 2's,
    // sharing the junction with plane 0/2 mid rails (nodes 3 -> 15)
    geometry
        .add_segment(SegmentDef::new(
            NodeId(3),
            NodeId(15),
            0.2e-3,
            0.1e-3,
            COPPER,
        ))
        .unwrap();
    let ports = vec![port(a_pos, a_neg).named("A"), port(b_pos, b_neg).named("B")];
    let subdivisions = vec![Subdivision::new(1, 1); 22];
    (geometry, ports, Discretization::PerSegment(subdivisions))
}

#[test]
fn conv_precond_matches_the_dense_path() {
    std::env::set_var("FASTERHENRY_PRECOND", "conv");
    let (geometry, ports, discretization) = three_layer_structure();
    let dense = MeshSystem::assemble(&geometry, &ports, &discretization).unwrap();
    let iterative = IterativeSystem::assemble(
        &geometry,
        &ports,
        &discretization,
        &IterativeParams::default(),
    )
    .unwrap();
    assert_eq!(iterative.counts(), dense.counts());
    for frequency in [1.0e3, 5.0e6] {
        let expected = dense.impedance(frequency).unwrap();
        let solution = iterative.solve(frequency).unwrap();
        let difference = (solution.impedance.clone() - &expected).norm() / expected.norm();
        let iterations: Vec<usize> = solution.gmres.iter().map(|o| o.iterations).collect();
        println!(
            "conv at {frequency:e} Hz: difference {difference:e}, GMRES iterations {iterations:?}"
        );
        assert!(
            difference < 1.0e-4,
            "conv at {frequency} Hz: iterative and dense differ by {difference:e}"
        );
        for (port_i, outcome) in solution.gmres.iter().enumerate() {
            assert!(outcome.converged, "conv port {port_i}: {outcome:?}");
        }
    }
    // DC: the near model collapses to the resistive network - the
    // smoother kernel's diagonal push (mean R) alone carries it, and
    // the cross-layer inductive kernels are structurally zero.
    let expected = dense.impedance(0.0).unwrap();
    let solution = iterative.solve(0.0).unwrap();
    let difference = (solution.impedance.clone() - &expected).norm() / expected.norm();
    println!("conv at DC: difference {difference:e}");
    assert!(difference < 1.0e-4, "conv at DC differs by {difference:e}");
    std::env::remove_var("FASTERHENRY_PRECOND");
}

#[test]
fn tab_compositions_never_change_the_answer() {
    // The tab knob may change precond quality, never the answer: each
    // composition's reference is the same dense path (precision at
    // 1 kHz + convergence at 5 MHz), the two frequency-less knob
    // faces only enter the near model, and the GMRES referee sees
    // the true operator.
    for mode in ["both", "matvec", "off"] {
        std::env::set_var("FASTERHENRY_PRECOND", "conv");
        std::env::set_var("FASTERHENRY_CONV_TABS", mode);
        let (geometry, ports, discretization) = three_layer_structure();
        let dense = MeshSystem::assemble(&geometry, &ports, &discretization).unwrap();
        let iterative = IterativeSystem::assemble(
            &geometry,
            &ports,
            &discretization,
            &IterativeParams::default(),
        )
        .unwrap();
        let expected = dense.impedance(1.0e3).unwrap();
        let solution = iterative.solve(1.0e3).unwrap();
        let difference = (solution.impedance.clone() - &expected).norm() / expected.norm();
        println!("conv tabs={mode} at 1e3 Hz: difference {difference:e}");
        assert!(
            difference < 1.0e-4,
            "conv {mode} at 1e3 Hz: iterative and dense differ by {difference:e}"
        );
        for (port_i, outcome) in solution.gmres.iter().enumerate() {
            assert!(outcome.converged, "conv {mode} port {port_i}: {outcome:?}");
        }
    }
    std::env::remove_var("FASTERHENRY_CONV_TABS");
    std::env::remove_var("FASTERHENRY_PRECOND");
}
