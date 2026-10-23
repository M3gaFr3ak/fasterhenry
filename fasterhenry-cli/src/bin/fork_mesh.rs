//! Probe: build the fork's loop basis (mesh.rs) on a deck and print the
//! loop count plus sample rows of each family. Used to cross-check the
//! spike's ported loop basis (spike/hmat/src/loopbasis.jl) row for row.
use fasterhenry::mesh::MeshMatrix;
use std::env;

fn row_text(mesh: &MeshMatrix, i: usize) -> String {
    mesh.row(i)
        .iter()
        .map(|(b, s)| format!("{}:{:+.0}", b, s))
        .collect::<Vec<String>>()
        .join(" ")
}

fn main() -> Result<(), String> {
    let deck = env::args().nth(1).expect("usage: fork_mesh <deck> [rows]");
    let rows_to_print: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let problem = fasterhenry_cli::read_inputs(std::path::Path::new(&deck))
        .map_err(|e| format!("read {deck}: {e}"))?;
    let geometry = &problem.geometry;
    let ports = &problem.ports;
    // 1:1 decks only (.default nwinc = nhinc = 1 -> one filament per
    // segment); a cross-section-refined deck would need the real counts.
    let per_segment = vec![1usize; geometry.segment_defs().len()];
    let mesh =
        MeshMatrix::build(geometry, &per_segment, ports).map_err(|e| format!("mesh: {e}"))?;
    println!("segments          = {}", geometry.segment_defs().len());
    println!("nodes             = {}", geometry.nodes().len());
    // first few segments: their endpoint node ids + positions (to compare
    // the segment ORDER against the deck's E-line order)
    for i in 0..3.min(geometry.segment_defs().len()) {
        let seg = geometry.segment_defs()[i];
        let a = geometry.node(seg.a).unwrap();
        let b = geometry.node(seg.b).unwrap();
        println!(
            "seg {i}: N{} N{}  a=({:.6e},{:.6e},{:.6e}) b=({:.6e},{:.6e},{:.6e})",
            seg.a.0, seg.b.0, a.x, a.y, a.z, b.x, b.y, b.z
        );
    }
    println!("ports             = {}", ports.len());
    println!("internal loops    = {}", mesh.internal_count());
    println!("port loops        = {}", mesh.port_count());
    println!("total loops       = {}", mesh.loop_count());
    // Row families: bundle loops first (one per (nw*nh - 1) per segment),
    // then the conductor loops (one per non-tree segment), then the port
    // loops. Print sample rows of the bundle and conductor families.
    let bundle: usize = per_segment.iter().map(|&n| n.saturating_sub(1)).sum();
    let conductor = mesh.internal_count() - bundle;
    println!("bundle rows       = {bundle}");
    println!("conductor rows    = {conductor}");
    for i in 0..rows_to_print.min(bundle) {
        println!("bundle row {i}: {}", row_text(&mesh, i));
    }
    for i in bundle..(bundle + rows_to_print.min(conductor)) {
        println!("conductor row {i}: {}", row_text(&mesh, i));
    }
    Ok(())
}
