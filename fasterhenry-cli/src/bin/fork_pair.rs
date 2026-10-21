//! Probe: evaluate the fork kernel for specific deck pairs, printing
//! value / method / resolved. Diagnostic support for the Julia kernel port
//! (diag9-12); pair specs are 1-based indices in the first-N deck order, the
//! same order the pfft dump uses.
use fasterhenry::{geometry, NodeId, Port};
use std::env;

fn main() -> Result<(), String> {
    let deck = env::args().nth(1).expect("usage: fork_pair <deck> i:j ...");
    let problem = fasterhenry_cli::read_inputs(std::path::Path::new(&deck))
        .map_err(|e| format!("read {deck}: {e}"))?;
    let take: usize = env::var("FORK_PAIR_TAKE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let nodes: Vec<geometry::Node> = problem.geometry.nodes().to_vec();
    let mut segments: Vec<geometry::SegmentDef> = problem.geometry.segment_defs().to_vec();
    segments.truncate(take);
    let geometry =
        geometry::Geometry::from_parts(nodes, segments).expect("geometry");
    let ports = vec![Port::new(NodeId(0), NodeId(1)).named("probe")];
    let system = fasterhenry::IterativeSystem::assemble(
        &geometry,
        &ports,
        &problem.discretization,
        &fasterhenry::IterativeParams::default(),
    )
    .expect("assemble");
    let filaments: Vec<_> = system.filaments().to_vec();
    if std::env::var("FORK_PAIR_COORDS").is_ok() {
        for (i, f) in filaments.iter().take(14).enumerate() {
            let (s, e) = (f.start(), f.end());
            println!(
                "fil {} start=[{:.17e},{:.17e},{:.17e}] end=[{:.17e},{:.17e},{:.17e}] w={:.17e} h={:.17e}",
                i + 1, s.x, s.y, s.z, e.x, e.y, e.z, f.width(), f.height()
            );
        }
    }
    for spec in env::args().skip(2) {
        let (i, j) = spec.split_once(':').expect("i:j pair spec");
        let (i, j) = (i.parse::<usize>().expect("i"), j.parse::<usize>().expect("j"));
        let (a, b) = (&filaments[i - 1], &filaments[j - 1]);
        match fasterhenry::inductance::mutual_inductance_detailed(a, b) {
            Ok(m) => println!(
                "pair {i}:{j} value={:.17e} method={:?} resolved={}",
                m.value, m.method, m.resolved
            ),
            Err(e) => println!("pair {i}:{j} error={e}"),
        }
    }
    Ok(())
}
