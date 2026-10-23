//! Cross-language parity probe (spike slice 1b): apply the engine's
//! conv-matvec operator (the equation) at DC to a few unit vectors and
//! dump the answers as JSON for the Julia spike to compare.
//!
//! Usage: pfft_dump <deck.inp> <out.json>
//!
//! Knobs (same names the CLI uses, re-parsed here): FASTERHENRY_PFFT_CPF,
//! FASTERHENRY_PFFT_ORDER, FASTERHENRY_PFFT_RADIUS, FASTERHENRY_CONV_*.
//! PFFT_DUMP_TAKE truncates the deck subset (satellite: the truncated
//! deck still routes through the full assembly; the first-N filaments of
//! the discretized geometry carry the rows that are dumped).

use fasterhenry::conv::{MatvecCore, TabComposition};
use fasterhenry::pfft::PfftOperator;
use fasterhenry::{GridSpacing, IterativeParams};
use serde::Serialize;

#[derive(Serialize)]
struct FilJson {
    start: [f64; 3],
    end: [f64; 3],
    width: f64,
    height: f64,
}

#[derive(Serialize)]
struct VecJson {
    name: String,
    index: usize,
    y: Vec<f64>,
}

#[derive(Serialize, Default)]
struct Counts {
    #[serde(rename = "orthogonal")]
    orthogonal: usize,
    #[serde(rename = "bar_closed_form")]
    bar_closed_form: usize,
    #[serde(rename = "aligned_quadrature")]
    aligned_quadrature: usize,
    #[serde(rename = "point_quadrature")]
    point_quadrature: usize,
    #[serde(rename = "sampled_filaments")]
    sampled_filaments: usize,
}

#[derive(Serialize, Default)]
struct Dump {
    n: usize,
    filaments_head: Vec<FilJson>,
    filaments_tail: Vec<FilJson>,
    vectors: Vec<VecJson>,
    /// Fork-native dense inductance matrix (row-major, n x n), SI henries:
    /// the kernel ground truth for the spike's Julia port. Erroring pairs
    /// get their MATLAB-safe NaN marker.
    dense: Vec<f64>,
    /// Dispatch-path distribution over the same pairs.
    methods: Counts,
}

fn env_f(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

fn env_u(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

fn main() -> Result<(), String> {
    let deck_path = std::env::args()
        .nth(1)
        .expect("usage: pfft_dump <deck> <out.json>");
    let out_path = std::env::args()
        .nth(2)
        .expect("usage: pfft_dump <deck> <out.json>");
    let problem = fasterhenry_cli::read_inputs(std::path::Path::new(&deck_path))
        .map_err(|e| format!("read {deck_path}: {e}"))?;
    let params = IterativeParams::default();
    // Truncate to the first PFFT_DUMP_TAKE segments (deck_easy discretizes
    // 1:1: nwinc = nhinc = 1, so filament order is deck order). Ports are
    // dropped: the sweep is never run, only the filament rows are needed.
    let take = env_u("PFFT_DUMP_TAKE").unwrap_or(usize::MAX);
    let nodes: Vec<fasterhenry::geometry::Node> = problem.geometry.nodes().to_vec();
    let mut segments: Vec<fasterhenry::geometry::SegmentDef> =
        problem.geometry.segment_defs().to_vec();
    segments.truncate(take);
    eprintln!(
        "pfft_dump: segments taken {}/{}",
        segments.len(),
        nodes.len()
    );
    let geometry =
        fasterhenry::geometry::Geometry::from_parts(nodes, segments).map_err(|e| e.to_string())?;
    // assemble refuses an empty port list; the sweep is never run, so a
    // stand-in port on the first segment's node pair is enough. Its column
    // is computed and then ignored.
    let ports =
        vec![fasterhenry::Port::new(fasterhenry::NodeId(0), fasterhenry::NodeId(1)).named("probe")];
    let system =
        fasterhenry::IterativeSystem::assemble(&geometry, &ports, &problem.discretization, &params)
            .map_err(|e| e.to_string())?;
    let filaments: Vec<_> = system.filaments().to_vec();
    let n = filaments.len();
    eprintln!("pfft_dump: n={n}");

    let mut pfft_params = IterativeParams::default().pfft;
    if let Some(cpf) = env_f("FASTERHENRY_PFFT_CPF") {
        pfft_params.grid_spacing = GridSpacing::Auto {
            cells_per_filament: cpf,
        };
    }
    if let Some(order) = env_u("FASTERHENRY_PFFT_ORDER") {
        pfft_params.interpolation_order = order;
    }
    if let Some(radius) = env_f("FASTERHENRY_PFFT_RADIUS") {
        pfft_params.near_field_radius = radius;
    }
    // The conv-matvec posture: far-only operator, near band owned by the
    // conv model (kernels + tabs + spine) exactly as the equation builds.
    pfft_params.precorrect_near = false;
    let far = PfftOperator::new(&filaments, &pfft_params).map_err(|e| e.to_string())?;
    let conv_tabs = match std::env::var("FASTERHENRY_CONV_TABS").ok().as_deref() {
        Some("matvec") => TabComposition::Matvec,
        Some("off") => TabComposition::Off,
        _ => TabComposition::Both,
    };
    let conv_spine = match std::env::var("FASTERHENRY_CONV_SPINE").ok().as_deref() {
        Some("matvec") => TabComposition::Matvec,
        Some("off") => TabComposition::Off,
        _ => TabComposition::Both,
    };
    let tab_bin_m = fasterhenry::census::derived_radius(&filaments);
    let core = MatvecCore::build(&far, &filaments, conv_tabs, tab_bin_m, true, conv_spine)
        .map_err(|e| e.to_string())?;

    let fil_json = |f: &fasterhenry::Filament| FilJson {
        start: f.start().into(),
        end: f.end().into(),
        width: f.width(),
        height: f.height(),
    };
    let heads = 2;
    let dump_y = |index: usize, name: String, core: &MatvecCore, far: &PfftOperator| -> VecJson {
        let mut x = vec![0.0; n];
        x[index] = 1.0;
        // The engine's composite at DC on branch currents: far grid half
        // plus the near half's difference-kernel/tab/spine flux (the same
        // addition the wire's matvec hop makes; R and the mesh gather are
        // the caller's basis side, not the operator).
        let mut y = far.apply(&x);
        let mut flux = vec![0.0; n];
        core.apply(&x, &mut flux);
        for (yi, fi) in y.iter_mut().zip(&flux) {
            *yi += fi;
        }
        VecJson { name, index, y }
    };
    let indices = [0, n / 4, n / 2, 3 * n / 4, n - 1];
    let vectors: Vec<VecJson> = indices
        .iter()
        .map(|&i| dump_y(i, format!("e{i}"), &core, &far))
        .collect();
    let mut dense = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            dense[i * n + j] = match fasterhenry::mutual_inductance(&filaments[i], &filaments[j]) {
                Ok(value) => value,
                Err(_) => f64::NAN,
            };
        }
    }
    let methods = match fasterhenry::partial_inductance_method_counts(&filaments) {
        Ok(c) => Counts {
            orthogonal: c.orthogonal,
            bar_closed_form: c.bar_closed_form,
            aligned_quadrature: c.aligned_quadrature,
            point_quadrature: c.point_quadrature,
            sampled_filaments: c.sampled_filaments,
        },
        Err(_) => Counts::default(),
    };
    let dump = Dump {
        n,
        filaments_head: filaments.iter().take(heads).map(fil_json).collect(),
        filaments_tail: filaments
            .iter()
            .rev()
            .take(heads)
            .rev()
            .map(fil_json)
            .collect(),
        vectors,
        dense,
        methods,
    };
    let file = std::fs::File::create(&out_path).map_err(|e| e.to_string())?;
    serde_json::to_writer(&file, &dump).map_err(|e| e.to_string())?;
    eprintln!("pfft_dump: wrote {out_path}");
    Ok(())
}
