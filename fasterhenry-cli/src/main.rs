//! `fasterhenry` — clean-room PEEC inductance/resistance extraction.
//!
//! Run a sweep with `fasterhenry <deck.inp | problem.json>`, which writes a
//! `Zc.mat` in the working directory the way a FastHenry run does, or with
//! the explicit `fasterhenry run <deck.inp | problem.json>`, which writes
//! one only when `--zc-mat` asks for it. See `--help`.

use fasterhenry_cli::cli::{Invocation, RunArgs};
use fasterhenry_cli::inp::ParseOptions;
use fasterhenry_cli::spice::write_spice_subckt;
use fasterhenry_cli::{read_inputs_with, run_reporting_with, solver_for};

/// Runtime memory governor: while `FASTERHENRY_MEMORY_BUDGET_GB` is set,
/// a background thread reads /proc/self/status VmRSS once a second and
/// refuses (exit 134, never an OOM-kill) when the measured RSS crosses
/// the budget, naming the phase via [`fasterhenry::memgate`]. This is
/// the mid-run half of the policy; the pre-allocation census in
/// `PfftOperator::new` is the pre-flight half. Unset budget = no
/// watchdog, exactly like before.
fn arm_memory_governor() {
    let Ok(budget) = std::env::var("FASTERHENRY_MEMORY_BUDGET_GB") else {
        return;
    };
    let budget: f64 = match budget.trim().parse::<f64>() {
        Ok(value) if value.is_finite() && value > 0.0 => value,
        _ => {
            eprintln!(
                "error: FASTERHENRY_MEMORY_BUDGET_GB must be a finite number of GB, got {budget:?}"
            );
            std::process::exit(2);
        }
    };
    std::thread::spawn(move || {
        let budget_bytes = budget * 1e9;
        let refuse_at = 0.97 * budget_bytes;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let Some(rss_bytes) = read_own_vm_rss_bytes() else {
                continue; // /proc missing or unparseable: never false-fire
            };
            if (rss_bytes as f64) >= refuse_at {
                let stage = fasterhenry::memgate::stage_name();
                eprintln!(
                    "error: runtime memory governor: RSS {:.1} GB crossed the FASTERHENRY_MEMORY_BUDGET_GB={budget:.0} GB budget during {stage}; refusing to continue (exit 134). Raise the budget or shrink the operator/precond (FASTERHENRY_PFFT_CPF / _RADIUS / _ORDER, CUBEL_RADIUS_MULT).",
                    rss_bytes as f64 / 1e9,
                );
                std::process::exit(134);
            }
        }
    });
}

/// Resident memory of this process in bytes, from /proc/self/status.
fn read_own_vm_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        return Some(kb * 1024);
    }
    None
}

fn main() -> anyhow::Result<()> {
    arm_memory_governor();
    let invocation = Invocation::parse();
    // The one difference between the two forms: a bare invocation always
    // writes a Zc.mat, `run` only when asked.
    let zc_mat = invocation.zc_mat().map(std::path::Path::to_path_buf);
    let RunArgs {
        input,
        freq,
        zc_mat: _,
        json,
        spice,
        spice_freq,
        solver,
        fasthenry_compat,
    } = invocation.into_args();

    let options = ParseOptions { fasthenry_compat };
    let wall = std::time::Instant::now();
    let problem = read_inputs_with(&input, options).map_err(|m| anyhow::anyhow!("{m}"))?;
    eprintln!(
        "load: deck problem built, wall {:.1} s at this point",
        wall.elapsed().as_secs_f64()
    );
    let override_frequencies = match &freq {
        Some(values) => Some(decade_frequencies(values[0], values[1], values[2])?),
        None => None,
    };
    let choice = solver.into();
    // Say so when the run leaves the default dense path: the two
    // paths differ in their approximations, so which one produced
    // the JSON on stdout is worth a line on stderr.
    if solver_for(&problem, choice).map_err(|m| anyhow::anyhow!("{m}"))?
        == fasterhenry::Solver::Iterative
    {
        eprintln!("solver: iterative (matrix-free precorrected-FFT + GMRES)");
    }
    eprintln!(
        "load: solve dispatch reached, wall {:.1} s at this point",
        wall.elapsed().as_secs_f64()
    );
    let (result, warnings) = run_reporting_with(&problem, override_frequencies, choice)
        .map_err(|m| anyhow::anyhow!("{m}"))?;
    // Truncation is a physical approximation; say so on stderr, so it
    // never hides inside the JSON on stdout.
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
    let serialized = serde_json::to_string_pretty(&result)?;

    match json {
        Some(path) => std::fs::write(&path, serialized)?,
        None => println!("{serialized}"),
    }
    if let Some(path) = &zc_mat {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        fasterhenry_cli::mat::write_zc_mat(&mut writer, &result)
            .map_err(|error| anyhow::anyhow!("cannot write {}: {error}", path.display()))?;
    }
    if let Some(path) = &spice {
        let index = match spice_freq {
            Some(hz) => result
                .frequencies_hz
                .iter()
                .position(|&f| f == hz)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "--spice-freq {hz:e} Hz is not one of the sweep's frequencies ({})",
                        result
                            .frequencies_hz
                            .iter()
                            .map(|f| format!("{f:e}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?,
            None => result.frequencies_hz.len() - 1,
        };
        let name = input
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("fasterhenry")
            .replace('.', "_");
        std::fs::write(path, write_spice_subckt(&result, index, &name))?;
    }
    Ok(())
}

/// The `--freq` override; same decade sampling as `.freq`.
fn decade_frequencies(fmin: f64, fmax: f64, ndec: f64) -> anyhow::Result<Vec<f64>> {
    if !(fmin.is_finite() && fmin >= 0.0 && fmax.is_finite() && fmax >= 0.0) {
        anyhow::bail!("--freq values must be finite and >= 0");
    }
    if fmax < fmin {
        anyhow::bail!("--freq FMAX_HZ is below FMIN_HZ");
    }
    if fmin == fmax {
        return Ok(vec![fmin]);
    }
    if fmin == 0.0 {
        anyhow::bail!("--freq 0 is only allowed as the single-frequency case");
    }
    if !(ndec >= 1.0 && ndec.fract() == 0.0) {
        anyhow::bail!("--freq NDEC must be a whole number >= 1");
    }
    let ndec = ndec as usize;
    let decades = fmax.log10() - fmin.log10();
    let count = (decades * ndec as f64 + 1.0 + 1e-9).floor() as usize;
    Ok((0..count)
        .map(|k| {
            let (quotient, remainder) = (k / ndec, k % ndec);
            if remainder == 0 {
                fmin * 10f64.powi(quotient as i32)
            } else {
                fmin * 10f64.powf(k as f64 / ndec as f64)
            }
        })
        .collect())
}
