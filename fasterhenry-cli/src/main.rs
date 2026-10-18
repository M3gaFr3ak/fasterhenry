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

fn main() -> anyhow::Result<()> {
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
