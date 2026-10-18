//! The `fasterhenry` command-line front end: FastHenry-format `.inp` decks
//! (the public file format) or JSON problem documents in, JSON out.
//!
//! The binary logic lives in this library so the tests can drive it without
//! spawning a process; `src/main.rs` is a thin wrapper.

pub mod cli;
pub mod inp;
pub mod mat;
pub mod problem;
pub mod spice;

pub use problem::Problem;

use std::path::Path;

use fasterhenry::{
    IterativeParams, IterativeSystem, MeshSystem, Solver, SolverChoice, SweepResult,
};

/// Reads a problem from a file: `.inp`/`.fh` decks by extension, anything
/// else as a JSON problem document. Decks are read with the default
/// [`inp::ParseOptions`]; see [`read_inputs_with`].
pub fn read_inputs(path: &Path) -> Result<Problem, String> {
    read_inputs_with(path, inp::ParseOptions::default())
}

/// [`read_inputs`], reading a deck under `options` (for example
/// [`inp::ParseOptions::fasthenry_compat`]). JSON problem documents ignore
/// `options`.
pub fn read_inputs_with(path: &Path, options: inp::ParseOptions) -> Result<Problem, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("inp") | Some("fh") => {
            let wall = std::time::Instant::now();
            let deck = inp::parse_with_options(&text, options).map_err(|e| e.to_string())?;
            eprintln!(
                "deck: parse complete, wall {:.1} s",
                wall.elapsed().as_secs_f64()
            );
            let problem = Problem::from(deck);
            eprintln!(
                "deck: problem built, wall {:.1} s",
                wall.elapsed().as_secs_f64()
            );
            Ok(problem)
        }
        _ => serde_json::from_str(&text)
            .map_err(|error| format!("invalid problem document: {error}")),
    }
}

/// Runs the sweep, discarding any coupling-truncation warnings. See
/// [`run_reporting`] to receive them.
pub fn run(problem: &Problem, frequency_override: Option<Vec<f64>>) -> Result<SweepResult, String> {
    run_reporting(problem, frequency_override).map(|(result, _)| result)
}

/// The solve path `choice` takes on `problem`: its filament count against
/// [`fasterhenry::DENSE_PATH_MAX_FILAMENTS`] when the choice is
/// [`SolverChoice::Auto`], the forced path otherwise.
///
/// A [`fasterhenry::Coupling`] that truncates something is a dense-path
/// feature — the matrix-free operator couples every pair — so `Auto` stays
/// dense for such a problem whatever its size, and an explicit
/// [`SolverChoice::Iterative`] is an error rather than a silently different
/// physical approximation.
pub fn solver_for(problem: &Problem, choice: SolverChoice) -> Result<Solver, String> {
    let filaments = problem
        .discretization
        .filament_count(&problem.geometry)
        .map_err(|error| error.to_string())?;
    let solver = choice.resolve(filaments);
    if problem.coupling.is_all_pairs() {
        return Ok(solver);
    }
    match choice {
        SolverChoice::Iterative => Err(
            "--solver iterative cannot honour a .couples truncation: the matrix-free operator \
             couples every pair of filaments. Drop the truncation or use --solver dense."
                .to_string(),
        ),
        // Truncation is the user's own approximation; keep it.
        SolverChoice::Auto | SolverChoice::Dense => Ok(Solver::Dense),
    }
}

/// Runs the sweep and returns it together with the assembly's
/// coupling-truncation warnings — the group pairs whose mutual inductance
/// `.couples` dropped even though they are closer than one extent apart, in
/// the order [`fasterhenry::MeshSystem::truncation_warnings`] reports them.
/// The list is empty unless the problem truncates something.
///
/// `frequency_override`, when given, replaces the problem's frequencies (the
/// CLI's `--freq fmin fmax ndec`).
///
/// The solve path is [`SolverChoice::Auto`]'s; see [`run_reporting_with`] to
/// force one.
pub fn run_reporting(
    problem: &Problem,
    frequency_override: Option<Vec<f64>>,
) -> Result<(SweepResult, Vec<String>), String> {
    run_reporting_with(problem, frequency_override, SolverChoice::Auto)
}

/// [`run_reporting`] on a chosen solve path (the CLI's `--solver`).
///
/// [`SolverChoice::Auto`] keeps the dense path
/// ([`MeshSystem`]) at or below [`fasterhenry::DENSE_PATH_MAX_FILAMENTS`]
/// filaments and takes the matrix-free pFFT + GMRES path
/// ([`IterativeSystem`]) above it — see [`solver_for`]. The iterative path
/// reports no coupling-truncation warnings because it does not truncate:
/// its warning list is always empty.
pub fn run_reporting_with(
    problem: &Problem,
    frequency_override: Option<Vec<f64>>,
    choice: SolverChoice,
) -> Result<(SweepResult, Vec<String>), String> {
    let frequencies = frequency_override.unwrap_or_else(|| problem.frequencies_hz.clone());
    if frequencies.is_empty() {
        return Err(
            "no frequencies: give .freq in the deck, frequencies_hz in the document, or --freq"
                .to_string(),
        );
    }
    if problem.ports.is_empty() {
        return Err("no ports: give .external in the deck or ports in the document".to_string());
    }
    if solver_for(problem, choice)? == Solver::Iterative {
        let system = IterativeSystem::assemble(
            &problem.geometry,
            &problem.ports,
            &problem.discretization,
            &IterativeParams::default(),
        )
        .map_err(|error| error.to_string())?;
        let result = system
            .sweep(&frequencies)
            .map_err(|error| error.to_string())?;
        return Ok((result, Vec::new()));
    }
    let system = MeshSystem::assemble_with_coupling(
        &problem.geometry,
        &problem.ports,
        &problem.discretization,
        &problem.coupling,
    )
    .map_err(|error| error.to_string())?;
    let warnings = system
        .truncation_warnings()
        .iter()
        .map(ToString::to_string)
        .collect();
    let result = system
        .sweep(&frequencies)
        .map_err(|error| error.to_string())?;
    Ok((result, warnings))
}

#[cfg(test)]
mod tests {
    #[test]
    fn cli_definition_is_valid() {
        use clap::CommandFactory;
        crate::cli::Cli::command().debug_assert();
        crate::cli::BareCli::command().debug_assert();
    }
}
