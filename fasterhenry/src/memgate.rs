//! Runtime memory governor support: stage labels the CLI watchdog thread
//! reports when RSS crosses the configured budget, so a mid-run refusal
//! says WHICH phase was eating (the counterpart of the pre-allocation
//! census in [`crate::pfft`], which only covers the operator half).
//!
//! The budget dial is the same `FASTERHENRY_MEMORY_BUDGET_GB` env the
//! census reads: one dial, two refusal layers - census refuses
//! pre-allocation on the projected estimate, the watchdog refuses
//! mid-run on the measured RSS. Unset budget = no census refusal and
//! no watchdog, exactly like before this module existed.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Stage id: before any heavy phase (loader, dispatch).
pub const STAGE_INIT: usize = 0;
/// Stage id: pFFT operator build (projection, near-pair enumeration,
/// precorrect).
pub const STAGE_PFFT_OPERATOR: usize = 1;
/// Stage id: preconditioner construction (cubel-global pattern + LU
/// blocks).
pub const STAGE_PRECOND: usize = 2;
/// Stage id: GMRES sweep.
pub const STAGE_GMRES: usize = 3;

const STAGE_NAMES: [&str; 4] = [
    "solver initialization",
    "pfft operator build (projection/near-pair/precorrect)",
    "preconditioner construction (pattern + LU blocks)",
    "GMRES sweep",
];

static STAGE: AtomicUsize = AtomicUsize::new(STAGE_INIT);

/// Record the phase the solve is entering. Ids are this module's
/// constants; out-of-range writes are ignored (defensive - the name
/// table would be unindexable otherwise).
pub fn set_stage(stage: usize) {
    if stage < STAGE_NAMES.len() {
        STAGE.store(stage, Ordering::Relaxed);
    }
}

/// The name of the currently recorded phase.
pub fn stage_name() -> &'static str {
    let stage = STAGE.load(Ordering::Relaxed);
    STAGE_NAMES.get(stage).copied().unwrap_or("unknown phase")
}

/// Resident memory of this process in bytes, from /proc/self/status.
/// Linux-only by intent (this crate targets Linux batch runs); `None`
/// means the file is missing or unparseable, and callers must treat
/// that as "cannot measure", never as zero.
pub fn rss_bytes() -> Option<u64> {
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

#[cfg(test)]
mod tests {
    #[test]
    fn out_of_range_stage_is_ignored() {
        super::set_stage(999);
        assert_eq!(super::stage_name(), "solver initialization");
        super::set_stage(super::STAGE_GMRES);
        assert_eq!(super::stage_name(), super::STAGE_NAMES[3]);
        super::set_stage(super::STAGE_INIT);
    }
}
