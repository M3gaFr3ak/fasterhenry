//! Interface census: a cheap loader-side count of what the two-level
//! precond's level 2 will have to represent, WITHOUT building any
//! operator or factorization:
//! - distinct copper layers (z-quantized) and per-layer branch/row
//!   populations,
//! - rows whose branches span layers (via columns): the spanning-row
//!   population the tagging rule assigns to their first branch's layer
//!   (audit fold 2026-10-19, DECISIONS design notes),
//! - per-layer-pair near pair counts at an explicit census radius,
//!   STREAMED as counters only (a materialized pair list at these deck
//!   sizes is an OOM: the instrumented ekart runs measured directed
//!   pairs in the 1e9 range),
//! - estimated cross-layer CSC entries from the row expansion (each
//!   branch pair expands to `|rows(branch)| x |rows(partner)|` pattern
//!   entries - the same expansion the CUBEL global assembler pays),
//! - the distinct internal rows touched by any cross-layer edge: the
//!   interface row count that would feed a dense interface Schur (the
//!   v1 gate in DECISIONS D17: viable only if this stays small).
//!
//! The census radius is an explicit input (env
//! `FASTERHENRY_CENSUS_RADIUS_M` in the CLI; `derived_radius` supplies
//! a labeled first guess, not the precond radius by construction).

use crate::filament::Filament;
use crate::inductance::neumann::Bar;
use crate::mesh::MeshMatrix;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Result of the interface census.
#[derive(Debug, Clone, Default)]
pub struct Census {
    /// Distinct copper layers detected (z-quantized).
    pub layers: usize,
    /// Branches per layer, layers ascending in z.
    pub branches_per_layer: Vec<usize>,
    /// Internal rows per layer; each row is tagged with its first
    /// branch's layer.
    pub rows_per_layer: Vec<usize>,
    /// Internal rows whose branches touch more than one layer.
    pub spanning_rows: usize,
    /// Upper-triangle branch near pairs, in-layer (same layer).
    pub branch_pairs_in_layer_upper: usize,
    /// Upper-triangle branch near pairs, cross-layer.
    pub branch_cross_pairs_upper: usize,
    /// Estimated cross-layer CSC entries from the row expansion,
    /// counted in BOTH directions (the upper expansion sum x 2 mirrors
    /// the pattern's upper/mirror split).
    pub csc_cross_entries: usize,
    /// Distinct internal rows touched by any cross-layer edge - the
    /// v1 gate's dense-Schur viability number.
    pub interface_rows: usize,
}

/// First-guess census radius when the env is unset:
/// `3 x cbrt(V_board / n)`, labeled as a guess in the report (it uses
/// the deck's natural pitch as the unit; it is NOT the precond's
/// radius, by construction).
pub fn derived_radius(filaments: &[Filament]) -> f64 {
    let n = filaments.len().max(1);
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for f in filaments {
        let bar = Bar::new(f);
        for k in 0..3 {
            let c = bar.centre[k];
            let half = bar.half[k];
            lo[k] = lo[k].min(c - half);
            hi[k] = hi[k].max(c + half);
        }
    }
    let volume: f64 = lo
        .iter()
        .zip(&hi)
        .map(|(&a, &b)| (b - a).max(0.0))
        .product();
    3.0 * (volume / n as f64).cbrt()
}

/// Tags every branch with a copper layer: z centres sorted, clustered
/// with a tolerance of `1e-6 x` the z extent (numerically co-located
/// levels merge; genuinely distinct planes stay distinct). Equal
/// values always share a cluster, so duplicates resolve through their
/// first sorted occurrence (rank = count of strictly smaller values).
pub(crate) fn branch_layers(filaments: &[Filament]) -> Vec<usize> {
    let zs: Vec<f64> = filaments.iter().map(|f| Bar::new(f).centre.z).collect();
    let mut sorted: Vec<usize> = (0..zs.len()).collect();
    sorted.sort_by(|&a, &b| {
        zs[a]
            .partial_cmp(&zs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let extent = zs[sorted[sorted.len() - 1]] - zs[sorted[0]];
    let tol = (extent * 1e-6).max(f64::MIN_POSITIVE);
    let mut cluster_first: Vec<f64> = Vec::new();
    let mut layer_of_sorted: Vec<usize> = Vec::with_capacity(zs.len());
    for &index in &sorted {
        let z = zs[index];
        match cluster_first.last() {
            Some(&first) if z - first <= tol => {}
            _ => cluster_first.push(z),
        }
        layer_of_sorted.push(cluster_first.len() - 1);
    }
    // The cluster id of a value's FIRST sorted occurrence holds for all
    // its duplicates (the walk merged them); rank = count of strictly
    // smaller values = position of the first occurrence.
    (0..zs.len())
        .map(|b| {
            let rank = sorted.partition_point(|&idx| zs[idx] < zs[b]);
            layer_of_sorted[rank]
        })
        .collect()
}

/// XY bin map (bin size = radius) over one layer's member branches.
struct LayerGrid {
    /// The layer's branch indices, in member order; slots index `index`.
    members: Vec<usize>,
    /// CSS bin key -> slots whose centre falls in it.
    bins: HashMap<u64, Vec<usize>>,
}

fn bin_key(x: f64, y: f64, radius: f64) -> u64 {
    let bx = (x / radius).floor() as i64;
    let by = (y / radius).floor() as i64;
    ((bx as u64) << 32) | (by as i64 as u64 & 0xffff_ffff)
}

impl LayerGrid {
    fn build(filaments: &[Filament], members_in_layer: &[usize], radius: f64) -> Self {
        let mut bins: HashMap<u64, Vec<usize>> = HashMap::new();
        for (slot, &b) in members_in_layer.iter().enumerate() {
            let centre = Bar::new(&filaments[b]).centre;
            let key = bin_key(centre.x, centre.y, radius);
            bins.entry(key).or_default().push(slot);
        }
        LayerGrid {
            members: members_in_layer.iter().copied().collect(),
            bins,
        }
    }

    fn slice(&self, key: u64) -> &[usize] {
        match self.bins.get(&key) {
            Some(bucket) => bucket.as_slice(),
            None => &[],
        }
    }
}

/// One sweep between two layers' grids. When `cross` is false the scan
/// runs the in-layer upper triangle (slot-ordered skip of j <= a);
/// when true, both sides counted with the touching pattern.
fn scan_pair(
    filaments: &[Filament],
    branch_rows: &[Vec<u32>],
    a_grid: &LayerGrid,
    radius: f64,
    cross: bool,
    touched: Option<&[AtomicUsize]>,
    counters: &[AtomicUsize],
) {
    const IN_LAYER_PAIRS: usize = 0;
    const CROSS_PAIRS: usize = 1;
    const CROSS_ENTRIES: usize = 2;
    let bin_r = radius.max(f64::MIN_POSITIVE);
    // Orthogonality matches the pattern stage (near.rs thresholds 1e-14).
    const ORTHOGONAL: f64 = 1e-14;

    a_grid.members.par_iter().enumerate().for_each(|(_slot_a, &a)| {
        let bar = Bar::new(&filaments[a]);
        let dir_a = filaments[a].direction();
        let bx = (bar.centre.x / bin_r).floor() as i64;
        let by = (bar.centre.y / bin_r).floor() as i64;
        for dx in -1..=1 {
            for dy in -1..=1 {
                let key =
                    ((bx + dx) as u64) << 32 | ((by + dy) as i64 as u64 & 0xffff_ffff);
                for slot_b in a_grid.slice(key) {
                    let b = a_grid.members[*slot_b];
                    if cross {
                        if b == a {
                            continue;
                        }
                    } else if b <= a {
                        continue;
                    }
                    let bar_b = Bar::new(&filaments[b]);
                    let delta = bar_b.centre - bar.centre;
                    if delta.norm() >= radius {
                        continue;
                    }
                    if dir_a.dot(&filaments[b].direction()).abs() < ORTHOGONAL {
                        continue;
                    }
                    if cross {
                        counters[CROSS_PAIRS].fetch_add(1, Ordering::Relaxed);
                        let entries = branch_rows[a].len() * branch_rows[b].len();
                        counters[CROSS_ENTRIES].fetch_add(entries, Ordering::Relaxed);
                        if let Some(touch) = touched {
                            for &row in &branch_rows[a] {
                                touch[row as usize].store(1, Ordering::Relaxed);
                            }
                            for &row in &branch_rows[b] {
                                touch[row as usize].store(1, Ordering::Relaxed);
                            }
                        }
                    } else {
                        counters[IN_LAYER_PAIRS].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    });
}

/// The CLI entry point for the census: builds the same mesh the
/// iterative path builds (no operator, no factorization), then runs
/// [`interface_census`] at `radius` (when `None`, `derived_radius`
/// supplies the labeled first guess and it is printed).
pub fn census_for(
    geometry: &crate::geometry::Geometry,
    ports: &[crate::mesh::Port],
    discretization: &crate::solve::Discretization,
    radius: Option<f64>,
) -> Result<Census, crate::solve::SolveError> {
    let (filaments, per_segment) =
        crate::solve::discretize_geometry(geometry, discretization)?;
    let mesh = crate::mesh::MeshMatrix::build(geometry, &per_segment, ports)?;
    let internal = mesh.internal_count();
    let radius = match radius {
        Some(value) => {
            eprintln!("census: radius {value} m (explicit)");
            value
        }
        None => {
            let guess = derived_radius(&filaments);
            eprintln!("census: radius {guess} m (derived first guess)");
            guess
        }
    };
    Ok(interface_census(&mesh, &filaments, internal, radius))
}

/// Full interface census against the loop pattern the precond would
/// build, at an explicit radius. Shared counters stream counts only;
/// the touched-row set is a flat 0/1 array (`internal` words).
pub fn interface_census(
    mesh: &MeshMatrix,
    filaments: &[Filament],
    internal: usize,
    radius: f64,
) -> Census {
    let n_l: Vec<usize> = branch_layers(filaments);
    let mut per_layer_branches: Vec<Vec<usize>> = Vec::new();
    let max_layer = n_l.iter().copied().max().unwrap_or(0);
    for _ in 0..=max_layer {
        per_layer_branches.push(Vec::new());
    }
    for (b, &layer) in n_l.iter().enumerate() {
        per_layer_branches[layer].push(b);
    }
    // Row tagging: first branch of the row; while there, build the
    // branch -> rows incidence (the same structure the CUBEL assembler
    // builds; needed for the cross-entry estimate).
    let mut rows_per_layer = vec![0usize; max_layer + 1];
    let mut spanning_rows = 0usize;
    let mut branch_rows: Vec<Vec<u32>> = {
        let mut counts: Vec<usize> = vec![0; filaments.len()];
        for row in 0..internal {
            for &(b, _) in mesh.row(row) {
                counts[b] += 1;
            }
        }
        counts
            .iter()
            .map(|&c| Vec::with_capacity(c))
            .collect::<Vec<_>>()
    };
    let mut first_layers: Vec<usize> = Vec::with_capacity(internal);
    for row in 0..internal {
        let row_branches = mesh.row(row);
        let spread = row_branches
            .iter()
            .map(|&(b, _)| n_l[b])
            .collect::<std::collections::BTreeSet<_>>();
        if spread.len() > 1 {
            spanning_rows += 1;
        }
        let first = row_branches.first().map(|&(b, _)| b).unwrap_or(0);
        first_layers.push(n_l[first]);
        for &(b, _) in row_branches {
            branch_rows[b].push(row as u32);
        }
    }
    for &layer in &first_layers {
        rows_per_layer[layer] += 1;
    }

    // Filters off the pair scans: shared counters + flat touched set.
    let counters: Vec<AtomicUsize> = (0..3).map(|_| AtomicUsize::new(0)).collect();
    let touched: Vec<AtomicUsize> = (0..internal).map(|_| AtomicUsize::new(0)).collect();
    let touch: &[AtomicUsize] = &touched;

    let wall = std::time::Instant::now();
    let mut grids: Vec<LayerGrid> = per_layer_branches
        .iter()
        .map(|members| LayerGrid::build(filaments, members, radius))
        .collect();
    for a_layer in 0..grids.len() {
        for b_layer in a_layer..grids.len() {
            scan_pair(
                filaments,
                &branch_rows,
                &grids[a_layer],
                radius,
                a_layer != b_layer,
                if a_layer != b_layer { Some(touch) } else { None },
                &counters,
            );
        }
    }
    let branch_pairs_in_layer_upper = counters[0].load(Ordering::Relaxed);
    let branch_cross_pairs_upper = counters[1].load(Ordering::Relaxed);
    let csc_cross_entries = 2 * counters[2].load(Ordering::Relaxed);
    let interface_rows = touched
        .iter()
        .filter(|t| t.load(Ordering::Relaxed) != 0)
        .count();
    eprintln!(
        "census: layers {}, branches {:?}", max_layer + 1, per_layer_branches.iter().map(|v| v.len()).collect::<Vec<_>>()
    );
    let _ = wall;
    Census {
        layers: max_layer + 1,
        branches_per_layer: per_layer_branches.iter().map(|v| v.len()).collect(),
        rows_per_layer,
        spanning_rows,
        branch_pairs_in_layer_upper,
        branch_cross_pairs_upper,
        csc_cross_entries,
        interface_rows,
    }
}
