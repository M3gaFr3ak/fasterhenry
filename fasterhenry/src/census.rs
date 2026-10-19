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
    /// The via-density census (v2 conv gate), `Some` from
    /// `census_for`; `None` from a bare `interface_census`.
    pub via: Option<ViaCensus>,
}

/// The threshold split of the via-density census (v2 conv engine,
/// NOTES 2026-10-19/20). The engine's decision unit is the PATCH: a
/// patch holding any cell at or above the density threshold
/// homogenizes whole (conv side), every other patch stays explicit
/// via tabs - a per-independent-cell decision would shatter a via
/// field into mixed conv/tab seams.
#[derive(Debug, Clone)]
pub struct ThresholdSplit {
    /// The split threshold, vias per cell.
    pub threshold: usize,
    /// Cells inside conv patches.
    pub conv_cells: usize,
    /// Via branches inside conv patches.
    pub conv_branches: usize,
    /// Patches homogenized (conv side).
    pub conv_patches: usize,
    /// Via branches in conv patches (== `conv_branches` by
    /// construction; kept for print symmetry with the tabs row).
    pub conv_patch_branches: usize,
    /// Cells in tab patches.
    pub tab_cells: usize,
    /// Via branches in tab patches.
    pub tab_branches: usize,
    /// Patches staying explicit tabs.
    pub tab_patches: usize,
    /// Via branches in tab patches (== `tab_branches` by
    /// construction).
    pub tab_patch_branches: usize,
}

/// Result of the via-density census: the per-XY-cell via population
/// the `FASTERHENRY_CONV_VIA_DENSITY` threshold switch needs. A branch
/// counts as a via when its current direction is predominantly
/// z-directed (2.5D decks: only via barrels span layers). O(N),
/// streamed, no pair scans.
#[derive(Debug, Clone, Default)]
pub struct ViaCensus {
    /// Total branches scanned.
    pub total_branches: usize,
    /// Branches classified as via barrels.
    pub via_branches: usize,
    /// Via branches per z-cluster (`branch_layers` tagging; a via
    /// barrel's centre sits between copper planes, so it lands in a
    /// cluster of its own - this row mirrors ekart's via populations).
    pub vias_per_cluster: Vec<usize>,
    /// The bin pitch actually used (metres).
    pub bin_m: f64,
    /// XY bins holding at least one via branch.
    pub occupied_cells: usize,
    /// Largest per-cell via count.
    pub max_per_cell: usize,
    /// Per-count occupancy histogram, count ascending:
    /// `(per-cell count, cells holding that count)`.
    pub histogram: Vec<(usize, usize)>,
    /// The threshold split (only when a threshold was supplied).
    pub threshold: Option<ThresholdSplit>,
    /// Connected via-field patches (4-connected flood fill over the
    /// bin lattice, threshold-free): the unit local thresholding
    /// should act on - the conv side homogenizes via-field PATCHES,
    /// never scattered single cells (mixed conv/tab cells inside one
    /// field is the seam to avoid). Always computed; local-threshold
    /// variants (quantile, neighborhood-relative) read this, not the
    /// raw cells.
    pub patches: PatchCensus,
}

/// Connected-patch structure of the occupied-cell lattice.
#[derive(Debug, Clone, Default)]
pub struct PatchCensus {
    /// Number of connected via-field patches.
    pub count: usize,
    /// Largest patch, in cells.
    pub largest_cells: usize,
    /// Patch-size occupancy: `(patch size in cells, patches of that
    /// size)`, size ascending (the same compressed form as the
    /// per-cell histogram).
    pub size_histogram: Vec<(usize, usize)>,
    /// Largest patch's total via branches.
    pub largest_branches: usize,
}

impl ViaCensus {
    /// The stderr report (one instrument, self-contained).
    pub(crate) fn report(&self) {
        let hist: Vec<String> = self
            .histogram
            .iter()
            .take(12)
            .map(|(count, cells)| format!("{}:{}", count, cells))
            .collect();
        let more = match self.histogram.len() {
            len if len > 12 => format!(", +{} more buckets", len - 12),
            _ => String::new(),
        };
        eprintln!(
            "census: via: {} of {} branches are vias, per-cluster {:?}",
            self.via_branches, self.total_branches, self.vias_per_cluster
        );
        eprintln!(
            "census: via: bin {:.3e} m: occupied cells {}, max/cell {}, histogram [{}{}]",
            self.bin_m,
            self.occupied_cells,
            self.max_per_cell,
            hist.join(" "),
            more
        );
        let patches = &self.patches;
        let size_hist: Vec<String> = patches
            .size_histogram
            .iter()
            .take(12)
            .map(|(size, count)| format!("{}:{}", size, count))
            .collect();
        eprintln!(
            "census: via: patches {} (largest {} cells / {} via branches), sizes [{}]",
            patches.count,
            patches.largest_cells,
            patches.largest_branches,
            size_hist.join(" ")
        );
        match &self.threshold {
            Some(split) => eprintln!(
                "census: via: threshold {}: conv {} patches / {} cells ({} via branches) / tab {} patches / {} cells ({} via branches)",
                split.threshold,
                split.conv_patches,
                split.conv_cells,
                split.conv_branches,
                split.tab_patches,
                split.tab_cells,
                split.tab_branches
            ),
            None => eprintln!(
                "census: via: no FASTERHENRY_CENSUS_VIA_DENSITY threshold, split not evaluated"
            ),
        }
    }
}

/// A branch is a via when its current direction is predominantly
/// z-directed; 2.5D decks route everything else in the XY plane, so
/// only via barrels come near vertical.
const VIA_IS_Z: f64 = 0.5;

/// Per-XY-cell via density at bin pitch `bin_m`, with the optional
/// threshold split. Single-threaded by design: the via population is
/// orders of magnitude below the branch population (ekart: 5
/// via-barrel clusters of 3,596 branches against 2.86M filaments), so
/// a parallel scan would cost more than it saves.
pub fn via_census(filaments: &[Filament], bin_m: f64, threshold: Option<usize>) -> ViaCensus {
    let bin = if bin_m.is_finite() && bin_m > 0.0 {
        bin_m
    } else {
        f64::MIN_POSITIVE
    };
    let layers = branch_layers(filaments);
    let max_layer = layers.iter().copied().max().unwrap_or(0);
    let mut vias_per_cluster = vec![0usize; max_layer + 1];
    let mut per_cell: HashMap<u64, usize> = HashMap::new();
    let mut via_branches = 0usize;
    for (filament, layer) in filaments.iter().zip(&layers) {
        if filament.direction().z.abs() < VIA_IS_Z {
            continue;
        }
        via_branches += 1;
        vias_per_cluster[*layer] += 1;
        let centre = filament.center();
        let key = bin_key(centre.x, centre.y, bin);
        *per_cell.entry(key).or_insert(0) += 1;
    }
    // Connected-patch flood fill over the occupied-cell lattice
    // (4-connectivity, same key packing as `bin_key`): the patch is
    // the unit a local threshold acts on - scattered single-cell
    // decisions shatter a via field into mixed conv/tab seams.
    let mut patch_of: HashMap<u64, u32> = HashMap::with_capacity(per_cell.len());
    let mut patch_cells: Vec<usize> = Vec::new();
    let mut patch_branches: Vec<usize> = Vec::new();
    if !per_cell.is_empty() {
        let mut frontier: Vec<u64> = Vec::new();
        for &key in per_cell.keys() {
            if patch_of.contains_key(&key) {
                continue;
            }
            let id = patch_cells.len() as u32;
            patch_of.insert(key, id);
            frontier.clear();
            frontier.push(key);
            let mut cells = 0usize;
            let mut branches = 0usize;
            while let Some(cell) = frontier.pop() {
                cells += 1;
                branches += per_cell[&cell];
                let bx = (cell >> 32) as i32 as i64;
                let by = (cell as u32 as i32) as i64;
                let neighbours = [
                    (((bx + 1) as u64) << 32) | (by as i64 as u64 & 0xffff_ffff),
                    (((bx - 1) as u64) << 32) | (by as i64 as u64 & 0xffff_ffff),
                    ((bx as u64) << 32) | ((by + 1) as i64 as u64 & 0xffff_ffff),
                    ((bx as u64) << 32) | ((by - 1) as i64 as u64 & 0xffff_ffff),
                ];
                for neighbour in neighbours {
                    if per_cell.contains_key(&neighbour) && !patch_of.contains_key(&neighbour) {
                        patch_of.insert(neighbour, id);
                        frontier.push(neighbour);
                    }
                }
            }
            patch_cells.push(cells);
            patch_branches.push(branches);
        }
    }
    let mut patch_of: HashMap<u64, u32> = HashMap::with_capacity(per_cell.len());
    let mut patch_cells: Vec<usize> = Vec::new();
    let mut patch_branches: Vec<usize> = Vec::new();
    let mut patch_max: Vec<usize> = Vec::new();
    if !per_cell.is_empty() {
        let mut frontier: Vec<u64> = Vec::new();
        for &key in per_cell.keys() {
            if patch_of.contains_key(&key) {
                continue;
            }
            let id = patch_cells.len() as u32;
            patch_of.insert(key, id);
            frontier.clear();
            frontier.push(key);
            let mut cells = 0usize;
            let mut branches = 0usize;
            let mut widest = 0usize;
            while let Some(cell) = frontier.pop() {
                let count = per_cell[&cell];
                cells += 1;
                branches += count;
                widest = widest.max(count);
                let bx = (cell >> 32) as i32 as i64;
                let by = (cell as u32 as i32) as i64;
                let neighbours = [
                    (((bx + 1) as u64) << 32) | (by as i64 as u64 & 0xffff_ffff),
                    (((bx - 1) as u64) << 32) | (by as i64 as u64 & 0xffff_ffff),
                    ((bx as u64) << 32) | ((by + 1) as i64 as u64 & 0xffff_ffff),
                    ((bx as u64) << 32) | ((by - 1) as i64 as u64 & 0xffff_ffff),
                ];
                for neighbour in neighbours {
                    if per_cell.contains_key(&neighbour) && !patch_of.contains_key(&neighbour) {
                        patch_of.insert(neighbour, id);
                        frontier.push(neighbour);
                    }
                }
            }
            patch_cells.push(cells);
            patch_branches.push(branches);
            patch_max.push(widest);
        }
    }
    let mut size_by_count: HashMap<usize, usize> = HashMap::new();
    let mut largest_cells = 0usize;
    let mut largest_branches = 0usize;
    for (index, &cells) in patch_cells.iter().enumerate() {
        *size_by_count.entry(cells).or_insert(0) += 1;
        if cells > largest_cells {
            largest_cells = cells;
            largest_branches = patch_branches[index];
        }
    }
    let mut size_histogram: Vec<(usize, usize)> = size_by_count.into_iter().collect();
    size_histogram.sort_unstable();

    let mut by_count: HashMap<usize, usize> = HashMap::new();
    let mut max_per_cell = 0usize;
    for &count in per_cell.values() {
        max_per_cell = max_per_cell.max(count);
        *by_count.entry(count).or_insert(0) += 1;
    }
    let mut hist_pairs: Vec<(usize, usize)> = by_count
        .iter()
        .map(|(&count, &cells)| (count, cells))
        .collect();
    hist_pairs.sort_unstable();
    let threshold = threshold.map(|t| {
        let mut split = ThresholdSplit {
            threshold: t,
            conv_cells: 0,
            conv_branches: 0,
            conv_patches: 0,
            conv_patch_branches: 0,
            tab_cells: 0,
            tab_branches: 0,
            tab_patches: 0,
            tab_patch_branches: 0,
        };
        for (index, &cells) in patch_cells.iter().enumerate() {
            // The engine's decision unit is the patch: a patch holds a
            // cell at or above the threshold and its whole field
            // homogenizes; every other patch stays explicit tabs.
            if patch_max[index] >= t {
                split.conv_patches += 1;
                split.conv_patch_branches += patch_branches[index];
                split.conv_cells += cells;
                split.conv_branches += patch_branches[index];
            } else {
                split.tab_patches += 1;
                split.tab_patch_branches += patch_branches[index];
                split.tab_cells += cells;
                split.tab_branches += patch_branches[index];
            }
        }
        split
    });
    ViaCensus {
        total_branches: filaments.len(),
        via_branches,
        vias_per_cluster,
        bin_m: bin,
        occupied_cells: per_cell.len(),
        max_per_cell,
        histogram: hist_pairs,
        threshold,
        patches: PatchCensus {
            count: patch_cells.len(),
            largest_cells,
            size_histogram,
            largest_branches,
        },
    }
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

    a_grid
        .members
        .par_iter()
        .enumerate()
        .for_each(|(_slot_a, &a)| {
            let bar = Bar::new(&filaments[a]);
            let dir_a = filaments[a].direction();
            let bx = (bar.centre.x / bin_r).floor() as i64;
            let by = (bar.centre.y / bin_r).floor() as i64;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let key = ((bx + dx) as u64) << 32 | ((by + dy) as i64 as u64 & 0xffff_ffff);
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
    via_bin_m: Option<f64>,
    via_threshold: Option<usize>,
) -> Result<Census, crate::solve::SolveError> {
    let (filaments, per_segment) = crate::solve::discretize_geometry(geometry, discretization)?;
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
    let via_bin = via_bin_m.unwrap_or(radius);
    eprintln!(
        "census: via bin {:.3e} m ({})",
        via_bin,
        if via_bin_m.is_some() {
            "explicit"
        } else {
            "= census radius"
        }
    );
    let via = via_census(&filaments, via_bin, via_threshold);
    via.report();
    let mut census = interface_census(&mesh, &filaments, internal, radius);
    census.via = Some(via);
    Ok(census)
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
    let grids: Vec<LayerGrid> = per_layer_branches
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
                if a_layer != b_layer {
                    Some(touch)
                } else {
                    None
                },
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
        "census: layers {}, branches {:?}",
        max_layer + 1,
        per_layer_branches
            .iter()
            .map(|v| v.len())
            .collect::<Vec<_>>()
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
        via: None,
    }
}

#[cfg(test)]
mod via_tests {
    use super::*;
    use crate::geometry::{Node, Segment};

    fn horizontal(x0: f64, x1: f64, z: f64) -> Filament {
        Filament::new(&Segment::new(
            Node::new(x0, 0.0, z),
            Node::new(x1, 0.0, z),
            1.0e-4,
            3.5e-5,
            5.8e7,
        ))
        .unwrap()
    }

    fn via(x: f64, y: f64, z0: f64, z1: f64) -> Filament {
        Filament::new(&Segment::new(
            Node::new(x, y, z0),
            Node::new(x, y, z1),
            1.0e-4,
            3.5e-5,
            5.8e7,
        ))
        .unwrap()
    }

    #[test]
    fn via_census_buckets_verticals_per_cell() {
        let filaments = vec![
            horizontal(0.0, 1.0e-3, 0.0),
            horizontal(0.0, 1.0e-3, 7.0e-5),
            via(2.0e-3, 2.0e-3, 0.0, 7.0e-5),
            via(2.0e-3, 2.0e-3, 0.0, 7.0e-5),
            via(9.0e-3, 9.0e-3, 0.0, 7.0e-5),
        ];
        let census = via_census(&filaments, 1.0e-3, Some(2));
        assert_eq!(census.via_branches, 3);
        assert_eq!(census.total_branches, 5);
        assert_eq!(census.occupied_cells, 2);
        assert_eq!(census.max_per_cell, 2);
        assert_eq!(census.vias_per_cluster, vec![0, 3, 0]);
        let split = census.threshold.expect("split evaluated");
        assert_eq!(split.conv_patches, 1);
        assert_eq!(split.conv_patch_branches, 2);
        assert_eq!(split.conv_cells, 1);
        assert_eq!(split.tab_patches, 1);
        assert_eq!(split.tab_patch_branches, 1);
        assert_eq!(split.tab_cells, 1);
    }

    #[test]
    fn via_census_flood_fills_adjacent_cells_into_one_patch() {
        let filaments = vec![
            via(0.0, 0.0, 0.0, 7.0e-5),
            via(1.5e-3, 0.0, 0.0, 7.0e-5),
            via(9.0e-3, 9.0e-3, 0.0, 7.0e-5),
        ];
        let census = via_census(&filaments, 1.0e-3, None);
        assert!(census.threshold.is_none());
        assert_eq!(census.patches.count, 2);
        assert_eq!(census.patches.largest_cells, 2);
        assert_eq!(census.patches.largest_branches, 2);
        assert_eq!(census.patches.size_histogram, vec![(1, 1), (2, 1)]);
    }
}
