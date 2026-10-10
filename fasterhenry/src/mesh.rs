//! Ports and the mesh (loop) basis of the filament circuit.
//!
//! After discretization the conductors are an electrical network: every
//! filament is a *branch* joining the two nodes of its segment, so the
//! `nw × nh` filaments of one segment are branches in parallel. Kamon, Tsuk &
//! White (IEEE T-MTT 1994, §II.B) solve that network by *mesh analysis*: the
//! unknowns are currents circulating in a set of independent closed loops
//! (meshes), which satisfy Kirchhoff's current law by construction, and the
//! equations are Kirchhoff's voltage law around each loop.
//!
//! This module builds the loop basis, as the sparse signed incidence matrix
//! `M` (loops × branches) of [`MeshMatrix`]. The impedance assembly and the
//! solve live in [`mod@crate::solve`].
//!
//! # Conventions
//!
//! * Branch `k` is oriented like its segment, from [`SegmentDef::a`] to
//!   [`SegmentDef::b`]. Its current `I_k` is positive in that direction and
//!   its voltage is `v_k = φ(a) − φ(b)`, so that `v = Z_b · I` with
//!   `Z_b = R + jωL`.
//! * `M[i][k]` is `+1` if loop `i` traverses branch `k` forwards, `−1` if
//!   backwards, `0` if not at all. Branch currents follow from loop currents
//!   by `I_b = Mᵀ · I_m`.
//! * A [`Port`] is closed externally by a source: the port current `I_p`
//!   enters the conductor at [`Port::positive`], leaves it at
//!   [`Port::negative`], and the port voltage is
//!   `V_p = φ(positive) − φ(negative)`. The port's loop runs through the
//!   conductor from `positive` to `negative` and back through the source, so
//!   its loop current *is* the port current.
//!
//! # The loop basis
//!
//! Three families of loops, which together are a basis of the cycle space of
//! the network closed by its port sources:
//!
//! 1. **Bundle loops** — within every segment, filament `k ≥ 1` forwards and
//!    filament `0` backwards: `nw·nh − 1` two-branch loops per segment. These
//!    carry the redistribution of current over a cross-section (skin and
//!    proximity effect).
//! 2. **Conductor loops** — a breadth-first spanning forest of the segment
//!    graph is grown; every segment left out of it closes one loop through
//!    the forest, realised on filament `0` of each segment involved. A
//!    geometry without closed rings has none.
//! 3. **Port loops** — for every port, the forest path from its positive to
//!    its negative node, again on filament `0` of each segment.
//!
//! Families 1 and 2 are the *internal* loops (no source in them); they come
//! first in the row order, then the port loops in port order. The
//! construction is deterministic: it depends only on the order of nodes,
//! segments and ports.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::geometry::{Geometry, NodeId, SegmentDef};

/// A terminal pair at which the impedance is extracted.
///
/// See the [module documentation](self) for the sign convention: current
/// enters at [`positive`](Self::positive), and the voltage is that of
/// `positive` relative to [`negative`](Self::negative). Swapping the two
/// nodes flips the sign of the port's off-diagonal impedances and leaves its
/// self-impedance unchanged.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Port {
    /// Node at which the port current enters the conductor.
    pub positive: NodeId,
    /// Node at which the port current leaves the conductor.
    pub negative: NodeId,
    /// Optional label, carried through to the results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Port {
    /// An unnamed port from `positive` to `negative`.
    pub const fn new(positive: NodeId, negative: NodeId) -> Self {
        Self {
            positive,
            negative,
            name: None,
        }
    }

    /// Returns the port with a label.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Why a set of ports cannot be attached to a geometry.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MeshError {
    /// No port was given, so there is no impedance to extract.
    #[error("at least one port is required")]
    NoPorts,
    /// A port refers to a node the geometry does not have.
    #[error("port {port} refers to node {node}, but the geometry has {node_count} nodes")]
    UnknownPortNode {
        /// Index of the offending port.
        port: usize,
        /// The node index it refers to.
        node: usize,
        /// Number of nodes in the geometry.
        node_count: usize,
    },
    /// Both terminals of a port are the same node.
    #[error("port {port} has the same node ({node}) as both terminals")]
    DegeneratePort {
        /// Index of the offending port.
        port: usize,
        /// The repeated node index.
        node: usize,
    },
    /// The two terminals of a port are not joined by conductor, so no current
    /// can flow between them and the port has no impedance.
    #[error(
        "port {port}: nodes {positive} and {negative} are not connected by any chain of segments"
    )]
    PortNotConnected {
        /// Index of the offending port.
        port: usize,
        /// The port's positive node index.
        positive: usize,
        /// The port's negative node index.
        negative: usize,
    },
    /// The per-segment filament counts do not match the geometry.
    #[error("got filament counts for {got} segments, but the geometry has {expected}")]
    SegmentCountMismatch {
        /// Number of segments in the geometry.
        expected: usize,
        /// Number of filament counts supplied.
        got: usize,
    },
    /// A segment was given no filaments.
    #[error("segment {segment} has no filaments")]
    EmptySegment {
        /// Index of the offending segment.
        segment: usize,
    },
}

/// A single (a, b) branch in the matrix-injection path
/// ([`MeshMatrix::from_branches`]). Each branch is a node-to-node
/// conductor; downstream adapters convert their IR's offset filaments
/// into one Branch per strand (no sub-bundling on the matrix path -
/// the caller's subdivide controls the discretization).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Branch {
    /// Start node.
    pub a: NodeId,
    /// End node.
    pub b: NodeId,
}

impl Branch {
    /// A branch from `a` to `b`.
    pub const fn new(a: NodeId, b: NodeId) -> Self {
        Self { a, b }
    }
}

/// The sparse signed loop–branch incidence matrix `M`; see the
/// [module documentation](self).
#[derive(Clone, Debug, PartialEq)]
pub struct MeshMatrix {
    rows: Vec<Vec<(usize, f64)>>,
    branch_count: usize,
    internal_count: usize,
}

impl MeshMatrix {
    /// Builds the loop basis of `geometry`, whose segment `s` contributes
    /// `filaments_per_segment[s]` parallel branches, closed by `ports`.
    ///
    /// Branches are numbered segment by segment, in the order
    /// [`discretize`](crate::discretize) produces the filaments.
    ///
    /// # Errors
    ///
    /// A [`MeshError`] if the ports or the filament counts do not fit the
    /// geometry.
    pub fn build(
        geometry: &Geometry,
        filaments_per_segment: &[usize],
        ports: &[Port],
    ) -> Result<Self, MeshError> {
        Self::build_inner(
            geometry.nodes().len(),
            geometry.segment_defs(),
            filaments_per_segment,
            ports,
        )
    }

    /// Builds the loop basis from a custom branch list (the matrix-injection
    /// path - `peec-solve::FasterhenryMatrixSolver`). Each branch is a
    /// `(a, b)` node pair; the total `node_count` defines the forest's
    /// universe (the caller's IR owns the node addressing). Every branch
    /// contributes exactly **one** parallel filament (no sub-bundling on
    /// this path - the caller's subdivide controls the discretization).
    ///
    /// # Errors
    ///
    /// A [`MeshError`] if the ports do not fit the branch graph
    /// (`NoPorts`, `UnknownPortNode`, `DegeneratePort`,
    /// `PortNotConnected`).
    pub fn from_branches(
        branches: &[Branch],
        node_count: usize,
        ports: &[Port],
    ) -> Result<Self, MeshError> {
        if branches.is_empty() {
            // The matrix path always passes at least one branch per
            // filament. With zero branches the inner logic would
            // SegmentCountMismatch (expected 0, got 0); we surface a
            // clearer empty-branches error here.
            return Err(MeshError::SegmentCountMismatch {
                expected: 0,
                got: 0,
            });
        }
        // Build a minimal `SegmentDef` vector for the forest / loop basis.
        // The dimensions are placeholders (`1.0, 1.0, 1.0`); the loop
        // basis only needs the (a, b) connectivity graph.
        let defs: Vec<SegmentDef> = branches
            .iter()
            .map(|b| SegmentDef::new(b.a, b.b, 1.0, 1.0, 1.0))
            .collect();
        let filaments_per_branch: Vec<usize> = vec![1; branches.len()];
        Self::build_inner(node_count, &defs, &filaments_per_branch, ports)
    }

    /// Inner loop-basis builder - the shared logic between [`build`] and
    /// [`from_branches`]. Public callers should prefer those two.
    fn build_inner(
        node_count: usize,
        defs: &[SegmentDef],
        filaments_per_segment: &[usize],
        ports: &[Port],
    ) -> Result<Self, MeshError> {
        if filaments_per_segment.len() != defs.len() {
            return Err(MeshError::SegmentCountMismatch {
                expected: defs.len(),
                got: filaments_per_segment.len(),
            });
        }
        if let Some(segment) = filaments_per_segment.iter().position(|&n| n == 0) {
            return Err(MeshError::EmptySegment { segment });
        }
        if ports.is_empty() {
            return Err(MeshError::NoPorts);
        }

        // First branch of every segment: the one conductor and port loops use.
        let first_branch: Vec<usize> = filaments_per_segment
            .iter()
            .scan(0, |next, &n| {
                let first = *next;
                *next += n;
                Some(first)
            })
            .collect();
        let branch_count = filaments_per_segment.iter().sum();

        let mut rows = Vec::new();
        for (&first, &n) in first_branch.iter().zip(filaments_per_segment) {
            rows.extend((1..n).map(|k| vec![(first + k, 1.0), (first, -1.0)]));
        }

        let forest = Forest::grow(node_count, defs);
        for (s, def) in defs.iter().enumerate() {
            if !forest.in_tree[s] {
                // Forwards along the segment, then home through the forest.
                let mut row = vec![(s, 1.0)];
                row.extend(forest.path(def.b.0, def.a.0));
                rows.push(row);
            }
        }
        let internal_count = rows.len();

        for (port, p) in ports.iter().enumerate() {
            let (positive, negative) = (p.positive.0, p.negative.0);
            if let Some(&node) = [positive, negative].iter().find(|&&n| n >= node_count) {
                return Err(MeshError::UnknownPortNode {
                    port,
                    node,
                    node_count,
                });
            }
            if positive == negative {
                return Err(MeshError::DegeneratePort {
                    port,
                    node: positive,
                });
            }
            match (forest.component[positive], forest.component[negative]) {
                (Some(a), Some(b)) if a == b => rows.push(forest.path(positive, negative)),
                _ => {
                    return Err(MeshError::PortNotConnected {
                        port,
                        positive,
                        negative,
                    })
                }
            }
        }

        // The forest works in segment indices; the loops live on branches.
        for row in &mut rows[internal_count - forest.link_count()..] {
            for entry in row {
                entry.0 = first_branch[entry.0];
            }
        }
        Ok(Self {
            rows,
            branch_count,
            internal_count,
        })
    }

    /// Total number of loops: internal loops plus one per port.
    pub fn loop_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of internal (source-free) loops; they are rows
    /// `0..internal_count()`.
    pub fn internal_count(&self) -> usize {
        self.internal_count
    }

    /// Number of port loops; port `p` is row `internal_count() + p`.
    pub fn port_count(&self) -> usize {
        self.rows.len() - self.internal_count
    }

    /// Number of branches (columns).
    pub fn branch_count(&self) -> usize {
        self.branch_count
    }

    /// The non-zero entries `(branch, ±1)` of loop `index`, in the order the
    /// loop traverses them.
    ///
    /// # Panics
    ///
    /// If `index >= loop_count()`.
    pub fn row(&self, index: usize) -> &[(usize, f64)] {
        &self.rows[index]
    }
}

/// Breadth-first spanning forest of the segment graph.
struct Forest {
    /// For every non-root node reached: its parent, the tree segment joining
    /// them, and whether that segment is oriented parent → node.
    parent: Vec<Option<(usize, usize, bool)>>,
    depth: Vec<usize>,
    /// Connected component of every node that has at least one segment.
    component: Vec<Option<usize>>,
    in_tree: Vec<bool>,
}

impl Forest {
    fn grow(node_count: usize, defs: &[SegmentDef]) -> Self {
        let mut adjacency = vec![Vec::new(); node_count];
        for (s, def) in defs.iter().enumerate() {
            adjacency[def.a.0].push((def.b.0, s, true));
            adjacency[def.b.0].push((def.a.0, s, false));
        }
        let mut forest = Self {
            parent: vec![None; node_count],
            depth: vec![0; node_count],
            component: vec![None; node_count],
            in_tree: vec![false; defs.len()],
        };
        let mut components = 0;
        for root in 0..node_count {
            if forest.component[root].is_some() || adjacency[root].is_empty() {
                continue;
            }
            forest.component[root] = Some(components);
            let mut queue = VecDeque::from([root]);
            while let Some(node) = queue.pop_front() {
                for &(next, segment, forwards) in &adjacency[node] {
                    if forest.component[next].is_none() {
                        forest.component[next] = Some(components);
                        forest.parent[next] = Some((node, segment, forwards));
                        forest.depth[next] = forest.depth[node] + 1;
                        forest.in_tree[segment] = true;
                        queue.push_back(next);
                    }
                }
            }
            components += 1;
        }
        forest
    }

    fn link_count(&self) -> usize {
        self.in_tree.iter().filter(|&&t| !t).count()
    }

    /// The signed tree segments traversed from `from` to `to`, which must be
    /// in the same component.
    fn path(&self, from: usize, to: usize) -> Vec<(usize, f64)> {
        let (mut up, mut down) = (Vec::new(), Vec::new());
        let (mut x, mut y) = (from, to);
        while x != y {
            // Climb from whichever end is deeper (from `from` on a tie).
            if self.depth[x] >= self.depth[y] {
                let (parent, segment, forwards) = self.parent[x].expect("same component");
                // Moving child → parent runs against a parent → child segment.
                up.push((segment, if forwards { -1.0 } else { 1.0 }));
                x = parent;
            } else {
                let (parent, segment, forwards) = self.parent[y].expect("same component");
                down.push((segment, if forwards { 1.0 } else { -1.0 }));
                y = parent;
            }
        }
        up.extend(down.into_iter().rev());
        up
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Node;
    use nalgebra::DMatrix;

    const COPPER: f64 = 5.8e7;

    /// Nodes at `points`, joined by `segments` given as node-index pairs.
    fn geometry(points: &[[f64; 3]], segments: &[(usize, usize)]) -> Geometry {
        let mut geometry = Geometry::new();
        for &point in points {
            geometry.add_node(Node::from(point)).unwrap();
        }
        for &(a, b) in segments {
            geometry
                .add_segment(SegmentDef::new(NodeId(a), NodeId(b), 1e-4, 1e-4, COPPER))
                .unwrap();
        }
        geometry
    }

    /// A unit square ring 0-1-2-3-0 with a stub 3-4, and a separate bar 5-6.
    fn ring_with_stub_and_floating_bar() -> Geometry {
        geometry(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            // The ring is deliberately not oriented consistently.
            &[(0, 1), (2, 1), (2, 3), (0, 3), (3, 4), (5, 6)],
        )
    }

    /// Net current leaving every node when loop `row` carries unit current.
    fn node_divergence(
        geometry: &Geometry,
        per_segment: &[usize],
        row: &[(usize, f64)],
    ) -> Vec<f64> {
        let owner: Vec<usize> = per_segment
            .iter()
            .enumerate()
            .flat_map(|(s, &n)| std::iter::repeat_n(s, n))
            .collect();
        let mut divergence = vec![0.0; geometry.nodes().len()];
        for &(branch, sign) in row {
            let def = geometry.segment_defs()[owner[branch]];
            divergence[def.a.0] += sign;
            divergence[def.b.0] -= sign;
        }
        divergence
    }

    fn dense(mesh: &MeshMatrix) -> DMatrix<f64> {
        let mut dense = DMatrix::zeros(mesh.loop_count(), mesh.branch_count());
        for i in 0..mesh.loop_count() {
            for &(branch, sign) in mesh.row(i) {
                assert_eq!(dense[(i, branch)], 0.0, "loop {i} repeats branch {branch}");
                dense[(i, branch)] = sign;
            }
        }
        dense
    }

    #[test]
    fn loops_are_closed_independent_and_complete() {
        let geometry = ring_with_stub_and_floating_bar();
        let per_segment = [3, 1, 2, 4, 2, 3];
        let ports = [
            Port::new(NodeId(1), NodeId(4)),
            Port::new(NodeId(3), NodeId(0)),
        ];
        let mesh = MeshMatrix::build(&geometry, &per_segment, &ports).unwrap();

        // Cycle-space dimension: branches − nodes + components, plus the ports.
        let branches: usize = per_segment.iter().sum();
        assert_eq!(mesh.branch_count(), branches);
        assert_eq!(mesh.internal_count(), branches - 7 + 2);
        assert_eq!(mesh.port_count(), 2);
        assert_eq!(mesh.loop_count(), mesh.internal_count() + 2);

        for i in 0..mesh.internal_count() {
            let divergence = node_divergence(&geometry, &per_segment, mesh.row(i));
            assert!(
                divergence.iter().all(|&d| d == 0.0),
                "loop {i} is not closed"
            );
        }
        for (p, port) in ports.iter().enumerate() {
            let row = mesh.row(mesh.internal_count() + p);
            let mut expected = vec![0.0; 7];
            expected[port.positive.0] = 1.0;
            expected[port.negative.0] = -1.0;
            assert_eq!(node_divergence(&geometry, &per_segment, row), expected);
        }

        let rank = dense(&mesh).rank(1e-9);
        assert_eq!(rank, mesh.loop_count(), "the loops are not independent");
    }

    #[test]
    fn single_filament_chain_has_only_the_port_loop() {
        let geometry = geometry(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            &[(0, 1), (2, 1)],
        );
        let mesh =
            MeshMatrix::build(&geometry, &[1, 1], &[Port::new(NodeId(0), NodeId(2))]).unwrap();
        assert_eq!(mesh.internal_count(), 0);
        // Forwards along segment 0, backwards along segment 1.
        assert_eq!(mesh.row(0), &[(0, 1.0), (1, -1.0)]);

        let reversed =
            MeshMatrix::build(&geometry, &[1, 1], &[Port::new(NodeId(2), NodeId(0))]).unwrap();
        assert_eq!(reversed.row(0), &[(1, 1.0), (0, -1.0)]);
    }

    #[test]
    fn bundle_loops_pair_each_filament_with_the_first() {
        let geometry = geometry(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], &[(0, 1)]);
        let mesh = MeshMatrix::build(&geometry, &[3], &[Port::new(NodeId(0), NodeId(1))]).unwrap();
        assert_eq!(mesh.internal_count(), 2);
        assert_eq!(mesh.row(0), &[(1, 1.0), (0, -1.0)]);
        assert_eq!(mesh.row(1), &[(2, 1.0), (0, -1.0)]);
        assert_eq!(mesh.row(2), &[(0, 1.0)]);
    }

    #[test]
    fn construction_is_deterministic() {
        let geometry = ring_with_stub_and_floating_bar();
        let ports = [Port::new(NodeId(0), NodeId(2)).named("across")];
        let first = MeshMatrix::build(&geometry, &[2; 6], &ports).unwrap();
        assert_eq!(
            first,
            MeshMatrix::build(&geometry, &[2; 6], &ports).unwrap()
        );
    }

    #[test]
    fn invalid_ports_and_counts_are_rejected() {
        let geometry = ring_with_stub_and_floating_bar();
        let build = |counts: &[usize], ports: &[Port]| MeshMatrix::build(&geometry, counts, ports);
        let ok = [1; 6];

        assert_eq!(build(&ok, &[]), Err(MeshError::NoPorts));
        assert_eq!(
            build(&ok, &[Port::new(NodeId(0), NodeId(7))]),
            Err(MeshError::UnknownPortNode {
                port: 0,
                node: 7,
                node_count: 7
            })
        );
        assert_eq!(
            build(
                &ok,
                &[
                    Port::new(NodeId(0), NodeId(1)),
                    Port::new(NodeId(2), NodeId(2))
                ]
            ),
            Err(MeshError::DegeneratePort { port: 1, node: 2 })
        );
        // Node 5 belongs to the floating bar, node 0 to the ring.
        assert_eq!(
            build(&ok, &[Port::new(NodeId(0), NodeId(5))]),
            Err(MeshError::PortNotConnected {
                port: 0,
                positive: 0,
                negative: 5
            })
        );
        assert_eq!(
            build(&[1; 5], &[Port::new(NodeId(0), NodeId(1))]),
            Err(MeshError::SegmentCountMismatch {
                expected: 6,
                got: 5
            })
        );
        assert_eq!(
            build(&[1, 1, 0, 1, 1, 1], &[Port::new(NodeId(0), NodeId(1))]),
            Err(MeshError::EmptySegment { segment: 2 })
        );
    }

    #[test]
    fn a_node_without_segments_cannot_carry_a_port() {
        let geometry = geometry(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 9.0, 9.0]],
            &[(0, 1)],
        );
        assert_eq!(
            MeshMatrix::build(&geometry, &[1], &[Port::new(NodeId(0), NodeId(2))]),
            Err(MeshError::PortNotConnected {
                port: 0,
                positive: 0,
                negative: 2
            })
        );
    }
}
