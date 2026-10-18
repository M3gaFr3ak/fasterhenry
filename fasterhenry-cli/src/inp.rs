//! Reader for the FastHenry `.inp` deck format — a public, documented file
//! format — restricted to the subset documented below.
//!
//! # Clean-room note
//!
//! The `.inp` deck format is a published file format, and a file format is
//! not code; this parser is written from the public format description and
//! from self-authored example decks only. No FastHenry or FastCap source,
//! text, or tables were consulted, in keeping with the repository's
//! clean-room rule (see `CONTRIBUTING.md`).
//!
//! # Supported subset
//!
//! | Line | Meaning |
//! |------|---------|
//! | `.units km\|m\|cm\|mm\|um\|in\|mils` | Length unit for every coordinate and dimension (`mil` also accepted) |
//! | `.default <field>=<v> …` | Defaults for later lines: `x`, `y`, `z`, `w`, `h`, `nwinc`, `nhinc`, `rw`, `rh`, `wx`, `wy`, `wz`, `sigma` or `rho` |
//! | `N<name> [x]=<v> [y]=<v> [z]=<v>` | Node; each coordinate falls back to its `.default` |
//! | `E<name> N<a> N<b> [field]=<v> …` | Segment between two nodes; fields as for `.default` minus `x`/`y`/`z`, plus `group=<name>` |
//! | `.external N<+> N<-> [name]` | A port: current in at `N<+>`, out at `N<->`, labelled `name` (extension; default `<+>/<->`) |
//! | `.freq fmin=<v> fmax=<v> ndec=<n>` | Frequency sweep in hertz (see below) |
//! | `G<name> x1=… y1=… z1=… x2=… y2=… z2=… x3=… y3=… z3=… thick=… seg1=… seg2=… [sigma=\|rho=] [nhinc=]` | Ground plane, FastHenry corner-point form (see below) |
//! | `G<name> x1 y1 z1 x2 y2 z2 t [nx=] [ny=] [nhinc=] [sigma=\|rho=]` | Ground plane, extension form: extent, top surface `z`, thickness `t` down, `nx × ny` cells; conductivity falls back to `.default` |
//! | `.hole G<name> x1 y1 x2 y2` | Rectangular hole in that plane's footprint |
//! | `.contact G<name> x1 y1 x2 y2 [nx=] [ny=] [ratio=]` | Contact region: refine that rectangle to `nx × ny` cells, decaying outward by `ratio` |
//! | `.equiv N<a> N<b> [N<c> …]` | Electrically join two or more nodes into one |
//! | `.couples all \| <group> <group> …` | Which segment groups couple; default (no line) is all pairs |
//! | `.end` | End of deck (required) |
//!
//! Lines beginning with `*` are comments; a line beginning with `+`
//! continues the previous line. Directives are case-insensitive; node and
//! element names are case-sensitive alphanumeric tokens (`N1`, `Ea3`).
//! Conductivity is given either as `sigma=` or as its reciprocal, the
//! resistivity `rho=` (which must be positive); naming both on one line —
//! continuation lines included — is an error, while a per-line value in
//! either form overrides a `.default` in either form.
//!
//! Where this reader knowingly departs from the public format description —
//! `.title` as an explicit directive rather than an always-ignored first
//! line (unless [`ParseOptions::fasthenry_compat`] is set; see
//! [`parse_with_options`]), `.units` being mandatory, `.end` rejecting trailing content — the
//! reasoning is recorded field by field in
//! [`docs/fasthenry-compat.md`](https://github.com/2AMLogic/fasterhenry/blob/main/docs/fasthenry-compat.md),
//! alongside every directive this reader does and does not accept.
//!
//! # Ground planes: two `G` grammars
//!
//! A `G` statement may be written either way, and a deck may mix them; the
//! two are told apart by the shape of the first token after the name (a
//! bare number starts the extension form, `x1=…` the corner-point form),
//! never by a deck-wide mode. Both produce the same
//! [`fasterhenry::plane::GroundPlane`], so `.hole`, `.contact`, `.equiv`
//! and endpoint landing behave identically whichever form declared the
//! plane.
//!
//! ## FastHenry corner-point form
//!
//! ```text
//! G<name> x1=… y1=… z1=… x2=… y2=… z2=… x3=… y3=… z3=…
//! +       thick=… seg1=… seg2=… [sigma=…|rho=…] [nhinc=…]
//! +       N<name> (x, y, z) …
//! +       hole rect (x1, y1, z1, x2, y2, z2) …
//! +       hole point (x, y, z) …
//! +       hole circle (x, y, z, r) …
//! +       contact rect (x, y, z, xwidth, ywidth, xcell, ycell) …
//! +       contact rect (x1, y1, z1, x2, y2, z2) …
//! +       contact decay_rect (x, y, z, xwidth, ywidth, xcell, ycell, xmaxcell, ymaxcell) …
//! +       contact point (x, y, z, xcell, ycell) …
//! +       contact line (x0, y0, z0, x1, y1, z1, xcell, ycell) …
//! +       contact equiv_rect N<name> (x, y, z, xwidth, ywidth) …
//! +       contact connection N<name> (x, y, z, xwidth, ywidth, ratio) …
//! ```
//!
//! * **The three corner points** give one corner and its two neighbours:
//!   `p1 → p2` is the first edge (cut into `seg1` cells), `p2 → p3` the
//!   second (`seg2` cells). This engine's plane is an axis-aligned
//!   rectangle parallel to the xy plane, so `z1 = z2 = z3` is required and
//!   each edge must run along x or y; a tilted or rotated plane is
//!   rejected by name rather than silently squared off.
//! * **`thick=`** is the thickness, and the corner points give the plane's
//!   **mid-thickness** surface — as a segment's nodes give its axis. The
//!   extension form instead names the **top** surface, so the same plane is
//!   `z1 = z_mid + thick/2` there.
//! * **`seg1`/`seg2`** become the background cell counts of this engine's
//!   cell-centre mesh (`nx`/`ny`). The mesh itself is this project's own
//!   PEEC discretization, not FastHenry's panel mesh (see the
//!   `fasterhenry::plane` module documentation), so equal cell counts mean
//!   equal resolution, not an identical node set. The `contact
//!   initial_grid` / `contact initial_mesh_grid` clauses below set this
//!   same pair, and a statement giving both is an error.
//! * **The plane's own coordinate system.** `p1` is its origin, the vector
//!   `p1 → p2` its x-direction and `p2 → p3` its y-direction — the frame the
//!   public description defines for a corner-point plane, and the frame it
//!   then states the `contact` clauses' cell sizes in. So every `x…`/`y…`
//!   pair of *lengths* in the clauses below is read along those two edges
//!   rather than along global x and y: `contact point`'s and `contact
//!   line`'s `xcell`/`ycell`, `contact decay_rect`'s `xwidth`/`ywidth`,
//!   `xcell`/`ycell` and `xmaxcell`/`ymaxcell`, `contact equiv_rect`'s and
//!   `contact connection`'s `xwidth`/`ywidth`, and — as its own entry below
//!   already says — `contact initial_grid`'s two counts. On a plane whose
//!   `p1 → p2` edge runs along global **y**, `xcell` is therefore the cell
//!   size across global y, and an anisotropic request is not transposed
//!   (issue #118; `docs/fasthenry-compat.md` records the evidence). It is
//!   the same convention `seg1`/`seg2` already follow, and for the usual
//!   plane — `p1 → p2` along global x, as every public example writes it —
//!   the two readings coincide.
//!
//!   **Coordinates are not relative.** Every `(x, y, z)` in the statement's
//!   body — in-plane nodes, holes, and the contact clauses' points, ends and
//!   rectangle centres — is an ordinary global deck coordinate, checked
//!   against this plane's footprint and slab. `p1` is the origin of a frame
//!   used for *directions* and for the `initial_mesh_grid` cell numbering,
//!   not an offset to add.
//! * **`sigma=`** is per deck unit exactly as elsewhere, and so is its
//!   reciprocal **`rho=`**, which this form takes on the statement too;
//!   naming both on one statement — continuation lines included — is an
//!   error, and giving neither falls back to the `.default` conductivity
//!   (`.default sigma=` or `.default rho=`). **`nhinc=`** cuts every bar of
//!   the plane into that many filaments through the thickness.
//! * **`N<name> (x, y, z)`** declares an in-plane node. It is an ordinary
//!   deck node that belongs to this plane: reference it from a segment or
//!   `.external`, or join it to a segment node with `.equiv`, and the
//!   connection lands on the nearest live cell-centre node of *that* plane
//!   — whichever side `.equiv` named first. Joining in-plane nodes of two
//!   *different* planes is rejected (connect the planes with a segment).
//! * **`hole rect (x1, y1, z1, x2, y2, z2)`** takes **two opposite
//!   corners** and maps onto [`fasterhenry::plane::Hole::Rect`]. The `z`
//!   coordinates are redundant for a plane parallel to xy, but are checked
//!   against the plane's own slab so a rectangle meant for another plane
//!   cannot land here silently — as they are for every `hole` / `contact`
//!   clause below.
//! * **`contact rect`** maps onto [`fasterhenry::plane::ContactRegion`],
//!   and is read in **either of two spellings, told apart by value count**
//!   (issue #95; the decision and its reasoning are in
//!   `docs/fasthenry-compat.md`):
//!     * **seven values**, `(x, y, z, xwidth, ywidth, xcell, ycell)` — the
//!       documented form, and the same centre-and-widths shape of argument
//!       list `contact point`, `contact line`, `contact decay_rect` and
//!       `contact equiv_rect` use: the rectangle's **centre**, its **full
//!       widths** about that centre, and the largest cell wanted
//!       **inside** it. It is exactly `contact decay_rect` without the
//!       outward limits, so it is read as one with both `maxcell`s at the
//!       "no limit" sentinel — every rule in the `decay_rect` bullet below
//!       (the per-axis cell count, the documented decay law, `cell` smaller
//!       than `width`) holds here too;
//!     * **six values**, `(x1, y1, z1, x2, y2, z2)` — two opposite corners
//!       as `hole rect` spells them, this reader's own extension, refined
//!       to 2 × 2 fine cells at ratio 2 (use the seven-value form, or
//!       `.contact`, to choose other values).
//!
//!   The two are one clause, not two: the seven-value form at `xcell =
//!   xwidth/2`, `ycell = ywidth/2` is ratio 2 over 2 × 2 cells, so
//!   `contact rect (5, 3, 0, 2, 2, 1, 1)` and `contact rect (4, 2, 0, 6,
//!   4, 0)` are the same region. Any other value count is a line-numbered
//!   error naming both spellings.
//! * **`contact decay_rect`** maps onto a
//!   [`fasterhenry::plane::ContactRegion`] too — the shape that bounds how
//!   coarse its outward decay may grow, which no other contact shape
//!   states. Its nine
//!   values are the rectangle's **centre** `(x, y, z)`, its **full
//!   widths** `xwidth`/`ywidth` about that centre, the largest cell wanted
//!   **inside** it (`xcell`/`ycell`) and the largest cell its outward
//!   decay may grow to (`xmaxcell`/`ymaxcell`, negative for no limit). All
//!   three pairs run along the plane's own axes (`p1 → p2` first), as above.
//!   Each axis is read on its own:
//!     * the fine cells are the fewest whose extent is no larger than that
//!       axis's `cell` — `ceil(width / cell)`, so 2 mm at `cell=1` is 2
//!       cells and at `cell=0.1` is 20;
//!     * the cells outside grow geometrically by `1/(1 − cell/width)` per
//!       cell, the documented decay law, until they reach the plane's
//!       background cell. `cell` must therefore be smaller than `width`
//!       (the ratio is otherwise not a ratio at all), and a `cell` equal
//!       to half the width is the familiar ratio 2;
//!     * a positive `maxcell` **finer** than the plane's own background
//!       cell is rejected by name: this engine's grading levels off *at*
//!       the background cell, so honouring a tighter limit would need a
//!       finer plane (raise `seg1`/`seg2`), not a quietly coarser mesh.
//!       A limit at or above the background cell never binds, and a
//!       negative one asks for none.
//!
//!   `decay_rect` differs from the seven-value `contact rect` above in
//!   exactly its last two values, the outward limits: `contact decay_rect
//!   (x, y, z, xwidth, ywidth, xcell, ycell, -1, -1)` *is* `contact rect
//!   (x, y, z, xwidth, ywidth, xcell, ycell)`. Only `hole rect` (and
//!   `contact rect`'s six-value extension) names a rectangle by **two
//!   opposite corners** — the same rectangle written the other way, so
//!   `contact decay_rect (5, 3, 0, 2, 2, 1, 1, -1, -1)` and `contact rect
//!   (4, 2, 0, 6, 4, 0)` are one region.
//! * **`hole point (x, y, z)`** and **`hole circle (x, y, z, r)`** map onto
//!   [`fasterhenry::plane::Hole::Point`] and
//!   [`fasterhenry::plane::Hole::Circle`] (issue #98). A point removes
//!   exactly the one cell whose own extent — edges included — contains it;
//!   a point that lands exactly on a shared cell edge or corner is the
//!   documented tie, and removes every cell touching it rather than
//!   guessing a single winner. A circle removes every cell whose centre
//!   lies at or inside its radius `r` (a centre exactly on the circle is
//!   removed — a closed boundary, unlike `hole rect`'s open one). Both take
//!   the shape's own `z`, checked against the plane's slab like every other
//!   clause here.
//! * **`hole user1` … `user7`** are rejected by name, on the statement's
//!   own line, and that rejection is **permanent** — not a shape awaiting
//!   implementation (issue #99; the decision and its reasoning are in
//!   `docs/fasthenry-compat.md`). A user-defined hole is a generator
//!   compiled into the tool itself: the deck carries a shape name and a
//!   value list whose meaning is stated nowhere in the deck, so there is no
//!   geometry here to read, and no amount of plane-model work would let
//!   this reader read one. The error names two alternatives instead: the
//!   declarative shapes above, and — for a shape none of them describe —
//!   building the plane through the library, where
//!   [`fasterhenry::plane::GroundPlane::mesh`] reports the exact cell
//!   centres the plane meshes to and a
//!   [`fasterhenry::plane::Hole::Point`] at each centre an arbitrary rule
//!   selects removes precisely those cells.
//! * **`contact point (x, y, z, xcell, ycell)`** and **`contact line (x0,
//!   y0, z0, x1, y1, z1, xcell, ycell)`** (issue #100) ask that every cell
//!   holding the point, or crossed by the line, be no larger than `xcell`
//!   along the plane's first edge and `ycell` along its second (the plane's
//!   own coordinate system, as above). Both map onto a
//!   [`fasterhenry::plane::ContactRegion`]:
//!     * the region is the locus's bounding box padded by half a requested
//!       cell on every side, cut into the fewest cells no larger than that
//!       cell — so a point is exactly **one** `xcell × ycell` cell centred
//!       on it (a via landing there snaps onto that cell's centre), a line
//!       along x is a one-cell-high strip reaching half a cell past each
//!       end, and a zero-length line is the point at its ends;
//!     * outside, the cells grade back to the background at ratio 2, the
//!       same default the six-value `contact rect` and `.contact` use:
//!       neither shape documents a decay of its own, and a tensor-product
//!       mesh has to return to the background cell somehow;
//!     * a requested cell at or above the plane's background cell is
//!       already met there (grading never grows a cell past the
//!       background cell), so that axis is clamped to the background cell
//!       rather than coarsened — and clamped means *untouched*: the band
//!       spans the whole plane on that axis at the plane's own cell
//!       count, so its edges are the background mesh's own whether or not
//!       the request's coordinate lands on a background grid line. A
//!       point or line met on both axes adds no region at all;
//!     * both ends are checked against the plane's slab and footprint, on
//!       the statement's own line.
//!
//!   A **diagonal** line refines its whole padded bounding box, not a
//!   band along the diagonal. That is the exact cost of this engine's
//!   tensor-product mesh rather than an approximation of the request: a
//!   refined band on one axis spans the plane on the other (see the
//!   `fasterhenry::plane` module documentation), so any refinement that
//!   covers the line — a chain of small rectangles along it included —
//!   produces these same x- and y-bands, and their crossing is the box. A
//!   line running `Lx` by `Ly` therefore costs about `(Lx/xcell) ·
//!   (Ly/ycell)` fine cells where a mesh free to follow it would need
//!   about `Lx/xcell + Ly/ycell`; an axis-aligned line costs nothing
//!   extra. The request itself is always honoured, overlapping regions
//!   included: merged bands keep their finest cell.
//! * **`contact trace (x0, y0, z0, x1, y1, z1, trace_width,
//!   scale_factor)`** (issue #110) refines the plane under a trace whose
//!   *projection* onto the plane runs from `(x0, y0)` to `(x1, y1)` —
//!   finely across the trace, not along it. Source: the public memo
//!   *Nonuniformly Discretized Reference Planes in FastHenry 3.0* (M.
//!   Kamon, 10 October 1996, section "Grouped contact utilities"), which
//!   describes the utility as a handful of `contact line` refinements
//!   keeping the cells under the trace no bigger than `trace_width/2`
//!   across it and the cells beside it no bigger than `trace_width`, and
//!   spells the lines out in a worked example along x. For a trace of length `L` and width `w` along
//!   x, it is exactly these five `contact line`s (along y, the same with
//!   the axes swapped):
//!     * at `y`, `y + w/2` and `y − w/2`, cells `(L, w/2)`;
//!     * at `y + 3w/2` and `y − 3w/2`, cells `(L, w)`.
//!
//!   The along-trace cell is the trace's own length, so once that reaches
//!   the background cell the trace adds no refinement along itself; each
//!   line then maps onto a region exactly as `contact line` does above
//!   (clamping included). The memo states that `scale_factor` has no
//!   effect on a trace parallel to x or y, so here it is only checked to
//!   be positive. The clause carries no `x…`/`y…` pair of its own — one
//!   `trace_width`, and a direction taken from its ends — and its expansion
//!   is symmetric in the two axes, so the plane-coordinate convention above
//!   leaves it alone: rotating the plane rotates the trace with it
//!   (issue #118). Both ends are checked against the plane's slab and
//!   footprint on the statement's own line; the side lines are not — the
//!   deck named the trace, not them — so beside a trace along the plane's
//!   edge they are clipped to the plane (or dropped, where they miss it)
//!   instead. A zero-length trace has no direction to refine across and is
//!   an error naming `contact point` instead.
//!
//!   A trace **not parallel to x or y** is rejected by name, on the
//!   statement's own line, and the rejection is a decision rather than
//!   deferred work (see `docs/fasthenry-compat.md`, "Decision: a diagonal
//!   `contact trace` is rejected"): the same memo says the cells under a
//!   diagonal trace are magnified by `scale_factor^|tan θ|` (θ from the x
//!   axis) and, a paragraph earlier, that the magnification runs from 1 to
//!   `scale_factor` as θ goes from 90° to 45° — which disagree for every
//!   angle past 45° — and it never says where a diagonal trace's side
//!   lines go or what along-trace cell its lines ask for. The error names
//!   `contact line` instead, with which a deck states the cell outright.
//! * **`contact equiv_rect N<name> (x, y, z, xwidth, ywidth)`** declares a
//!   *contact area*: the rectangle — centre and full widths, as
//!   `decay_rect` spells it, and like `decay_rect`'s widths measured along
//!   the plane's own axes — is tied to one node, named here, so a deck
//!   can `.equiv` an external node onto a whole landing pad instead of a
//!   point (issue #101). It maps onto
//!   [`fasterhenry::plane::Equipotential`]: every live cell centre inside
//!   the rectangle (its boundary included) shares a single node, the bars
//!   that ran between those cells are not built, and the bars crossing the
//!   patch's boundary end on that one node — see the
//!   [`fasterhenry::plane`] module documentation for exactly what the tie
//!   does to the mesh. The node itself is an ordinary deck node: name it
//!   from `.external`, from a segment, or join it with `.equiv`, and the
//!   whole joined set lands on the patch (an `.equiv` set holding both a
//!   contact area and a plain in-plane node of the same plane lands on the
//!   *area* — attaching to the nearest single cell instead would throw away
//!   the very thing the clause declared). Its name follows the same rule as
//!   an in-plane node's — it starts with `N` — and a rectangle catching no
//!   live cell centre is an error naming the plane, not a silent landing on
//!   a neighbouring cell.
//! * **`contact connection N<name> (x, y, z, xwidth, ywidth, ratio)`** is
//!   the documented shorthand for that same rectangle tied *and* refined:
//!   exactly a `contact equiv_rect` over it plus a `contact decay_rect`
//!   whose cells are `xwidth/ratio` and `ywidth/ratio` with no decay limit
//!   (issue #101). `ratio` must be greater than 1 — it is what the widths
//!   are divided by, so 1 or less asks for a cell as wide as the rectangle
//!   and grades nothing. Writing the two clauses out by hand gives exactly
//!   the same deck; the pairing matters because a tie is only as good as the
//!   mesh under it, and the `decay_rect` is what puts cells inside the
//!   rectangle for it to tie.
//! * **`contact initial_grid (n1, n2)`** sets the plane's *initial*
//!   discretization, and it is the statement `seg1`/`seg2` already make:
//!   `n1` cells along `p1 → p2` and `n2` along `p2 → p3` (issue #113). That
//!   is the equivalence the clause's public description states outright —
//!   `seg1=10 seg2=12` may be replaced by `contact initial_grid (10,12)` —
//!   and the same description's worked example only comes out square-celled
//!   read that way, so the reader maps it rather than guessing: a
//!   transposition would silently mesh every non-square plane the wrong way
//!   round. Because the two clauses say the same thing, a statement giving
//!   both is a line-numbered error rather than a race between them. The
//!   description also calls `n1` a count of *rows*; `docs/fasthenry-compat.md`
//!   records why that word does not overturn the mapping.
//! * **`contact initial_mesh_grid (n1, n2)`** is that same initial grid with
//!   the documented checkerboard of holes punched into it: "every cell that
//!   has an even value for both of its indices where the numbering is from
//!   the top left" — so, with the cells numbered from 1 at the plane's own
//!   origin (its `p1` corner) along each axis, every cell whose two indices
//!   are both even. Each such cell is cut by a
//!   [`fasterhenry::plane::Hole::Rect`] over the cell's own rectangle: on
//!   the initial grid that is the same cut a `hole point` at the centre
//!   makes, but the grid is *initial*, and a later `contact` clause may
//!   refine that region — the documented hole is the whole square, not
//!   whichever smaller cell the refinement leaves under the centre. An axis
//!   with a single cell has no even index and so no hole.
//! * Every other documented shape is **rejected by name** too, on the
//!   statement's own line: this engine's holes are rectangles, points or
//!   circles and its contacts rectangles, points, lines, axis-aligned
//!   traces or tied areas, and
//!   a shape whose semantics have not been stated and tested must not be
//!   quietly approximated by one of those. Unlike the user-defined holes,
//!   these have a public meaning, and each is rejected for its own stated
//!   reason:
//!     * `contact trace` **not parallel to x or y** — see its own entry
//!       above (issue #110: the public description of the diagonal case
//!       does not determine the cell size);
//! * **`contact circle` is rejected by name as well — and it is not a
//!   documented shape at all** (issue #109). The public description of the
//!   `contact` family names the simple refinement utilities `point`,
//!   `line`, `rect` and `decay_rect`, the contact-*area* utility
//!   `equiv_rect`, the grouped `connection` and `trace` built on them, and
//!   the `initial_grid`/`initial_mesh_grid` pair that pre-divides the
//!   plane; `circle` is a **hole** shape (`hole circle (x, y, z, r)`), not
//!   a contact one.
//!   So there is no argument list to pin, nothing in the deck says what a
//!   `contact circle`'s values would mean, and the rejection stands unless
//!   documentation for such a utility surfaces — it is not work awaiting an
//!   implementation here.
//!
//!   The geometry is not what is missing. A disc needs no new shape on this
//!   mesh: a refined x-band spans the plane in y and a y-band spans it in
//!   x, so *any* refinement that covers a disc produces the disc's
//!   bounding-square bands — exactly the argument the diagonal `contact
//!   line` above rests on. `contact decay_rect (x, y, z, 2r, 2r, xcell,
//!   ycell, xmaxcell, ymaxcell)` over that bounding square is therefore the
//!   whole of what a `contact circle` could mean here, at the same cost
//!   (the square's corners are refined too, the disc's `4/π` overhead) —
//!   widen each width by its own cell, as `contact point` pads by half a
//!   cell on each side, to keep the cells grazing the rim fine as well. The
//!   error says exactly that rather than leaving the reader to work it out.
//! * **`file=`** names the plane's nonuniform-discretization *hierarchy*
//!   file — an input, not a dump of the finished mesh — and `file=NONE`
//!   says there is no such file: the hierarchy is a single root cell,
//!   discretized at run time from the statement's own clauses (issue #122).
//!   That is the only case this reader ever has, so `file=NONE` is accepted
//!   as the no-op it is (matched without regard to case) and the documented
//!   `file=NONE contact initial_grid (n1, n2)` spelling reads as written; a
//!   *named* file is rejected by name, as the input this reader does not
//!   read.
//! * The remaining documented plane parameters are rejected by name too,
//!   each with the reason and the alternative: `rh` (plane filaments are
//!   uniform), `segwid1`/`segwid2` (bar widths follow
//!   the cells), and `relx`/`rely`/`relz` (name the in-plane nodes
//!   instead). Nothing on a `G` statement is silently ignored.
//!
//! # Semantics
//!
//! * **Units are mandatory.** A deck without an explicit `.units` line is
//!   rejected rather than silently assuming a default, so a missing unit can
//!   never scale a result by a factor of 10 or 100. The unit is
//!   case-insensitive: `km` (10³ m), `m`, `cm`, `mm`, `um` (10⁻⁶ m), `in`
//!   (0.0254 m) or `mils` (10⁻³ in; `mil` is a synonym). Lengths
//!   (coordinates, `w`, `h`) scale with the unit; **conductivity and
//!   resistivity are per deck unit** — `sigma=5.8e4` under `.units mm` is
//!   copper (5.8e4 S/mm = 5.8e7 S/m), and so is `rho=1.7241e-5`
//!   (Ω·mm = 1.7241e-8 Ω·m). `rho=r` is exactly `sigma=1/r`.
//! * **`nwinc`/`nhinc` and `rw`/`rh` cut the cross-section into
//!   filaments.** `nwinc` filaments go across the width and `nhinc` across
//!   the height (default 1 each). `rw` and `rh` are the ratio of adjacent
//!   filament extents along the width and the height respectively, each
//!   axis independently: every filament is `rw` (or `rh`) times as wide
//!   (or high) as its neighbour one step nearer the conductor surface, so
//!   the thinnest filaments line both surfaces and the fattest fill the
//!   core — `nwinc=5 rw=2` cuts the width in the proportions 1:2:4:2:1, and
//!   `nwinc=4 rw=3` in 1:3:3:1. This is the library's graded
//!   discretization (`fasterhenry::discretize_graded_per_axis`), per
//!   segment. A ratio must be at least 1 (1 is a uniform axis; a ratio
//!   below 1, coarsening *toward* the surface, is rejected rather than
//!   guessed at), and it has no effect on an axis with a single filament.
//!   **An omitted ratio means a uniform axis** (ratio 1): this reader's
//!   long-standing behaviour, kept so that a deck without `rw`/`rh` solves
//!   exactly as before. Set `rw`/`rh` explicitly (on the segment or in
//!   `.default`) to request a graded grid.
//! * **`wx`/`wy`/`wz` orient the cross-section.** Together they are a
//!   vector along the segment's *width* (dimensionless: it is not scaled by
//!   `.units`, and need be neither normalized nor exactly perpendicular to
//!   the segment — only its component perpendicular to the centreline
//!   counts). The height then runs along `length × width`. As soon as one
//!   component is given (on the segment or in `.default`), the vector is
//!   explicit and any component given nowhere is 0; each component falls
//!   back to its own `.default`. A zero vector, or one parallel to the
//!   segment, fixes no orientation and is an error on the `E` line. Without
//!   any of the three, the default rule applies: the width lies in the x–y
//!   plane, perpendicular to the segment (`+x` for a segment along z) —
//!   see `fasterhenry::Segment::basis`.
//! * **`.freq fmin fmax ndec`** samples `ndec` points per decade,
//!   log-spaced: `f(k) = fmin · 10^(k/ndec)` for `k = 0 … n−1`, with
//!   `n = floor(ndec · log10(fmax/fmin)) + 1`; `fmin` is always the first
//!   point and `fmax` is reached when the decades divide evenly.
//!   `fmin = fmax` (any `ndec`) is the single-frequency case; `fmin = 0` is
//!   allowed only there (the DC solve).
//! * **`.equiv a b …`** makes every later name an alias of `a`: every
//!   reference — declared before or after the directive — resolves to `a`,
//!   and the aliased nodes do not appear in the resulting geometry. One
//!   exception to "the first name wins": if any node in the joined set is an
//!   in-plane node, the whole set lands on that plane (see above), because
//!   joining a via's node to a plane node is what wires a deck into a plane
//!   and the result must not depend on the argument order.
//! * **`.couples` truncates, and defaults to truncating nothing.** Without a
//!   `.couples` line — and with `.couples all` — every pair of conductors is
//!   coupled, exactly as before. A `.couples g1 g2 …` line switches the deck
//!   into truncating mode and declares the listed groups mutually coupled (a
//!   clique: every listed group with every other). Every other pair of groups
//!   has its mutual inductance dropped without ever being computed. Segments
//!   take their group from `group=<name>` on the `E` line (case-sensitive,
//!   like node names); untagged segments and ground planes share one default
//!   group. Naming a group no segment carries is an error, not a no-op.
//!   Truncation is a physical approximation — see the `fasterhenry::coupling`
//!   module documentation for when it is safe; groups closer than their own
//!   extent are reported on stderr.
//! * **Planes connect by landing**: a segment or port endpoint that lies
//!   within a plane's footprint and depth is snapped to the nearest live
//!   cell-centre node, wiring the segment into the plane mesh. Declare the
//!   plane before or after the segment — the assembly is order-independent.
//!   A landing that is a *pad* rather than a point is a contact **area**
//!   instead: `contact equiv_rect` (or its `contact connection` shorthand)
//!   names a rectangle of the plane, ties every cell inside it to one node,
//!   and lets the deck land on that (see the corner-point form above).
//! * **`.contact` grades the mesh under a landing.** A plane's `nx`/`ny`
//!   are its *background* resolution; every `.contact` rectangle is cut
//!   into `nx × ny` fine cells of its own (default `2 × 2`) and the cells
//!   outside it grow by `ratio` each (default `2`, `1` meaning no decay)
//!   until they reach the background cell. Put one under each via landing
//!   to resolve the current crowding there without paying for a fine mesh
//!   across the whole plane; grading is per axis, so a region refines a
//!   band across the plane in each direction (see the
//!   `fasterhenry::plane` module documentation for the cell-count
//!   formula). `.hole` and `.contact` both name a plane declared earlier
//!   in the deck, with the leading `G` optional.
//! * Everything else is rejected with an error carrying the line number.
//!   Deck-level problems that belong to no single line (missing `.units`, a
//!   ground-plane assembly failure) report line 0; a segment the geometry
//!   rejects is reported on its own `E` line.

use std::collections::HashMap;

use fasterhenry::coupling::Coupling;
use fasterhenry::geometry::{Geometry, Node, NodeId, Segment, SegmentDef, SegmentError};
use fasterhenry::mesh::Port;
use fasterhenry::plane::{ContactRegion, Equipotential, GroundPlane, Hole};
use fasterhenry::solve::{AxisGrading, Discretization, Subdivision};

/// A parsed `.inp` deck: everything [`fasterhenry::solve::solve`] needs.
#[derive(Clone, Debug, PartialEq)]
pub struct Deck {
    /// The deck's title, if any: its last `.title` directive, else (in
    /// [`ParseOptions::fasthenry_compat`] mode) its non-blank first line.
    pub title: Option<String>,
    /// Nodes and segments, positions in metres.
    pub geometry: Geometry,
    /// Ports from `.external`, in deck order.
    pub ports: Vec<Port>,
    /// Filament subdivisions ([`Discretization::Uniform`] when every segment
    /// agrees, [`Discretization::PerSegment`] as soon as one differs, and
    /// [`Discretization::PerSegmentGraded`] as soon as any segment grades an
    /// axis with `rw`/`rh`).
    pub discretization: Discretization,
    /// Segment groups from `group=` and the couplings from `.couples`;
    /// [`Coupling::all_pairs`] for a deck that declares neither.
    pub coupling: Coupling,
    /// Frequencies in hertz from `.freq`.
    pub frequencies: Vec<f64>,
}

/// An `E` line pending assembly: its node slots, cross-section, optional
/// width direction, and the line it was declared on (for errors).
#[derive(Clone, Copy, Debug)]
struct SegmentSpec {
    a: usize,
    b: usize,
    width: f64,
    height: f64,
    sigma: f64,
    width_dir: Option<[f64; 3]>,
    line: usize,
}

/// A `G` line pending assembly: the plane, the name `.hole` / `.contact`
/// refers to, and the filament count its bars carry through the thickness.
#[derive(Clone, Debug)]
struct PlaneSpec {
    name: String,
    plane: GroundPlane,
    /// Filaments across each plane bar's thickness (`nhinc=`; 1 by default).
    nhinc: usize,
}

/// An in-plane node awaiting assembly: the declared node slot it occupies,
/// the plane it belongs to, the equipotential patch it names (for a node a
/// `contact equiv_rect` / `contact connection` clause declared), its
/// declared position in metres, and the line it was declared on.
#[derive(Clone, Copy, Debug)]
struct PlaneNodeSpec {
    slot: usize,
    plane: usize,
    equipotential: Option<usize>,
    position: [f64; 3],
    line: usize,
}

/// The declared plane a `.hole` / `.contact` line names; the leading `G` is
/// optional (`.hole Gp` and `.hole p` both find `Gp`).
fn plane_named<'a>(
    planes: &'a mut [PlaneSpec],
    name: &str,
    directive: &str,
    line: usize,
) -> Result<&'a mut PlaneSpec, ParseError> {
    let wanted = name.strip_prefix(['g', 'G']).unwrap_or(name);
    planes
        .iter_mut()
        .find(|spec| spec.name[1..] == *wanted || spec.name == name)
        .ok_or_else(|| {
            err(
                line,
                format!("'{directive}' names unknown ground plane '{name}'"),
            )
        })
}

/// A parse error: what went wrong, and on which line (`0` = deck-level).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("line {line}: {message}")]
pub struct ParseError {
    /// The 1-based line number the error was detected on, or 0 for
    /// deck-level problems.
    pub line: usize,
    /// What went wrong.
    pub message: String,
}

fn err(line: usize, message: impl Into<String>) -> ParseError {
    ParseError {
        line,
        message: message.into(),
    }
}

/// The `.units` directive's length unit, as a factor to metres.
/// Case-insensitive, like every other directive; `mil` is accepted as a
/// synonym of the documented `mils`.
fn unit_factor(unit: &str, line: usize) -> Result<f64, ParseError> {
    match unit.to_ascii_lowercase().as_str() {
        "km" => Ok(1e3),
        "m" => Ok(1.0),
        "cm" => Ok(1e-2),
        "mm" => Ok(1e-3),
        "um" => Ok(1e-6),
        "in" => Ok(0.0254),
        "mils" | "mil" => Ok(25.4e-6),
        _ => Err(err(
            line,
            format!("unknown length unit '{unit}' (supported: {UNITS})"),
        )),
    }
}

/// The accepted `.units` spellings, for error messages.
const UNITS: &str = "km, m, cm, mm, um, in, mils";

/// Rejects a line that gives its conductivity both ways: `sigma=` and
/// `rho=` among the same line's `<field>=<value>` tokens.
fn one_conductivity(fields: &[&str], line: usize) -> Result<(), ParseError> {
    let has = |wanted: &str| {
        fields.iter().any(|token| {
            token
                .split_once('=')
                .is_some_and(|(key, _)| key.eq_ignore_ascii_case(wanted))
        })
    };
    if has("sigma") && has("rho") {
        return Err(err(
            line,
            "both sigma= and rho= on one line: give the conductivity one way (sigma = 1/rho)",
        ));
    }
    Ok(())
}

/// Per-line field defaults set by `.default`; lengths are already scaled to
/// metres when stored, sigma (given directly or as `rho`) to S/m. Ratios and
/// width-direction components are dimensionless.
#[derive(Clone, Copy, Debug, Default)]
struct Defaults {
    x: Option<f64>,
    y: Option<f64>,
    z: Option<f64>,
    w: Option<f64>,
    h: Option<f64>,
    nwinc: Option<usize>,
    nhinc: Option<usize>,
    rw: Option<f64>,
    rh: Option<f64>,
    width_dir: [Option<f64>; 3],
    sigma: Option<f64>,
}

/// The fields `.default` accepts, for error messages.
const DEFAULT_FIELDS: &str = "x, y, z, w, h, nwinc, nhinc, rw, rh, wx, wy, wz, sigma, rho";

/// The fields an `E` line accepts, for error messages.
const SEGMENT_FIELDS: &str = "w, h, nwinc, nhinc, rw, rh, wx, wy, wz, sigma, rho, group";

/// Index of a width-direction component key (`wx`, `wy`, `wz`).
fn width_dir_axis(key: &str) -> Option<usize> {
    match key {
        "wx" => Some(0),
        "wy" => Some(1),
        "wz" => Some(2),
        _ => None,
    }
}

/// A filament grading ratio (`rw`/`rh`): the ratio of adjacent filament
/// extents, coarsening from the surface inward, so at least 1.
fn check_ratio(key: &str, value: f64, line: usize) -> Result<f64, ParseError> {
    if value <= 0.0 {
        return Err(err(
            line,
            format!("'{key}' must be positive (a ratio of adjacent filament extents), got {value}"),
        ));
    }
    if value < 1.0 {
        return Err(err(
            line,
            format!(
                "'{key}' must be ≥ 1, got {value}: filaments grow from the surface inward by this ratio (1 is uniform); coarsening toward the surface is not supported"
            ),
        ));
    }
    Ok(value)
}

/// The explicit width direction of a segment, if any component is set on the
/// line or in `.default`; unset components are 0. A zero vector is an error.
fn resolve_width_dir(
    components: [Option<f64>; 3],
    segment: &str,
    line: usize,
) -> Result<Option<[f64; 3]>, ParseError> {
    if components.iter().all(Option::is_none) {
        return Ok(None);
    }
    let direction = components.map(|component| component.unwrap_or(0.0));
    if direction.iter().all(|&component| component == 0.0) {
        return Err(err(
            line,
            format!(
                "segment '{segment}': width direction (wx, wy, wz) = (0, 0, 0) is the zero vector and fixes no orientation"
            ),
        ));
    }
    Ok(Some(direction))
}

/// A `<field>=<value>` token, keys lowercased.
fn parse_field(token: &str, line: usize) -> Result<(String, String), ParseError> {
    let (key, value) = token.split_once('=').ok_or_else(|| {
        err(
            line,
            format!("expected <field>=<value>, got '{token}' (supported: {DEFAULT_FIELDS})"),
        )
    })?;
    if key.is_empty() || value.is_empty() {
        return Err(err(line, format!("empty side of '{token}'")));
    }
    Ok((key.to_ascii_lowercase(), value.to_string()))
}

fn parse_number(text: &str, line: usize) -> Result<f64, ParseError> {
    let value = text
        .parse::<f64>()
        .map_err(|_| err(line, format!("'{text}' is not a number")))?;
    if !value.is_finite() {
        return Err(err(line, format!("'{text}' is not finite")));
    }
    Ok(value)
}

fn parse_count(text: &str, what: &str, line: usize) -> Result<usize, ParseError> {
    let value = text.parse::<usize>().map_err(|_| {
        err(
            line,
            format!("'{text}' is not a valid {what} (a whole number ≥ 1)"),
        )
    })?;
    if value < 1 {
        return Err(err(line, format!("{what} must be ≥ 1, got {value}")));
    }
    Ok(value)
}

/// The value side of a `<field>=<value>` pair: a length (scaled), a
/// conductivity (per deck unit, converted to S/m), a resistivity (per deck
/// unit, converted to a conductivity in S/m), a count, a dimensionless
/// ratio, or a (dimensionless) width-direction component.
fn parse_value(key: &str, value: &str, unit: f64, line: usize) -> Result<f64, ParseError> {
    match key {
        "sigma" => Ok(parse_number(value, line)? / unit),
        "rho" => {
            let rho = parse_number(value, line)?;
            if rho <= 0.0 {
                return Err(err(
                    line,
                    format!("rho (resistivity) must be > 0, got {value}"),
                ));
            }
            // `1/rho` first, then the unit, exactly as `sigma = 1/rho`
            // would be: the two spellings give bit-identical conductivities
            // whenever the deck's sigma is the correctly rounded 1/rho.
            Ok(1.0 / rho / unit)
        }
        "nwinc" | "nhinc" | "nx" | "ny" => Ok(parse_count(value, key, line)? as f64),
        "ratio" | "wx" | "wy" | "wz" => parse_number(value, line),
        "rw" | "rh" => check_ratio(key, parse_number(value, line)?, line),
        _ => Ok(parse_number(value, line)? * unit),
    }
}

/// Sets one default field, rejecting unknown keys with the field list.
fn set_default(
    defaults: &mut Defaults,
    key: &str,
    value: f64,
    line: usize,
) -> Result<(), ParseError> {
    match key {
        "x" => defaults.x = Some(value),
        "y" => defaults.y = Some(value),
        "z" => defaults.z = Some(value),
        "w" => defaults.w = Some(value.abs()),
        "h" => defaults.h = Some(value.abs()),
        "nwinc" => defaults.nwinc = Some(value as usize),
        "nhinc" => defaults.nhinc = Some(value as usize),
        "rw" => defaults.rw = Some(value),
        "rh" => defaults.rh = Some(value),
        "wx" | "wy" | "wz" => {
            let axis = width_dir_axis(key).expect("matched a width-direction key");
            defaults.width_dir[axis] = Some(value);
        }
        // Already converted to S/m by parse_value either way.
        "sigma" | "rho" => defaults.sigma = Some(value),
        other => {
            return Err(err(
                line,
                format!("unknown field '{other}' (supported: {DEFAULT_FIELDS})"),
            ));
        }
    }
    Ok(())
}

/// One item in the body of a FastHenry-form `G` ground-plane statement.
#[derive(Clone, Debug)]
enum PlaneItem {
    /// `<key>=<value>`, the key lowercased.
    Field(String, String),
    /// `N<name> (x, y, z)`: an in-plane node declaration.
    Node(String, Vec<String>),
    /// `hole <shape> (…)` or `contact <shape> (…)`, both lowercased — with
    /// the node name the `contact equiv_rect` / `contact connection` forms
    /// carry between the shape word and the value list (case-sensitive,
    /// like every other node name; `None` when the clause has none).
    Clause {
        kind: String,
        shape: String,
        name: Option<String>,
        values: Vec<String>,
    },
}

fn skip_space(chars: &[char], at: &mut usize) {
    while chars.get(*at).is_some_and(|c| c.is_whitespace()) {
        *at += 1;
    }
}

/// A bare word: everything up to whitespace or one of `= ( ) ,`.
fn read_word(chars: &[char], at: &mut usize) -> String {
    let start = *at;
    while chars
        .get(*at)
        .is_some_and(|c| !c.is_whitespace() && !matches!(c, '=' | '(' | ')' | ','))
    {
        *at += 1;
    }
    chars[start..*at].iter().collect()
}

/// A parenthesised value list, separated by commas and/or whitespace.
fn read_list(
    chars: &[char],
    at: &mut usize,
    what: &str,
    line: usize,
) -> Result<Vec<String>, ParseError> {
    skip_space(chars, at);
    if chars.get(*at) != Some(&'(') {
        return Err(err(
            line,
            format!("{what} needs a parenthesised value list, as in '(x, y, z)'"),
        ));
    }
    *at += 1;
    let mut values = Vec::new();
    loop {
        skip_space(chars, at);
        match chars.get(*at) {
            None => return Err(err(line, format!("{what} has an unterminated '('"))),
            Some(&')') => {
                *at += 1;
                return Ok(values);
            }
            Some(&',') => *at += 1,
            Some(&other) => {
                let word = read_word(chars, at);
                if word.is_empty() {
                    return Err(err(
                        line,
                        format!("{what}: unexpected '{other}' inside (…)"),
                    ));
                }
                values.push(word);
            }
        }
    }
}

/// Splits the body of a FastHenry-form `G` statement into its items.
///
/// Deliberately tolerant of the layouts the documented format allows:
/// whitespace around `=`, and value lists in parentheses separated by
/// commas, whitespace, or both. Continuation (`+`) lines have already been
/// folded into one token list by [`parse`], so the whole statement arrives
/// here as one string carrying one line number.
fn scan_plane_items(body: &str, line: usize) -> Result<Vec<PlaneItem>, ParseError> {
    let chars: Vec<char> = body.chars().collect();
    let mut at = 0usize;
    let mut items = Vec::new();
    loop {
        skip_space(&chars, &mut at);
        let Some(&here) = chars.get(at) else {
            return Ok(items);
        };
        let word = read_word(&chars, &mut at);
        if word.is_empty() {
            return Err(err(
                line,
                format!("unexpected '{here}' in ground-plane statement"),
            ));
        }
        if word.eq_ignore_ascii_case("hole") || word.eq_ignore_ascii_case("contact") {
            skip_space(&chars, &mut at);
            let shape = read_word(&chars, &mut at);
            if shape.is_empty() {
                return Err(err(
                    line,
                    format!(
                        "'{word}' needs a shape name, as in '{} rect (…)'",
                        word.to_ascii_lowercase()
                    ),
                ));
            }
            let what = format!(
                "'{} {}'",
                word.to_ascii_lowercase(),
                shape.to_ascii_lowercase()
            );
            // A node name may stand between the shape word and the value
            // list (`contact equiv_rect N<name> (…)`). It is read for every
            // clause and rejected by the ones that take none, so a stray
            // name is named in the error rather than reported as a missing
            // value list.
            skip_space(&chars, &mut at);
            let name = match chars.get(at) {
                Some(&'(') => None,
                _ => {
                    let name = read_word(&chars, &mut at);
                    (!name.is_empty()).then_some(name)
                }
            };
            let values = read_list(&chars, &mut at, &what, line)?;
            items.push(PlaneItem::Clause {
                kind: word.to_ascii_lowercase(),
                shape: shape.to_ascii_lowercase(),
                name,
                values,
            });
            continue;
        }
        let mut probe = at;
        skip_space(&chars, &mut probe);
        match chars.get(probe) {
            Some(&'=') => {
                at = probe + 1;
                skip_space(&chars, &mut at);
                let value = read_word(&chars, &mut at);
                if value.is_empty() {
                    return Err(err(line, format!("'{word}=' has no value")));
                }
                items.push(PlaneItem::Field(word.to_ascii_lowercase(), value));
            }
            Some(&'(') => {
                at = probe;
                let what = format!("in-plane node '{word}'");
                let values = read_list(&chars, &mut at, &what, line)?;
                items.push(PlaneItem::Node(word, values));
            }
            _ => {
                return Err(err(
                    line,
                    format!(
                        "'{word}' is not a ground-plane parameter (expected <field>=<value>, 'N<name> (x, y, z)', 'hole <shape> (…)' or 'contact <shape> (…)')"
                    ),
                ));
            }
        }
    }
}

/// An in-plane node declared inside a `G` statement: its name, its position
/// (m), and — for a node a `contact equiv_rect` / `contact connection`
/// clause declared — the index of the [`GroundPlane::equipotentials`] entry
/// it names, which is what it attaches to instead of the nearest cell.
type PlaneNode = (String, [f64; 3], Option<usize>);

/// The `(x, y, z)` of a clause or node value list.
fn triple(values: &[String], what: &str, unit: f64, line: usize) -> Result<[f64; 3], ParseError> {
    if values.len() != 3 {
        return Err(err(
            line,
            format!("{what} takes 3 values (x, y, z), got {}", values.len()),
        ));
    }
    Ok([
        parse_number(&values[0], line)? * unit,
        parse_number(&values[1], line)? * unit,
        parse_number(&values[2], line)? * unit,
    ])
}

/// Whether `shape` names one of the seven documented **user-defined** hole
/// generators, `user1` … `user7` — rejected permanently and by name rather
/// than through the generic "unknown hole shape" arm, because they are not
/// an unimplemented shape but a shape the deck cannot state at all. See the
/// [module documentation](self) and `docs/fasthenry-compat.md` (issue #99).
///
/// `shape` arrives lowercased from [`scan_plane_items`].
fn is_user_hole(shape: &str) -> bool {
    matches!(
        shape.strip_prefix("user"),
        Some("1" | "2" | "3" | "4" | "5" | "6" | "7")
    )
}

/// The two opposite corners of a `rect` clause: `(x1, y1, z1, x2, y2, z2)`.
fn rect_corners(
    values: &[String],
    what: &str,
    unit: f64,
    line: usize,
) -> Result<[[f64; 3]; 2], ParseError> {
    if values.len() != 6 {
        return Err(err(
            line,
            format!(
                "{what} takes 6 values (x1, y1, z1, x2, y2, z2), got {}",
                values.len()
            ),
        ));
    }
    Ok([
        triple(&values[..3], what, unit, line)?,
        triple(&values[3..], what, unit, line)?,
    ])
}

/// A `contact equiv_rect` / `contact connection` clause: the node it names
/// and the rectangle that node ties, kept raw until the plane's own
/// geometry is known (its `z` is checked against the plane's slab).
#[derive(Clone, Debug)]
struct EquivRect {
    /// Which clause declared it, quoted for error messages.
    what: String,
    /// The node name the clause carries, as written (case-sensitive).
    name: String,
    /// The rectangle's centre `(x, y, z)`, metres — global coordinates.
    centre: [f64; 3],
    /// Its full widths `(xwidth, ywidth)`, metres, in the **plane's own**
    /// coordinate system: index 0 along `p1 → p2`, index 1 along `p2 → p3`.
    /// [`PlaneFrame::plane_pair`] maps the pair onto the global axes.
    widths: [f64; 2],
}

/// The five leading values of a `contact equiv_rect` / `contact connection`
/// clause: `(x, y, z, xwidth, ywidth)` — the rectangle's centre and its full
/// widths about that centre, the same centre-and-widths spelling
/// `contact decay_rect` uses. `values` may be longer (a `connection`'s
/// trailing `ratio`); the caller checks its own arity first.
fn equiv_rect_values(
    values: &[String],
    what: &str,
    unit: f64,
    line: usize,
) -> Result<([f64; 3], [f64; 2]), ParseError> {
    let centre = triple(&values[..3], what, unit, line)?;
    let mut widths = [0.0f64; 2];
    for axis in 0..2 {
        let name = ['x', 'y'][axis];
        widths[axis] = parse_number(&values[3 + axis], line)? * unit;
        if widths[axis] <= 0.0 {
            return Err(err(
                line,
                format!(
                    "{what}: {name}width={} must be > 0 — it is the rectangle's full width about its centre, not a corner",
                    widths[axis]
                ),
            ));
        }
    }
    Ok((centre, widths))
}

/// A `contact decay_rect` clause, kept raw until the plane's own geometry
/// — and so its background cell — is known.
#[derive(Clone, Copy, Debug)]
struct DecayRect {
    /// The rectangle's centre `(x, y, z)`, metres — global coordinates.
    centre: [f64; 3],
    /// Its full widths `(xwidth, ywidth)`, metres.
    widths: [f64; 2],
    /// The largest cell wanted inside it, `(xcell, ycell)`, metres.
    cell: [f64; 2],
    /// The largest cell the outward decay may grow to, `(xmaxcell,
    /// ymaxcell)`, metres; `None` where the deck gave a negative value (no
    /// limit).
    limit: [Option<f64>; 2],
    // All three pairs above are in the **plane's own** coordinate system:
    // index 0 along `p1 → p2`, index 1 along `p2 → p3`. See
    // [`PlaneFrame::plane_pair`], which maps them onto the global axes.
}

/// The seven values of the **documented** `contact rect` form:
/// `(x, y, z, xwidth, ywidth, xcell, ycell)` — the rectangle's centre, its
/// full widths about that centre, and the largest cell wanted inside it.
/// That is exactly `contact decay_rect` without the outward limits, so it
/// *is* a [`DecayRect`] with both `maxcell`s at the "no limit" sentinel:
/// the cells outside grade by the documented law all the way back to the
/// plane's background cell. Issue #95; see the [module documentation](self).
///
/// `values` may be longer — [`decay_rect_values`] reads its own two
/// trailing limits over the top of this — so the caller checks its own
/// arity first.
fn contact_rect_values(
    values: &[String],
    what: &str,
    unit: f64,
    line: usize,
) -> Result<DecayRect, ParseError> {
    let centre = triple(&values[..3], what, unit, line)?;
    let mut widths = [0.0f64; 2];
    let mut cell = [0.0f64; 2];
    for axis in 0..2 {
        let name = ['x', 'y'][axis];
        widths[axis] = parse_number(&values[3 + axis], line)? * unit;
        // `parse_number` has already rejected a non-finite value, so a
        // plain comparison is total here.
        if widths[axis] <= 0.0 {
            return Err(err(
                line,
                format!(
                    "{what}: {name}width={} must be > 0 — it is the rectangle's full width about its centre, not a corner",
                    widths[axis]
                ),
            ));
        }
        cell[axis] = parse_number(&values[5 + axis], line)? * unit;
        if !(cell[axis] > 0.0 && cell[axis] < widths[axis]) {
            return Err(err(
                line,
                format!(
                    "{what}: {name}cell={} metres must be > 0 and smaller than {name}width={} — the decay ratio is 1/(1 − {name}cell/{name}width), so a cell as wide as the rectangle grades nothing",
                    cell[axis], widths[axis]
                ),
            ));
        }
    }
    Ok(DecayRect {
        centre,
        widths,
        cell,
        limit: [None, None],
    })
}

/// The nine values of a `contact decay_rect` clause:
/// `(x, y, z, xwidth, ywidth, xcell, ycell, xmaxcell, ymaxcell)` — the
/// seven of the documented `contact rect` above, plus the largest cell the
/// outward decay may grow to (negative for no limit). See the
/// [module documentation](self).
fn decay_rect_values(
    values: &[String],
    what: &str,
    unit: f64,
    line: usize,
) -> Result<DecayRect, ParseError> {
    if values.len() != 9 {
        return Err(err(
            line,
            format!(
                "{what} takes 9 values (x, y, z, xwidth, ywidth, xcell, ycell, xmaxcell, ymaxcell), got {}",
                values.len()
            ),
        ));
    }
    let mut rect = contact_rect_values(values, what, unit, line)?;
    for axis in 0..2 {
        let name = ['x', 'y'][axis];
        let raw = parse_number(&values[7 + axis], line)?;
        rect.limit[axis] = if raw < 0.0 {
            None
        } else if raw > 0.0 {
            Some(raw * unit)
        } else {
            return Err(err(
                line,
                format!(
                    "{what}: {name}maxcell=0 is neither a cell size nor the 'no limit' sentinel; give a positive cell size, or a negative value to decay all the way to the plane's background cell"
                ),
            ));
        };
    }
    Ok(rect)
}

/// A `contact point` or `contact line` clause, kept raw until the plane's
/// own geometry — and so its background cell — is known. A point is the
/// line whose two ends coincide.
#[derive(Clone, Copy, Debug)]
struct RefineLine {
    /// The locus's two ends `(x, y, z)`, metres (equal for a point) —
    /// global coordinates.
    ends: [[f64; 3]; 2],
    /// The largest cell wanted along it, `(xcell, ycell)`, metres, in the
    /// **plane's own** coordinate system: index 0 along `p1 → p2`, index 1
    /// along `p2 → p3`. [`PlaneFrame::plane_pair`] maps the pair onto the
    /// global axes ([`PlaneFrame::refine_region`] applies it).
    cell: [f64; 2],
}

/// The values of a `contact point (x, y, z, xcell, ycell)` or `contact line
/// (x0, y0, z0, x1, y1, z1, xcell, ycell)` clause. See the [module
/// documentation](self).
fn refine_line_values(
    values: &[String],
    line_shape: bool,
    what: &str,
    unit: f64,
    line: usize,
) -> Result<RefineLine, ParseError> {
    let (count, form) = if line_shape {
        (8, "x0, y0, z0, x1, y1, z1, xcell, ycell")
    } else {
        (5, "x, y, z, xcell, ycell")
    };
    if values.len() != count {
        return Err(err(
            line,
            format!("{what} takes {count} values ({form}), got {}", values.len()),
        ));
    }
    let first = triple(&values[..3], what, unit, line)?;
    let (ends, rest) = if line_shape {
        (
            [first, triple(&values[3..6], what, unit, line)?],
            &values[6..],
        )
    } else {
        ([first, first], &values[3..])
    };
    let mut cell = [0.0f64; 2];
    for axis in 0..2 {
        cell[axis] = parse_number(&rest[axis], line)? * unit;
        // `parse_number` has already rejected a non-finite value.
        if cell[axis] <= 0.0 {
            return Err(err(
                line,
                format!(
                    "{what}: {}cell={} metres must be > 0 — it is the largest cell wanted along the {}",
                    ['x', 'y'][axis],
                    cell[axis],
                    if line_shape { "line" } else { "point" }
                ),
            ));
        }
    }
    Ok(RefineLine { ends, cell })
}

/// A `contact trace` clause parallel to x or y, kept raw until the plane's
/// own geometry is known (its ends are checked against the slab and the
/// footprint there). See the [module documentation](self).
#[derive(Clone, Copy, Debug)]
struct RefineTrace {
    /// The trace projection's two ends `(x, y, z)`, metres.
    ends: [[f64; 3]; 2],
    /// `trace_width`, metres.
    width: f64,
    /// The axis the trace runs along: 0 for x, 1 for y.
    along: usize,
}

impl RefineTrace {
    /// The documented expansion into `contact line`s (the public memo
    /// *Nonuniformly Discretized Reference Planes in FastHenry 3.0*, M.
    /// Kamon, 1996, section "Grouped contact utilities"): its worked
    /// example of a trace along x expands to five lines along the trace —
    /// at offsets `0` and `±w/2` across it with cells `(L, w/2)`, and at
    /// `±3w/2` with cells `(L, w)`, where `w` is `trace_width` and `L` the
    /// trace's length, so there is no refinement along it. The memo states
    /// that `scale_factor` has no effect on such a trace.
    ///
    /// The lines' *ends* are global, like the trace's own; their cell pairs
    /// are in the plane's own coordinate system, like every other
    /// `RefineLine`'s, so `axis1` (the global axis the plane's `p1 → p2`
    /// edge runs along) is needed to say which of the pair is the
    /// along-trace one. The expansion itself is symmetric in the two axes,
    /// so a rotated plane changes nothing about the mesh it asks for
    /// (issue #118).
    fn lines(&self, axis1: usize) -> [RefineLine; 5] {
        let across = 1 - self.along;
        let length = (self.ends[1][self.along] - self.ends[0][self.along]).abs();
        // Which half of a plane-coordinate pair the trace runs along.
        let along_slot = usize::from(self.along != axis1);
        let w = self.width;
        [
            (0.0, w / 2.0),
            (w / 2.0, w / 2.0),
            (-w / 2.0, w / 2.0),
            (1.5 * w, w),
            (-1.5 * w, w),
        ]
        .map(|(offset, across_cell)| {
            let mut ends = self.ends;
            for end in &mut ends {
                end[across] += offset;
            }
            let mut cell = [0.0; 2];
            cell[along_slot] = length;
            cell[1 - along_slot] = across_cell;
            RefineLine { ends, cell }
        })
    }
}

/// The values of a `contact trace (x0, y0, z0, x1, y1, z1, trace_width,
/// scale_factor)` clause. A trace not parallel to x or y is rejected here,
/// by name: see the [module documentation](self) for why.
fn refine_trace_values(
    values: &[String],
    what: &str,
    unit: f64,
    line: usize,
) -> Result<RefineTrace, ParseError> {
    if values.len() != 8 {
        return Err(err(
            line,
            format!(
                "{what} takes 8 values (x0, y0, z0, x1, y1, z1, trace_width, scale_factor), got {}",
                values.len()
            ),
        ));
    }
    let ends = [
        triple(&values[..3], what, unit, line)?,
        triple(&values[3..6], what, unit, line)?,
    ];
    let width = parse_number(&values[6], line)? * unit;
    if width <= 0.0 {
        return Err(err(
            line,
            format!("{what}: trace_width={width} metres must be > 0"),
        ));
    }
    // Dimensionless: a magnification of the cells under a diagonal trace.
    let scale = parse_number(&values[7], line)?;
    if scale <= 0.0 {
        return Err(err(
            line,
            format!(
                "{what}: scale_factor={scale} must be > 0 — it magnifies the cells under the trace"
            ),
        ));
    }
    let dx = (ends[1][0] - ends[0][0]).abs();
    let dy = (ends[1][1] - ends[0][1]).abs();
    if dx == 0.0 && dy == 0.0 {
        return Err(err(
            line,
            format!(
                "{what} has zero length, so it has no direction to refine across; use 'contact point (x, y, z, xcell, ycell)' for a point"
            ),
        ));
    }
    let along = if dy <= 1e-9 * dx {
        0
    } else if dx <= 1e-9 * dy {
        1
    } else {
        return Err(err(
            line,
            format!(
                "{what} from ({}, {}) to ({}, {}) metres is not parallel to x or y, and a diagonal 'contact trace' is not supported: the public description says the cells under it are magnified by scale_factor^|tan θ| yet also that the magnification runs from 1 to scale_factor as θ goes from 90° to 45°, which disagree past 45°, and it does not say where a diagonal trace's refining lines go or what cell they ask for along the trace, so this reader will not guess the cell size. Refine under it explicitly with 'contact line (x0, y0, z0, x1, y1, z1, xcell, ycell)' — a diagonal line refines its bounding box to the cell you choose (see docs/fasthenry-compat.md)",
                ends[0][0], ends[0][1], ends[1][0], ends[1][1]
            ),
        ));
    };
    Ok(RefineTrace { ends, width, along })
}

/// The fewest uniform cells across `span` whose extent is no larger than
/// `cell` — `ceil(span / cell)`, except that a span that is an exact
/// multiple of `cell` must not gain a spurious extra cell from a last-bit
/// rounding of the division.
fn cells_no_coarser_than(span: f64, cell: f64) -> usize {
    let quotient = span / cell;
    let rounded = quotient.round();
    (if (quotient - rounded).abs() <= 1e-9 * rounded {
        rounded
    } else {
        quotient.ceil()
    } as usize)
        .max(1)
}

/// Parses a FastHenry-form `G` ground-plane statement — the corner-point
/// grammar — into the same [`PlaneSpec`] the extension form produces, plus
/// its in-plane node declarations. See the [module documentation](self).
///
/// Every documented parameter is either mapped onto
/// [`fasterhenry::plane::GroundPlane`] or rejected by name: nothing on the
/// statement is silently dropped.
///
/// The items [`scan_plane_items`] finds are applied in order to a
/// [`PlaneStatement`] (one method per clause kind), which then checks the
/// plane's geometry ([`PlaneStatement::frame`]) and assembles the plane
/// against it ([`PlaneStatement::finish`]).
fn parse_plane_statement(
    head: &str,
    body: &str,
    unit: f64,
    defaults: &Defaults,
    line: usize,
) -> Result<(PlaneSpec, Vec<PlaneNode>), ParseError> {
    let mut statement = PlaneStatement::new(head, unit, defaults, line);
    for item in scan_plane_items(body, line)? {
        statement.apply_item(item)?;
    }
    statement.finish()
}

/// A FastHenry-form `G` statement under construction: what each clause of
/// its body has set so far, between [`scan_plane_items`] and
/// [`PlaneStatement::finish`].
///
/// Holes and contacts are kept raw until the plane's own geometry is known,
/// because their `z` is checked against the plane's slab and some of them
/// are sized against its background cell.
#[derive(Default)]
struct PlaneStatement<'a> {
    /// The statement's head (`G<name>`), quoted in errors.
    head: &'a str,
    /// The `.units` factor to metres in force for the statement.
    unit: f64,
    /// The line the statement was declared on (for errors).
    line: usize,
    /// Corner points, indexed [point][axis]; `None` until the deck sets it.
    corners: [[Option<f64>; 3]; 3],
    /// `thick=`, metres (its magnitude).
    thickness: Option<f64>,
    /// `seg1=` and `seg2=`: cells along p1→p2 and along p2→p3. `contact
    /// initial_grid` / `contact initial_mesh_grid` set this same pair.
    segments: [Option<usize>; 2],
    /// The `contact initial_grid` / `contact initial_mesh_grid` clause that
    /// set [`PlaneStatement::segments`], quoted for errors — `None` until
    /// one does.
    initial_grid: Option<&'static str>,
    /// Whether that clause was the meshed form, which additionally punches
    /// the documented checkerboard of holes into the initial grid.
    meshed_grid: bool,
    /// `sigma=` / `rho=` (S/m), or the `.default` conductivity.
    sigma: Option<f64>,
    /// `nhinc=` (1 unless set).
    nhinc: usize,
    /// `N<name> (x, y, z)` declarations, in statement order.
    nodes: Vec<PlaneNode>,
    /// `hole rect (x1, y1, z1, x2, y2, z2)` corners.
    hole_rects: Vec<[[f64; 3]; 2]>,
    /// `hole point (x, y, z)`.
    hole_points: Vec<[f64; 3]>,
    /// `hole circle (x, y, z, r)`: the centre and radius.
    hole_circles: Vec<([f64; 3], f64)>,
    /// The six-value `contact rect (x1, y1, z1, x2, y2, z2)` corners. Its
    /// documented seven-value spelling is a `contact_decays` entry instead
    /// (issue #95).
    contact_rects: Vec<[[f64; 3]; 2]>,
    /// `contact decay_rect` and the documented seven-value `contact rect`,
    /// plus the decay half of each `contact connection` — each with the
    /// clause's own name, for errors raised once the plane's geometry is
    /// known.
    contact_decays: Vec<(&'static str, DecayRect)>,
    /// `contact point` / `contact line`, with the clause's own name.
    contact_lines: Vec<(&'static str, RefineLine)>,
    /// `contact trace` parallel to x or y: five `contact line`s once its
    /// own ends are checked.
    contact_traces: Vec<RefineTrace>,
    /// `contact equiv_rect` / `contact connection`: the named contact areas,
    /// each becoming one `Equipotential` and one in-plane node.
    contact_equivs: Vec<EquivRect>,
}

/// The checked geometry of a FastHenry-form `G` statement, which every
/// hole, contact, and in-plane node is placed against.
struct PlaneFrame<'a> {
    /// The statement's head (`G<name>`), quoted in errors.
    head: &'a str,
    /// The line the statement was declared on (for errors).
    line: usize,
    /// The plane's xy extent, metres.
    lo: [f64; 2],
    hi: [f64; 2],
    /// Cells along x and y.
    cells: [usize; 2],
    /// The axis (0 = x, 1 = y) the p1→p2 edge — and so `seg1` — runs along.
    axis1: usize,
    /// The p1 corner's xy: the origin of the plane's own coordinate system,
    /// and so the corner `contact initial_mesh_grid` numbers its cells from.
    origin: [f64; 2],
    /// The slack a coordinate may be off the plane's extent by, metres.
    tolerance: f64,
    /// The corner points' `z`: the plane's mid-thickness surface, metres.
    mid_z: f64,
    /// The plane's thickness, metres (> 0).
    thickness: f64,
    /// The plane's conductivity, S/m.
    sigma: f64,
}

impl<'a> PlaneStatement<'a> {
    fn new(head: &'a str, unit: f64, defaults: &Defaults, line: usize) -> Self {
        PlaneStatement {
            head,
            unit,
            line,
            sigma: defaults.sigma,
            nhinc: 1,
            ..PlaneStatement::default()
        }
    }

    /// Dispatches one item of the statement's body to the method that
    /// reads it.
    fn apply_item(&mut self, item: PlaneItem) -> Result<(), ParseError> {
        match item {
            PlaneItem::Field(key, raw) => self.apply_field(&key, &raw),
            PlaneItem::Node(name, values) => self.apply_node(name, &values),
            PlaneItem::Clause {
                kind,
                shape,
                name,
                values,
            } => self.apply_clause(&kind, &shape, name, &values),
        }
    }

    /// A `<key>=<value>` parameter: a corner coordinate, one of the
    /// supported parameters, or a documented one rejected by name.
    fn apply_field(&mut self, key: &str, raw: &str) -> Result<(), ParseError> {
        let (head, unit, line) = (self.head, self.unit, self.line);
        // `x1` … `z3`: one corner coordinate each.
        if let Some(slot) = corner_slot(key) {
            self.corners[slot.0][slot.1] = Some(parse_number(raw, line)? * unit);
            return Ok(());
        }
        match key {
            "thick" => self.thickness = Some(parse_number(raw, line)?.abs() * unit),
            "seg1" | "seg2" => {
                // `contact initial_grid` sets this very pair, so a statement
                // carrying both says the same thing twice — and, if the two
                // disagree, says it twice differently. Reject rather than
                // letting the statement's order decide.
                if let Some(grid) = self.initial_grid {
                    return Err(err(
                        line,
                        format!(
                            "ground plane '{head}': '{key}' and {grid} both set this plane's cell counts — {grid}'s first value *is* 'seg1' (cells along p1→p2) and its second 'seg2' (cells along p2→p3) — so give one or the other, not both"
                        ),
                    ));
                }
                self.segments[usize::from(key == "seg2")] = Some(parse_count(raw, key, line)?);
            }
            // Already converted to S/m by parse_value either way; a
            // line naming both is rejected before we get here.
            "sigma" | "rho" => self.sigma = Some(parse_value(key, raw, unit, line)?),
            "nhinc" => self.nhinc = parse_count(raw, "nhinc", line)?,
            "rh" => {
                return Err(err(
                    line,
                    format!(
                        "ground plane '{head}': 'rh' (filaments graded through the thickness) is not supported; plane filaments are uniform, so use 'nhinc' alone"
                    ),
                ));
            }
            "segwid1" | "segwid2" => {
                return Err(err(
                    line,
                    format!(
                        "ground plane '{head}': '{key}' (an explicit plane-segment width) is not supported; this engine's plane bars take their width from the cell they span (set seg1/seg2, or refine locally with '.contact')"
                    ),
                ));
            }
            "relx" | "rely" | "relz" => {
                return Err(err(
                    line,
                    format!(
                        "ground plane '{head}': '{key}' (a reference point for the plane's internal node numbering) is not supported; name the in-plane nodes you need with 'N<name> (x, y, z)'"
                    ),
                ));
            }
            "file" => {
                // `file=` is an *input*: the file holding the plane's
                // nonuniform-discretization hierarchy, with `file=NONE`
                // meaning there is none — the hierarchy is a single root
                // cell, discretized at run time from the statement's own
                // clauses (issue #122). `NONE` is therefore the one case
                // this reader ever has, so it is accepted as the no-op it
                // is and the documented `file=NONE contact initial_grid
                // (n1, n2)` spelling reads; it is matched without regard to
                // case, as every other token this reader interprets is. A
                // *named* file is rejected by name: an input this reader
                // does not read, not an output it does not write.
                if !raw.eq_ignore_ascii_case("NONE") {
                    return Err(err(
                        line,
                        format!(
                            "ground plane '{head}': 'file={raw}' names the plane's nonuniform-discretization hierarchy, an input file this reader does not read (of the documented spellings only 'file=NONE' — no hierarchy file — is accepted); state the discretization in the statement itself instead, with 'seg1'/'seg2' or 'contact initial_grid (n1, n2)' plus the 'contact' refinement clauses"
                        ),
                    ));
                }
            }
            "nx" | "ny" => {
                return Err(err(
                    line,
                    format!(
                        "ground plane '{head}': the corner-point form counts cells with 'seg1'/'seg2' ('{key}' belongs to the 'G<name> x1 y1 z1 x2 y2 z2 t' extension form)"
                    ),
                ));
            }
            other => {
                return Err(err(
                    line,
                    format!(
                        "unknown ground-plane parameter '{other}' (supported: x1…z3, thick, seg1, seg2, sigma, rho, nhinc)"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// An `N<name> (x, y, z)` in-plane node declaration.
    fn apply_node(&mut self, name: String, values: &[String]) -> Result<(), ParseError> {
        if !name.starts_with(['n', 'N']) {
            return Err(err(
                self.line,
                format!(
                    "'{name} (…)' is not an in-plane node declaration: node names start with 'N'"
                ),
            ));
        }
        let what = format!("in-plane node '{name}'");
        let position = triple(values, &what, self.unit, self.line)?;
        self.nodes.push((name, position, None));
        Ok(())
    }

    /// A `hole <shape> (…)` or `contact <shape> (…)` clause; `kind` and
    /// `shape` arrive lowercased from [`scan_plane_items`].
    fn apply_clause(
        &mut self,
        kind: &str,
        shape: &str,
        name: Option<String>,
        values: &[String],
    ) -> Result<(), ParseError> {
        let (unit, line) = (self.unit, self.line);
        let what = format!("'{kind} {shape}'");
        // Only the named contact areas take a node name; anywhere
        // else it is a mistake to report, not a token to drop.
        if let Some(name) = &name {
            if !matches!(
                (kind, shape),
                ("contact", "equiv_rect") | ("contact", "connection")
            ) {
                return Err(err(
                    line,
                    format!(
                        "{what} takes no node name, but '{name}' stands before its value list; only 'contact equiv_rect' and 'contact connection' name a node"
                    ),
                ));
            }
        }
        match (kind, shape) {
            ("hole", "rect") => {
                self.hole_rects
                    .push(rect_corners(values, &what, unit, line)?);
            }
            ("hole", "point") => {
                self.hole_points.push(triple(values, &what, unit, line)?);
            }
            ("hole", "circle") => self.apply_hole_circle(&what, values)?,
            // Two spellings of one clause, told apart by value count
            // (issue #95): the documented seven — centre, full widths and
            // the largest cell wanted inside — or this reader's own six,
            // two opposite corners as `hole rect` spells them, taking the
            // default 2 × 2 cells at ratio 2.
            ("contact", "rect") => match values.len() {
                7 => self.contact_decays.push((
                    "'contact rect'",
                    contact_rect_values(values, &what, unit, line)?,
                )),
                6 => self
                    .contact_rects
                    .push(rect_corners(values, &what, unit, line)?),
                got => {
                    return Err(err(
                        line,
                        format!(
                            "{what} takes 7 values (x, y, z, xwidth, ywidth, xcell, ycell) — the rectangle's centre, its full widths about that centre, and the largest cell wanted inside it — or 6 (x1, y1, z1, x2, y2, z2), two opposite corners at this reader's default 2 × 2 cells; got {got}"
                        ),
                    ))
                }
            },
            ("contact", "decay_rect") => self.contact_decays.push((
                "'contact decay_rect'",
                decay_rect_values(values, &what, unit, line)?,
            )),
            ("contact", "point") => {
                self.contact_lines.push((
                    "'contact point'",
                    refine_line_values(values, false, &what, unit, line)?,
                ));
            }
            ("contact", "line") => {
                self.contact_lines.push((
                    "'contact line'",
                    refine_line_values(values, true, &what, unit, line)?,
                ));
            }
            ("contact", "trace") => {
                self.contact_traces
                    .push(refine_trace_values(values, &what, unit, line)?);
            }
            ("contact", "equiv_rect") => self.apply_contact_equiv_rect(what, name, values)?,
            ("contact", "connection") => self.apply_contact_connection(what, name, values)?,
            ("contact", "initial_grid") => self.apply_initial_grid(false, what, values)?,
            ("contact", "initial_mesh_grid") => self.apply_initial_grid(true, what, values)?,
            _ => return Err(self.unsupported_clause(kind, shape)),
        }
        Ok(())
    }

    /// `hole circle (x, y, z, r)`.
    fn apply_hole_circle(&mut self, what: &str, values: &[String]) -> Result<(), ParseError> {
        let (unit, line) = (self.unit, self.line);
        if values.len() != 4 {
            return Err(err(
                line,
                format!("{what} takes 4 values (x, y, z, r), got {}", values.len()),
            ));
        }
        let centre = triple(&values[..3], what, unit, line)?;
        let radius = parse_number(&values[3], line)? * unit;
        if radius < 0.0 {
            return Err(err(line, format!("{what}: r={radius} metres must be >= 0")));
        }
        self.hole_circles.push((centre, radius));
        Ok(())
    }

    /// `contact equiv_rect N<name> (x, y, z, xwidth, ywidth)`.
    fn apply_contact_equiv_rect(
        &mut self,
        what: String,
        name: Option<String>,
        values: &[String],
    ) -> Result<(), ParseError> {
        let line = self.line;
        if values.len() != 5 {
            return Err(err(
                line,
                format!(
                    "{what} takes 5 values (x, y, z, xwidth, ywidth), got {}",
                    values.len()
                ),
            ));
        }
        let (centre, widths) = equiv_rect_values(values, &what, self.unit, line)?;
        let name = contact_node_name(name, &what, "equiv_rect", line)?;
        self.contact_equivs.push(EquivRect {
            what,
            name,
            centre,
            widths,
        });
        Ok(())
    }

    /// `contact connection N<name> (x, y, z, xwidth, ywidth, ratio)`.
    fn apply_contact_connection(
        &mut self,
        what: String,
        name: Option<String>,
        values: &[String],
    ) -> Result<(), ParseError> {
        let line = self.line;
        if values.len() != 6 {
            return Err(err(
                line,
                format!(
                    "{what} takes 6 values (x, y, z, xwidth, ywidth, ratio), got {}",
                    values.len()
                ),
            ));
        }
        let (centre, widths) = equiv_rect_values(values, &what, self.unit, line)?;
        // The documented shorthand: this rectangle tied to
        // one node, plus a `decay_rect` over the same
        // rectangle whose cells are its widths divided by
        // `ratio`, decaying without a limit.
        let ratio = parse_number(&values[5], line)?;
        if ratio <= 1.0 {
            return Err(err(
                line,
                format!(
                    "{what}: ratio={ratio} must be > 1 — it divides the rectangle's widths into the contact's own cells (xwidth/ratio, ywidth/ratio), so 1 or less asks for a cell as wide as the rectangle and grades nothing"
                ),
            ));
        }
        let name = contact_node_name(name, &what, "connection", line)?;
        self.contact_equivs.push(EquivRect {
            what,
            name,
            centre,
            widths,
        });
        self.contact_decays.push((
            "'contact connection'",
            DecayRect {
                centre,
                widths,
                cell: [widths[0] / ratio, widths[1] / ratio],
                limit: [None, None],
            },
        ));
        Ok(())
    }

    /// `contact initial_grid (n1, n2)` and `contact initial_mesh_grid (n1,
    /// n2)`: the plane's *initial* discretization, `n1` cells along p1→p2
    /// and `n2` along p2→p3 — the same pair `seg1`/`seg2` set, which is how
    /// the public description of the clause states it ("`seg1=10 seg2=12`
    /// could be replaced with `file=NONE contact initial_grid (10,12)`").
    /// The meshed form additionally records that the checkerboard of holes
    /// is to be punched; [`PlaneFrame::mesh_grid_holes`] places it once the
    /// plane's geometry is known. See the [module documentation](self).
    fn apply_initial_grid(
        &mut self,
        meshed: bool,
        what: String,
        values: &[String],
    ) -> Result<(), ParseError> {
        let (head, line) = (self.head, self.line);
        if values.len() != 2 {
            return Err(err(
                line,
                format!(
                    "{what} takes 2 values (cells along p1→p2, cells along p2→p3 — exactly what 'seg1' and 'seg2' set), got {}",
                    values.len()
                ),
            ));
        }
        if let Some(previous) = self.initial_grid {
            return Err(err(
                line,
                format!(
                    "ground plane '{head}': {what} sets the initial discretization {previous} has already set on this statement; a plane has one initial grid, so give one of the two clauses"
                ),
            ));
        }
        if self.segments.iter().any(Option::is_some) {
            return Err(err(
                line,
                format!(
                    "ground plane '{head}': {what} and 'seg1'/'seg2' both set this plane's cell counts — {what}'s first value *is* 'seg1' (cells along p1→p2) and its second 'seg2' (cells along p2→p3) — so give one or the other, not both"
                ),
            ));
        }
        for (index, edge) in ["p1→p2", "p2→p3"].into_iter().enumerate() {
            let role = format!(
                "cell count along {edge} (the {} value of {what}, what 'seg{}' sets)",
                ["first", "second"][index],
                index + 1
            );
            self.segments[index] = Some(parse_count(&values[index], &role, line)?);
        }
        self.initial_grid = Some(if meshed {
            "'contact initial_mesh_grid'"
        } else {
            "'contact initial_grid'"
        });
        self.meshed_grid = meshed;
        Ok(())
    }

    /// The error for a `hole` / `contact` shape this reader does not read:
    /// each documented one named with the reason, and anything else pointed
    /// at the shapes that are supported.
    fn unsupported_clause(&self, kind: &str, shape: &str) -> ParseError {
        let (head, line) = (self.head, self.line);
        match (kind, shape) {
            ("contact", "circle") => {
                // Issue #109. The public documentation of the `contact`
                // family — "Nonuniformly Discretized Reference Planes in
                // FastHenry 3.0" (M. Kamon, 10 October 1996), the
                // supplement the FastHenry 3.0 user's guide points at for
                // nonuniform planes — names the simple refinement
                // utilities point, line, rect and decay_rect, the
                // contact-area utility equiv_rect, the grouped connection
                // and trace built on them, and the initial_grid /
                // initial_mesh_grid pair. There is no circle among them:
                // `circle` is a *hole* shape, and the same supplement
                // records that the hole utility does not apply to
                // nonuniformly discretized planes. So there is no
                // documented argument list to read, and the clean-room
                // rule forbids inventing one.
                err(
                    line,
                    format!(
                        "ground plane '{head}': 'contact circle' is not supported, and no contact shape of that name is documented: 'circle' is a *hole* shape ('hole circle (x, y, z, r)'), while the documented contact utilities are point, line, rect, decay_rect, equiv_rect, connection, trace and initial_grid/initial_mesh_grid — none of them a disc, so nothing here says what these values mean. Refine the disc's bounding square instead: 'contact decay_rect (x, y, z, 2r, 2r, xcell, ycell, xmaxcell, ymaxcell)' (widen each width by its own cell to keep the cells grazing the rim fine too), or 'contact point (x, y, z, xcell, ycell)' at the centre of a disc smaller than one cell. That costs nothing a disc would not: this engine's mesh is a tensor product, so a refined band on one axis spans the plane on the other and any refinement covering a disc refines its bounding square anyway (see docs/fasthenry-compat.md)"
                    ),
                )
            }
            ("hole", other) if is_user_hole(other) => err(
                line,
                format!(
                    "ground plane '{head}': 'hole {other}' is not supported, and will not be: a user-defined hole is a generator compiled into the tool itself, so nothing in the deck says what its values mean and there is no shape here to read. Cut the cells you want with 'hole rect (x1, y1, z1, x2, y2, z2)', 'hole point (x, y, z)' or 'hole circle (x, y, z, r)'; for a shape none of those describe, build the plane through the library instead — 'fasterhenry::plane::GroundPlane::mesh' reports the exact cell centres this plane meshes to, and a 'fasterhenry::plane::Hole::Point' at each centre your own rule selects removes precisely those cells (see docs/fasthenry-compat.md)"
                ),
            ),
            ("hole", other) => err(
                line,
                format!(
                    "ground plane '{head}': 'hole {other}' is not supported; this engine's holes are 'hole rect (x1, y1, z1, x2, y2, z2)', 'hole point (x, y, z)' or 'hole circle (x, y, z, r)'"
                ),
            ),
            (_, other) => err(
                line,
                format!(
                    "ground plane '{head}': 'contact {other}' is not supported; this engine's contacts are axis-aligned rectangles refined in place, so use 'contact rect (x, y, z, xwidth, ywidth, xcell, ycell)', 'contact decay_rect (x, y, z, xwidth, ywidth, xcell, ycell, xmaxcell, ymaxcell)', 'contact point (x, y, z, xcell, ycell)', 'contact line (x0, y0, z0, x1, y1, z1, xcell, ycell)' or 'contact trace (x0, y0, z0, x1, y1, z1, trace_width, scale_factor)' along x or y (and '.contact' to set a rectangle's refinement directly, or 'contact equiv_rect N<name> (x, y, z, xwidth, ywidth)' / 'contact connection N<name> (x, y, z, xwidth, ywidth, ratio)' to tie a rectangle of cells to one node)"
                ),
            ),
        }
    }

    /// Checks the statement's geometry — three corners of an axis-aligned
    /// rectangle parallel to xy, both cell counts, a positive thickness,
    /// and a conductivity — and returns the frame the plane's features are
    /// placed against.
    fn frame(&self) -> Result<PlaneFrame<'a>, ParseError> {
        let (head, line) = (self.head, self.line);
        let mut points = [[0.0f64; 3]; 3];
        for (index, point) in self.corners.iter().enumerate() {
            for (axis, value) in point.iter().enumerate() {
                points[index][axis] = value.ok_or_else(|| {
                    err(
                        line,
                        format!(
                            "ground plane '{head}' is missing '{}{}' (the corner points are x1…z1, x2…z2, x3…z3)",
                            ['x', 'y', 'z'][axis],
                            index + 1
                        ),
                    )
                })?;
            }
        }
        let edge1 = [points[1][0] - points[0][0], points[1][1] - points[0][1]];
        let edge2 = [points[2][0] - points[1][0], points[2][1] - points[1][1]];
        let span: f64 = edge1[0].abs() + edge1[1].abs() + edge2[0].abs() + edge2[1].abs();
        let tolerance = 1e-9 * span;
        if (points[1][2] - points[0][2]).abs() > tolerance
            || (points[2][2] - points[0][2]).abs() > tolerance
        {
            return Err(err(
                line,
                format!(
                    "ground plane '{head}' is not parallel to the xy plane (z1={}, z2={}, z3={} in metres); tilted planes are not supported",
                    points[0][2], points[1][2], points[2][2]
                ),
            ));
        }
        let axis_of = |edge: [f64; 2]| -> Option<usize> {
            match (edge[0].abs() > tolerance, edge[1].abs() > tolerance) {
                (true, false) => Some(0),
                (false, true) => Some(1),
                _ => None,
            }
        };
        let (axis1, axis2) = match (axis_of(edge1), axis_of(edge2)) {
            (Some(a), Some(b)) if a != b => (a, b),
            _ => {
                return Err(err(
                    line,
                    format!(
                        "ground plane '{head}': the corner points do not form an axis-aligned rectangle (p1→p2 and p2→p3 must each run along x or y, and along different axes); rotated or non-rectangular planes are not supported"
                    ),
                ));
            }
        };
        let mut cells = [0usize; 2];
        cells[axis1] = self.segments[0].ok_or_else(|| {
            err(
                line,
                format!("ground plane '{head}' has no 'seg1' (cells along p1→p2)"),
            )
        })?;
        cells[axis2] = self.segments[1].ok_or_else(|| {
            err(
                line,
                format!("ground plane '{head}' has no 'seg2' (cells along p2→p3)"),
            )
        })?;
        let thickness = self.thickness.ok_or_else(|| {
            err(
                line,
                format!("ground plane '{head}' has no 'thick' (its thickness)"),
            )
        })?;
        if thickness <= 0.0 {
            return Err(err(
                line,
                format!(
                    "ground plane '{head}' needs a thickness > 0 (got 'thick={thickness}' metres)"
                ),
            ));
        }
        let sigma = self.sigma.ok_or_else(|| {
            err(
                line,
                format!(
                    "ground plane '{head}' has no conductivity: set sigma= on the statement, or sigma= or rho= in .default"
                ),
            )
        })?;
        let lo = [
            points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
            points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
        ];
        let hi = [
            points
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max),
            points
                .iter()
                .map(|p| p[1])
                .fold(f64::NEG_INFINITY, f64::max),
        ];
        Ok(PlaneFrame {
            head,
            line,
            lo,
            hi,
            cells,
            axis1,
            origin: [points[0][0], points[0][1]],
            tolerance,
            mid_z: points[0][2],
            thickness,
            sigma,
        })
    }

    /// Checks the geometry, places every hole, contact, and in-plane node
    /// against it, and assembles the [`PlaneSpec`] plus its in-plane nodes.
    fn finish(self) -> Result<(PlaneSpec, Vec<PlaneNode>), ParseError> {
        let frame = self.frame()?;
        let PlaneStatement {
            head,
            nhinc,
            meshed_grid,
            mut nodes,
            hole_rects,
            hole_points,
            hole_circles,
            contact_rects,
            contact_decays,
            contact_lines,
            contact_traces,
            contact_equivs,
            ..
        } = self;

        // The meshed initial grid's holes are cut into the *initial* grid,
        // before anything else refines it, so they come first.
        let mut holes = if meshed_grid {
            frame.mesh_grid_holes()
        } else {
            Vec::new()
        };
        holes.reserve(hole_rects.len() + hole_points.len() + hole_circles.len());
        for rect in hole_rects {
            let (lo, hi) = frame.footprint("'hole rect'", rect)?;
            holes.push(Hole::Rect { lo, hi });
        }
        for point in hole_points {
            frame.in_slab("'hole point'", point)?;
            holes.push(Hole::Point {
                at: [point[0], point[1]],
            });
        }
        for (centre, radius) in hole_circles {
            frame.in_slab("'hole circle'", centre)?;
            holes.push(Hole::Circle {
                centre: [centre[0], centre[1]],
                radius,
            });
        }

        let mut contacts =
            Vec::with_capacity(contact_rects.len() + contact_decays.len() + contact_lines.len());
        for rect in contact_rects {
            let (lo, hi) = frame.footprint("'contact rect'", rect)?;
            contacts.push(ContactRegion::new(lo, hi, [2, 2], 2.0));
        }
        for (what, decay) in contact_decays {
            contacts.push(frame.decay_contact(what, &decay)?);
        }
        for (what, refine) in contact_lines {
            if let Some(region) = frame.refine_contact(what, &refine)? {
                contacts.push(region);
            }
        }
        for trace in &contact_traces {
            contacts.extend(frame.trace_contacts(trace)?);
        }

        // A named contact area is one equipotential patch plus the in-plane
        // node that names it; the node sits at the rectangle's centre, so the
        // footprint and slab checks below cover it like any other.
        let mut equipotentials = Vec::with_capacity(contact_equivs.len());
        for equiv in contact_equivs {
            frame.in_slab(&equiv.what, equiv.centre)?;
            // The centre is global; the widths are a pair in the plane's own
            // coordinate system. See `PlaneFrame::plane_pair`.
            equipotentials.push(Equipotential::centred(
                [equiv.centre[0], equiv.centre[1]],
                frame.plane_pair(equiv.widths),
            ));
            nodes.push((equiv.name, equiv.centre, Some(equipotentials.len() - 1)));
        }

        for (name, position, _) in &nodes {
            frame.check_node(name, *position)?;
        }

        // The corner points give the plane's mid-thickness surface, as a
        // segment's nodes give its axis; this engine's `GroundPlane` is
        // specified by its top surface.
        let z_top = frame.mid_z + frame.thickness / 2.0;
        Ok((
            PlaneSpec {
                name: head.to_string(),
                plane: GroundPlane {
                    lo: frame.lo,
                    hi: frame.hi,
                    z_top,
                    thickness: frame.thickness,
                    nx: frame.cells[0],
                    ny: frame.cells[1],
                    sigma: frame.sigma,
                    holes,
                    contacts,
                    equipotentials,
                },
                nhinc,
            },
            nodes,
        ))
    }
}

/// The node a named contact area (`contact equiv_rect` / `contact
/// connection`) declares: `N`-prefixed like every other node name in a `G`
/// statement's body. `what` is the clause quoted for errors.
fn contact_node_name(
    name: Option<String>,
    what: &str,
    shape: &str,
    line: usize,
) -> Result<String, ParseError> {
    let kind = "contact";
    match name {
        Some(name) if name.starts_with(['n', 'N']) => Ok(name),
        Some(name) => Err(err(
            line,
            format!(
                "{what}: '{name}' is not a node name (node names start with 'N', as in '{kind} {shape} Ncontact (…)')"
            ),
        )),
        None => Err(err(
            line,
            format!(
                "{what} names the node it ties: write '{kind} {shape} N<name> (…)', with the node name between the shape and its values"
            ),
        )),
    }
}

impl PlaneFrame<'_> {
    /// A hole or contact z outside the plane's own slab is a deck mistake,
    /// not a hole somewhere else: the rectangles are cut through the full
    /// thickness of a plane that is parallel to xy.
    fn in_slab(&self, what: &str, corner: [f64; 3]) -> Result<(), ParseError> {
        let (mid_z, thickness) = (self.mid_z, self.thickness);
        if (corner[2] - mid_z).abs() > thickness + 1e-15 {
            return Err(err(
                self.line,
                format!(
                    "{what}: z={} is not in ground plane '{}' (mid-thickness z={mid_z}, thickness {thickness}, metres)",
                    corner[2], self.head
                ),
            ));
        }
        Ok(())
    }

    /// The checkerboard of holes a `contact initial_mesh_grid` punches into
    /// the initial grid: the documented rule is "every cell that has an even
    /// value for both of its indices where the numbering is from the top
    /// left", so with the cells numbered from 1 at the plane's own origin
    /// (its p1 corner, [`PlaneFrame::origin`]) along each axis, a cell is a
    /// hole when both of its indices are even.
    ///
    /// Each hole is the holed cell's own rectangle rather than a point at
    /// its centre. On the initial grid alone the two are the same cut, but
    /// the grid is *initial* — a later `contact` clause may refine that
    /// region into smaller cells, and the documented hole is the square
    /// ("no conductor will be defined in that square region"), not whatever
    /// one cell the refinement happens to leave under the centre.
    ///
    /// An axis with a single cell has no even index, and so no hole: a
    /// checkerboard needs a cell on either side of each hole.
    fn mesh_grid_holes(&self) -> Vec<Hole> {
        // The holed cells' extents along one axis, in plane coordinates.
        let holed = |axis: usize| -> Vec<(f64, f64)> {
            let count = self.cells[axis];
            let cell = self.background(axis);
            // Index 1 is the cell at the origin end of the axis, which is
            // the low end unless p1 sits at the plane's far corner there.
            let ascending = (self.origin[axis] - self.lo[axis]).abs()
                <= (self.hi[axis] - self.origin[axis]).abs();
            (0..count)
                // A 0-based index counted from the origin is even 1-based
                // exactly when it is odd here.
                .filter(|index| index % 2 == 1)
                .map(|index| {
                    let from_lo = if ascending { index } else { count - 1 - index };
                    (
                        self.lo[axis] + from_lo as f64 * cell,
                        self.lo[axis] + (from_lo + 1) as f64 * cell,
                    )
                })
                .collect()
        };
        let (rows, columns) = (holed(0), holed(1));
        let mut holes = Vec::with_capacity(rows.len() * columns.len());
        for &(x_lo, x_hi) in &rows {
            for &(y_lo, y_hi) in &columns {
                holes.push(Hole::Rect {
                    lo: [x_lo, y_lo],
                    hi: [x_hi, y_hi],
                });
            }
        }
        holes
    }

    /// The xy footprint of a `hole rect` / `contact rect`, both corners in
    /// the slab and the rectangle not degenerate.
    fn footprint(
        &self,
        what: &str,
        rect: [[f64; 3]; 2],
    ) -> Result<([f64; 2], [f64; 2]), ParseError> {
        self.in_slab(what, rect[0])?;
        self.in_slab(what, rect[1])?;
        let lo = [rect[0][0].min(rect[1][0]), rect[0][1].min(rect[1][1])];
        let hi = [rect[0][0].max(rect[1][0]), rect[0][1].max(rect[1][1])];
        if !(hi[0] > lo[0] && hi[1] > lo[1]) {
            return Err(err(
                self.line,
                format!("{what} is degenerate: its two corners share an x or a y"),
            ));
        }
        Ok((lo, hi))
    }

    /// Whether `point`'s xy lies on the plane, within the tolerance.
    fn contains(&self, point: [f64; 3]) -> bool {
        let (lo, hi, tolerance) = (self.lo, self.hi, self.tolerance);
        point[0] >= lo[0] - tolerance
            && point[0] <= hi[0] + tolerance
            && point[1] >= lo[1] - tolerance
            && point[1] <= hi[1] + tolerance
    }

    /// The plane's uniform background cell along `axis`, metres.
    fn background(&self, axis: usize) -> f64 {
        (self.hi[axis] - self.lo[axis]) / self.cells[axis] as f64
    }

    /// A per-axis pair written in the plane's **own** coordinate system —
    /// index 0 along `p1 → p2`, index 1 along `p2 → p3` — mapped onto the
    /// global x and y axes those two edges run along.
    ///
    /// Every `contact` clause that carries an `x…`/`y…` pair of *lengths*
    /// states that pair this way. The public memo (*Nonuniformly
    /// Discretized Reference Planes in FastHenry 3.0*, M. Kamon, 10 October
    /// 1996) makes `p1` the origin of a plane coordinate system whose
    /// x-direction is the vector `p1 → p2` and whose y-direction is
    /// `p2 → p3`, and then defines `contact point`'s two cell sizes as the
    /// cell's width "in the plane coordinate system's x-direction" and its
    /// width in the y-direction. `line`, `rect` and `decay_rect` are
    /// defined in that same memo as repeated `point` calls, and `connection`
    /// as an `equiv_rect` plus a `decay_rect` over the same widths, so the
    /// whole family inherits it. For a plane whose first edge runs along
    /// global x the mapping is the identity; for one whose first edge runs
    /// along global y it transposes the pair (issue #118) — exactly as
    /// `seg1`/`seg2` and `contact initial_grid` already follow the edges
    /// rather than the axes.
    ///
    /// Only the lengths are plane-relative: each clause's **coordinates**
    /// are ordinary global deck coordinates, checked against the plane's
    /// footprint and slab like every other coordinate here. The memo's
    /// examples place contacts by absolute position on the plane (its
    /// `decay_rect` example puts one at `(1,0,0)` on a plane spanning
    /// `y = -2 … 2`, which is that plane's mid-height, not an offset from
    /// `p1`), and `p1` is named as the origin only where the text needs a
    /// direction or a cell numbering.
    fn plane_pair<T>(&self, pair: [T; 2]) -> [T; 2] {
        if self.axis1 == 0 {
            pair
        } else {
            let [along_first, along_second] = pair;
            [along_second, along_first]
        }
    }

    /// The graded region a `contact decay_rect`, a documented seven-value
    /// `contact rect`, or the decay half of a `contact connection` refines.
    /// `what` is the clause quoted for errors.
    fn decay_contact(&self, what: &str, decay: &DecayRect) -> Result<ContactRegion, ParseError> {
        self.in_slab(what, decay.centre)?;
        // The clause's three length pairs are in the plane's own coordinate
        // system; the region below is global. See `PlaneFrame::plane_pair`.
        let widths = self.plane_pair(decay.widths);
        let cell = self.plane_pair(decay.cell);
        let limit = self.plane_pair(decay.limit);
        let mut region = ([0.0f64; 2], [0.0f64; 2]);
        let mut region_cells = [0usize; 2];
        let mut ratio = [0.0f64; 2];
        for axis in 0..2 {
            // The deck's own name for this axis: its `x…` values are the
            // plane's first edge, whichever global axis that edge follows.
            let (name, seg) = if axis == self.axis1 {
                ('x', 1)
            } else {
                ('y', 2)
            };
            let half = widths[axis] / 2.0;
            region.0[axis] = decay.centre[axis] - half;
            region.1[axis] = decay.centre[axis] + half;
            // The fine cells are the fewest whose own extent is no larger
            // than the deck's `<axis>cell`; a width that is an exact
            // multiple of it must not gain a spurious extra cell from a
            // last-bit rounding of the division.
            region_cells[axis] = cells_no_coarser_than(widths[axis], cell[axis]);
            // The documented decay law: with `r0` the requested cell as a
            // fraction of the rectangle's width, each cell outside the
            // rectangle is `1/(1 − r0)` times its inward neighbour.
            ratio[axis] = 1.0 / (1.0 - cell[axis] / widths[axis]);
            let background = self.background(axis);
            if limit[axis].is_some_and(|limit| limit < background * (1.0 - 1e-9)) {
                return Err(err(
                    self.line,
                    format!(
                        "{what}: {name}maxcell={} metres is finer than ground plane '{}'s own background cell ({background} metres), and this engine's grading levels off at that cell rather than below it; raise 'seg{seg}' so the whole plane is at least that fine, or drop the limit (a negative value) to accept the background cell",
                        limit[axis].unwrap_or_default(),
                        self.head,
                    ),
                ));
            }
        }
        Ok(ContactRegion::graded_per_axis(
            region.0,
            region.1,
            region_cells,
            ratio,
        ))
    }

    /// Both ends of a deck-given refinement locus (`contact point` /
    /// `contact line` / `contact trace`) in the slab and on the footprint.
    fn on_plane(&self, what: &str, end: [f64; 3]) -> Result<(), ParseError> {
        self.in_slab(what, end)?;
        if !self.contains(end) {
            return Err(err(
                self.line,
                format!(
                    "{what}: ({}, {}) metres is outside ground plane '{}'",
                    end[0], end[1], self.head
                ),
            ));
        }
        Ok(())
    }

    /// The region a `contact point` / `contact line` refines, once its ends
    /// are checked against the plane — see [`PlaneFrame::refine_region`].
    fn refine_contact(
        &self,
        what: &str,
        refine: &RefineLine,
    ) -> Result<Option<ContactRegion>, ParseError> {
        for end in refine.ends {
            self.on_plane(what, end)?;
        }
        Ok(self.refine_region(refine))
    }

    /// The regions a `contact trace` refines: its five documented lines,
    /// once the trace's own ends are checked against the plane. Only those
    /// ends are the deck's locus: the lines beside a trace running along
    /// the plane's edge may fall partly off it, and their regions are
    /// clipped like any other (or dropped, where they miss it).
    fn trace_contacts(&self, trace: &RefineTrace) -> Result<Vec<ContactRegion>, ParseError> {
        for end in trace.ends {
            self.on_plane("'contact trace'", end)?;
        }
        Ok(trace
            .lines(self.axis1)
            .iter()
            .filter_map(|refine| self.refine_region(refine))
            .collect())
    }

    /// The region a refinement line asks for: the segment's bounding box,
    /// padded by half a requested cell each side, cut into cells no coarser
    /// than that cell and graded outward at the reader's default ratio 2 —
    /// or `None` when the background mesh already meets the request on both
    /// axes, or the region misses the plane entirely. See the module
    /// documentation for why the box is exact, not a compromise, on this
    /// engine's tensor-product mesh.
    ///
    /// An axis the background mesh already meets is *not* banded around the
    /// request: it spans the whole plane at the plane's own cell count, so
    /// that axis's cell edges are exactly the ones it has without the
    /// clause (issue #116). A band of exactly that shape adds no resolution,
    /// and the mesher drops it before merging bands, so it cannot widen a
    /// genuinely finer refinement elsewhere on the axis across the whole of
    /// it — see `fasterhenry::plane` § *A band that is not a refinement*
    /// (issue #124).
    fn refine_region(&self, refine: &RefineLine) -> Option<ContactRegion> {
        // The requested cell is a pair in the plane's own coordinate system;
        // the region is global. See `PlaneFrame::plane_pair`.
        let requested = self.plane_pair(refine.cell);
        let mut region = ([0.0f64; 2], [0.0f64; 2]);
        let mut region_cells = [0usize; 2];
        let mut already_met = true;
        for (axis, count) in region_cells.iter_mut().enumerate() {
            // A request at or above the background cell is already met by
            // the background mesh (no cell is ever coarser than it), so it
            // is clamped there: a refinement must never coarsen the plane.
            let background = self.background(axis);
            if requested[axis] >= background * (1.0 - 1e-9) {
                // Clamped: that axis is left exactly as the plane meshes
                // it. Bounding a band to the request's own coordinate
                // would still cut a graded band into it unless the
                // coordinate happened to land on a background grid line
                // (issue #116), so the band spans the whole plane at the
                // plane's own cell count instead — the background mesh,
                // edge for edge. The mesher recognises a band of that
                // shape as the no-op it is and drops it rather than
                // merging it with the axis's real refinements (#124).
                region.0[axis] = self.lo[axis];
                region.1[axis] = self.hi[axis];
                *count = self.cells[axis];
                continue;
            }
            already_met = false;
            let cell = requested[axis];
            let (a, b) = (refine.ends[0][axis], refine.ends[1][axis]);
            region.0[axis] = a.min(b) - cell / 2.0;
            region.1[axis] = a.max(b) + cell / 2.0;
            *count = cells_no_coarser_than(region.1[axis] - region.0[axis], cell);
        }
        // Only a trace's side lines can miss the plane entirely (every
        // other locus has its ends on the footprint): such a region
        // refines no cell of this plane, so it is dropped rather than
        // handed to the library, which rejects a region off the plane.
        let off_plane = (0..2)
            .any(|axis| region.1[axis].min(self.hi[axis]) <= region.0[axis].max(self.lo[axis]));
        (!already_met && !off_plane)
            .then(|| ContactRegion::new(region.0, region.1, region_cells, 2.0))
    }

    /// An in-plane node must lie on the plane and in its slab.
    fn check_node(&self, name: &str, position: [f64; 3]) -> Result<(), ParseError> {
        if !self.contains(position) {
            return Err(err(
                self.line,
                format!(
                    "in-plane node '{name}' at ({}, {}) metres is outside ground plane '{}'",
                    position[0], position[1], self.head
                ),
            ));
        }
        self.in_slab(&format!("in-plane node '{name}'"), position)
    }
}

/// `x1` … `z3` as `(point index, axis)`; `None` for any other key.
fn corner_slot(key: &str) -> Option<(usize, usize)> {
    let bytes = key.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let axis = match bytes[0] {
        b'x' => 0,
        b'y' => 1,
        b'z' => 2,
        _ => return None,
    };
    let point = match bytes[1] {
        b'1' => 0,
        b'2' => 1,
        b'3' => 2,
        _ => return None,
    };
    Some((point, axis))
}

/// Node names to node slots, with `.equiv` aliases resolved at lookup and
/// aliased-away slots compacted out of the final geometry.
#[derive(Default)]
struct Names {
    ids: HashMap<String, usize>,
    aliases: HashMap<usize, usize>,
}

impl Names {
    fn define(&mut self, name: &str, id: usize, line: usize) -> Result<(), ParseError> {
        if self.ids.contains_key(name) {
            return Err(err(line, format!("duplicate node name '{name}'")));
        }
        self.ids.insert(name.to_string(), id);
        Ok(())
    }

    fn lookup(&self, name: &str, line: usize) -> Result<usize, ParseError> {
        let mut id = *self
            .ids
            .get(name)
            .ok_or_else(|| err(line, format!("unknown node '{name}'")))?;
        while let Some(&target) = self.aliases.get(&id) {
            id = target;
        }
        Ok(id)
    }

    /// The final index of every live slot: aliased-away slots are dropped
    /// and the rest compacted in order.
    fn compaction(&self, slot_count: usize) -> Vec<Option<usize>> {
        let mut live = 0;
        (0..slot_count)
            .map(|slot| {
                if self.aliases.contains_key(&slot) {
                    None
                } else {
                    live += 1;
                    Some(live - 1)
                }
            })
            .collect()
    }
}

/// How [`parse_with_options`] reads a deck. The [`Default`] is this
/// reader's own, safety-first framing; see [`ParseOptions::fasthenry_compat`]
/// for the opt-in alternative.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParseOptions {
    /// Read the deck's first physical line as an always-ignored title, as
    /// the public FastHenry format describes, instead of parsing it.
    ///
    /// Off (the default), there is no implicit first line: line 1 is parsed
    /// like every other line, so prose there is rejected as an unrecognized
    /// line, and `.title <text>` is the only way to set [`Deck::title`].
    ///
    /// On, line 1 of the file — whatever it holds, including nothing, a `*`
    /// comment, or something that looks like a directive — is taken before
    /// comment/blank/continuation folding and never dispatched; its words,
    /// single-spaced, become [`Deck::title`] (`None` when blank). It is the *physical*
    /// first line, not the first non-blank one: otherwise a deck with a
    /// blank line 1 would silently lose its first real directive. Line
    /// numbers in errors still count the title line, and a `+` line right
    /// after it is an error (a title is not continued).
    ///
    /// `.title` interaction: a `.title` directive later in the deck is still
    /// honored in compat mode and replaces the line-1 title. It is kept
    /// rather than disabled because it is additive and unambiguous — a deck
    /// written for this reader that also carries a prose first line reads
    /// the same either way — whereas disabling it would turn a line this
    /// reader otherwise accepts into an error for no gain in safety.
    pub fasthenry_compat: bool,
}

/// Parses a whole deck with the default [`ParseOptions`]. See the [module
/// documentation](self) for the subset.
pub fn parse(text: &str) -> Result<Deck, ParseError> {
    parse_with_options(text, ParseOptions::default())
}

/// Parses a whole deck under `options`. See the [module documentation](self)
/// for the subset and [`ParseOptions`] for what each option changes.
pub fn parse_with_options(text: &str, options: ParseOptions) -> Result<Deck, ParseError> {
    let folded = fold_lines(text, options)?;
    let mut deck = DeckBuilder {
        title: folded.title,
        ..DeckBuilder::default()
    };
    let parse_wall = std::time::Instant::now();
    for &(number, ref tokens) in &folded.lines {
        deck.apply_line(number, tokens)?;
        if number % 500_000 == 0 {
            eprintln!(
                "deck: applied through line {number}, wall {:.1} s",
                parse_wall.elapsed().as_secs_f64()
            );
        }
    }
    let parsed = deck.finish(folded.last_number)?;
    eprintln!(
        "deck: finish done (nodes/segments grouped), wall {:.1} s",
        parse_wall.elapsed().as_secs_f64()
    );
    Ok(parsed)
}

/// A deck's lines ready for dispatch: `*` comments and blank lines dropped,
/// and every `+` continuation folded into the line it continues.
struct FoldedDeck<'a> {
    /// The title taken from the physical first line in
    /// [`ParseOptions::fasthenry_compat`] mode, `None` otherwise (a later
    /// `.title` directive still replaces it).
    title: Option<String>,
    /// The number of the last physical line of the file, for the deck-level
    /// errors that belong to no directive (a missing `.end`, `.units`).
    last_number: usize,
    /// Each surviving line: the number of its *first* physical line, and its
    /// whitespace-separated tokens with every continuation appended.
    lines: Vec<(usize, Vec<&'a str>)>,
}

/// Reads the raw text into [`FoldedDeck`] — everything
/// [`parse_with_options`] does before the first directive is dispatched.
fn fold_lines(text: &str, options: ParseOptions) -> Result<FoldedDeck<'_>, ParseError> {
    let mut raw_lines = text.lines().enumerate();
    let mut title = None;
    let mut last_number = 0;
    if options.fasthenry_compat {
        // The public format's always-ignored first line: taken before any
        // folding so that no content on it can reach the dispatch below.
        if let Some((_, first)) = raw_lines.next() {
            last_number = 1;
            let first = first.split_whitespace().collect::<Vec<_>>().join(" ");
            if !first.is_empty() {
                title = Some(first);
            }
        }
    }

    // Fold `+` continuation lines into their first line, keeping the first
    // line's number for error reporting; `*` comments and blank lines drop.
    let mut lines: Vec<(usize, Vec<&str>)> = Vec::new();
    for (index, raw) in raw_lines {
        let number = index + 1;
        last_number = number;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('*') {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if let Some(continuation) = tokens[0].strip_prefix('+') {
            match lines.last_mut() {
                Some((_, previous)) => {
                    if !continuation.is_empty() {
                        previous.push(continuation);
                    }
                    previous.extend_from_slice(&tokens[1..]);
                }
                None => return Err(err(number, "deck starts with a '+' continuation line")),
            }
        } else {
            lines.push((number, tokens));
        }
    }

    Ok(FoldedDeck {
        title,
        last_number,
        lines,
    })
}

/// A deck under construction: everything the directives of a deck
/// accumulate, between [`fold_lines`] and [`DeckBuilder::finish`].
///
/// One field per piece of deck-wide state, so that each directive is read by
/// its own method ([`DeckBuilder::apply_line`] dispatches) rather than by an
/// arm of one function sharing a stack frame's worth of locals.
#[derive(Default)]
struct DeckBuilder {
    /// The deck's title: its last `.title`, or (in compat mode) line 1.
    title: Option<String>,
    /// The `.units` factor to metres; `None` until `.units` is read, which
    /// is what makes a length-bearing line before it an error.
    unit: Option<f64>,
    /// The `.default` fields in force, already scaled.
    defaults: Defaults,
    /// Node names to slots, plus the `.equiv` aliases between slots.
    names: Names,
    /// Every declared node's position in metres, by slot.
    positions: Vec<[f64; 3]>,
    /// Every `E` line, pending assembly.
    segment_defs: Vec<SegmentSpec>,
    /// One filament grid per `E` line, in `segment_defs` order.
    subdivisions: Vec<AxisGrading>,
    /// One `group=` name per `E` line (empty for untagged), same order.
    segment_groups: Vec<String>,
    /// The cross-group pairs `.couples` declared.
    couples: Vec<[String; 2]>,
    /// Every group name any `.couples` line mentions, with the line it was
    /// mentioned on, so a name no segment carries is an error even when it
    /// was named alone and so appears in no pair.
    couples_names: Vec<(usize, String)>,
    /// The line of the *first* `.couples` directive, for error reporting.
    couples_line: Option<usize>,
    /// Whether `.couples all` was declared.
    couples_all: bool,
    /// Whether any `.couples` named groups (the two are exclusive).
    named_couples: bool,
    /// Ports from `.external`, in deck order.
    ports: Vec<Port>,
    /// The `.freq` sweep in hertz.
    frequencies: Vec<f64>,
    /// Every `G` statement, pending assembly.
    planes: Vec<PlaneSpec>,
    /// In-plane nodes declared inside a FastHenry-form `G` statement; see
    /// [`PlaneNodeSpec`] for what each carries.
    plane_nodes: Vec<PlaneNodeSpec>,
    /// The line each `.equiv` alias was declared on, for the error an
    /// in-plane node joined across two planes has to raise.
    alias_lines: HashMap<usize, usize>,
    /// Whether `.end` has been read (content after it is an error).
    ended: bool,
}

impl DeckBuilder {
    /// Dispatches one folded line to the method that reads it. Every method
    /// it calls takes the line number it was read on, so that the error it
    /// raises carries the deck's own line rather than the parser's.
    fn apply_line(&mut self, number: usize, tokens: &[&str]) -> Result<(), ParseError> {
        if self.ended {
            return Err(err(number, "content after .end"));
        }
        let head = tokens[0];
        let keyword = head.to_ascii_lowercase();
        // The unit in force for *this* line: a line that needs one and was
        // read before `.units` is rejected by the method that reads it.
        let factor = self.unit.unwrap_or(1.0);

        if let Some(directive) = keyword.strip_prefix('.') {
            return self.apply_directive(directive, tokens, factor, number);
        }

        match keyword.chars().next() {
            Some('n') => self.apply_node(tokens, factor, number),
            Some('e') => self.apply_segment(tokens, factor, number),
            Some('g') => self.apply_ground_plane(tokens, factor, number),
            _ => Err(err(
                number,
                format!("unrecognized line '{head}' (expected N…, E…, or a .directive)"),
            )),
        }
    }

    /// Dispatches a `.`-prefixed line; `directive` is the keyword with its
    /// leading `.` stripped and already lowercased.
    fn apply_directive(
        &mut self,
        directive: &str,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        match directive {
            "title" => {
                self.title = Some(tokens[1..].join(" "));
                Ok(())
            }
            "units" => self.apply_units(tokens, number),
            "default" => self.apply_default(tokens, factor, number),
            "external" => self.apply_external(tokens, number),
            "freq" => self.apply_freq(tokens, number),
            "equiv" => self.apply_equiv(tokens, number),
            "couples" => self.apply_couples(tokens, number),
            "hole" => self.apply_hole(tokens, factor, number),
            "contact" => self.apply_contact(tokens, factor, number),
            "end" => {
                if tokens.len() != 1 {
                    return Err(err(number, "expected .end (no arguments)"));
                }
                self.ended = true;
                Ok(())
            }
            other => Err(err(
                number,
                format!("'.{other}' is not in the deck subset (the rest of FastHenry is M1+)"),
            )),
        }
    }

    /// `.units km|m|cm|mm|um|in|mils` — mandatory, and at most once.
    fn apply_units(&mut self, tokens: &[&str], number: usize) -> Result<(), ParseError> {
        if tokens.len() != 2 {
            return Err(err(number, "expected .units <unit>"));
        }
        if self.unit.is_some() {
            return Err(err(number, "duplicate .units directive"));
        }
        self.unit = Some(unit_factor(tokens[1], number)?);
        Ok(())
    }
    /// `.default <field>=<value> …` — the fields later lines fall back to.
    fn apply_default(
        &mut self,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        if self.unit.is_none() {
            return Err(err(number, ".default before .units (lengths need a unit)"));
        }
        one_conductivity(&tokens[1..], number)?;
        for token in &tokens[1..] {
            let (key, raw_value) = parse_field(token, number)?;
            let value = parse_value(&key, &raw_value, factor, number)?;
            set_default(&mut self.defaults, &key, value, number)?;
        }
        Ok(())
    }

    /// `.external N<+> N<-> [name]` — one port.
    fn apply_external(&mut self, tokens: &[&str], number: usize) -> Result<(), ParseError> {
        if tokens.len() != 3 && tokens.len() != 4 {
            return Err(err(number, "expected .external N<+> N<-> [name]"));
        }
        let positive = self.names.lookup(tokens[1], number)?;
        let negative = self.names.lookup(tokens[2], number)?;
        let name = tokens.get(3).map(|name| name.to_string());
        self.ports.push(Port {
            positive: NodeId(positive),
            negative: NodeId(negative),
            name: name.or_else(|| Some(format!("{}/{}", tokens[1], tokens[2]))),
        });
        Ok(())
    }

    /// `.freq fmin=<v> fmax=<v> ndec=<n>` — the decade sweep, in hertz.
    fn apply_freq(&mut self, tokens: &[&str], number: usize) -> Result<(), ParseError> {
        if tokens.len() != 4 {
            return Err(err(number, "expected .freq fmin=<v> fmax=<v> ndec=<n>"));
        }
        let (mut fmin, mut fmax, mut ndec) = (None, None, None);
        for token in &tokens[1..] {
            let (key, raw_value) = parse_field(token, number)?;
            match key.as_str() {
                "fmin" => fmin = Some(parse_number(&raw_value, number)?),
                "fmax" => fmax = Some(parse_number(&raw_value, number)?),
                "ndec" => ndec = Some(parse_count(&raw_value, "ndec", number)?),
                other => {
                    return Err(err(
                        number,
                        format!("unknown .freq field '{other}' (supported: fmin, fmax, ndec)"),
                    ));
                }
            }
        }
        let (fmin, fmax, ndec) = match (fmin, fmax, ndec) {
            (Some(fmin), Some(fmax), Some(ndec)) => (fmin, fmax, ndec),
            _ => return Err(err(number, ".freq needs fmin=, fmax= and ndec=")),
        };
        self.frequencies = frequency_sweep(fmin, fmax, ndec, number)?;
        Ok(())
    }
    /// `.equiv N<a> N<b> [N<c> …]` — join two or more nodes into one.
    fn apply_equiv(&mut self, tokens: &[&str], number: usize) -> Result<(), ParseError> {
        if tokens.len() < 3 {
            return Err(err(number, "expected .equiv N<a> N<b> [N<c> …]"));
        }
        let a = self.names.lookup(tokens[1], number)?;
        for name in &tokens[2..] {
            let b = self.names.lookup(name, number)?;
            if a == b {
                return Err(err(
                    number,
                    format!(".equiv of node '{}' with itself", tokens[1]),
                ));
            }
            self.names.aliases.insert(b, a);
            self.alias_lines.insert(b, number);
        }
        Ok(())
    }

    /// `.couples all | .couples <group> <group> …` — which segment groups
    /// couple; the two forms cannot be mixed in one deck.
    fn apply_couples(&mut self, tokens: &[&str], number: usize) -> Result<(), ParseError> {
        if tokens.len() < 2 {
            return Err(err(
                number,
                "expected .couples all | .couples <group> <group> …",
            ));
        }
        let first = *self.couples_line.get_or_insert(number);
        let conflict = || {
            err(
                number,
                format!(
                    "conflicting .couples declarations (the first is on line {first}): 'all' cannot be combined with named groups"
                ),
            )
        };
        if tokens[1].eq_ignore_ascii_case("all") {
            if tokens.len() != 2 {
                return Err(err(
                    number,
                    "'.couples all' takes no group names (it couples everything)",
                ));
            }
            if self.named_couples {
                return Err(conflict());
            }
            self.couples_all = true;
            return Ok(());
        }
        if self.couples_all {
            return Err(conflict());
        }
        self.named_couples = true;
        // Every listed group couples to every other listed one.
        for (index, a) in tokens[1..].iter().enumerate() {
            for b in &tokens[index + 2..] {
                if a == b {
                    return Err(err(number, format!("'.couples' lists group '{a}' twice")));
                }
                self.couples.push([(*a).to_string(), (*b).to_string()]);
            }
            // A single name is a legal (and meaningful) declaration: it
            // isolates that group without coupling it to anything.
            // Remembered separately so that it, too, is checked against the
            // segments.
            self.couples_names.push((number, (*a).to_string()));
        }
        Ok(())
    }
    /// `.hole G<name> x1 y1 x2 y2` — a rectangular hole in that plane.
    fn apply_hole(
        &mut self,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        if tokens.len() != 6 {
            return Err(err(number, "expected .hole G<name> x1 y1 x2 y2"));
        }
        let name = tokens[1];
        let mut bounds = [0.0f64; 4];
        for (slot, token) in bounds.iter_mut().zip(&tokens[2..]) {
            *slot = parse_number(token, number)? * factor;
        }
        let plane = plane_named(&mut self.planes, name, ".hole", number)?;
        plane.plane.holes.push(Hole::Rect {
            lo: [bounds[0].min(bounds[2]), bounds[1].min(bounds[3])],
            hi: [bounds[0].max(bounds[2]), bounds[1].max(bounds[3])],
        });
        Ok(())
    }

    /// `.contact G<name> x1 y1 x2 y2 [nx=] [ny=] [ratio=]` — a contact
    /// region refining that rectangle of the plane's mesh.
    fn apply_contact(
        &mut self,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        if tokens.len() < 6 {
            return Err(err(
                number,
                "expected .contact G<name> x1 y1 x2 y2 [nx=…] [ny=…] [ratio=…]",
            ));
        }
        let name = tokens[1];
        let mut bounds = [0.0f64; 4];
        for (slot, token) in bounds.iter_mut().zip(&tokens[2..6]) {
            *slot = parse_number(token, number)? * factor;
        }
        let mut cells = [2usize; 2];
        let mut ratio = 2.0f64;
        for token in &tokens[6..] {
            let (key, raw_value) = parse_field(token, number)?;
            match key.as_str() {
                "nx" => cells[0] = parse_count(&raw_value, "nx", number)?,
                "ny" => cells[1] = parse_count(&raw_value, "ny", number)?,
                "ratio" => {
                    ratio = parse_number(&raw_value, number)?;
                    if ratio < 1.0 {
                        return Err(err(
                            number,
                            format!(
                                "contact decay ratio must be ≥ 1 (got {raw_value}); 1 is an ungraded, uniformly fine axis"
                            ),
                        ));
                    }
                }
                other => {
                    return Err(err(
                        number,
                        format!("unknown .contact field '{other}' (supported: nx, ny, ratio)"),
                    ));
                }
            }
        }
        let plane = plane_named(&mut self.planes, name, ".contact", number)?;
        plane.plane.contacts.push(ContactRegion::new(
            [bounds[0].min(bounds[2]), bounds[1].min(bounds[3])],
            [bounds[0].max(bounds[2]), bounds[1].max(bounds[3])],
            cells,
            ratio,
        ));
        Ok(())
    }

    /// `N<name> [x]=<v> [y]=<v> [z]=<v>` — a node, each coordinate falling
    /// back to its `.default`.
    fn apply_node(
        &mut self,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        let head = tokens[0];
        if self.unit.is_none() {
            return Err(err(number, "node before .units (lengths need a unit)"));
        }
        let mut coordinates = [self.defaults.x, self.defaults.y, self.defaults.z];
        for token in &tokens[1..] {
            let (key, raw_value) = parse_field(token, number)?;
            let value = parse_value(&key, &raw_value, factor, number)?;
            match key.as_str() {
                "x" => coordinates[0] = Some(value),
                "y" => coordinates[1] = Some(value),
                "z" => coordinates[2] = Some(value),
                other => {
                    return Err(err(
                        number,
                        format!("unknown node field '{other}' (supported: x, y, z)"),
                    ));
                }
            }
        }
        let position = match coordinates {
            [Some(x), Some(y), Some(z)] => [x, y, z],
            _ => {
                return Err(err(
                    number,
                    format!(
                        "node '{head}': a coordinate is unset and no .default covers it (set it, or .default x=… y=… z=…)"
                    ),
                ));
            }
        };
        self.positions.push(position);
        self.names.define(head, self.positions.len() - 1, number)?;
        Ok(())
    }

    /// `E<name> N<a> N<b> [field]=<v> …` — a segment between two nodes.
    fn apply_segment(
        &mut self,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        let head = tokens[0];
        if self.unit.is_none() {
            return Err(err(number, "element before .units (lengths need a unit)"));
        }
        if tokens.len() < 3 {
            return Err(err(number, "expected E<name> N<a> N<b> [field]=<value> …"));
        }
        let a = self.names.lookup(tokens[1], number)?;
        let b = self.names.lookup(tokens[2], number)?;
        one_conductivity(&tokens[3..], number)?;
        let mut w = self.defaults.w;
        let mut h = self.defaults.h;
        let mut nwinc = self.defaults.nwinc;
        let mut nhinc = self.defaults.nhinc;
        let mut rw = self.defaults.rw;
        let mut rh = self.defaults.rh;
        let mut width_dir = self.defaults.width_dir;
        let mut sigma = self.defaults.sigma;
        let mut group = String::new();
        for token in &tokens[3..] {
            let (key, raw_value) = parse_field(token, number)?;
            if key == "group" {
                // A name, not a number, and case-sensitive like the node
                // names — so it is taken before parse_value.
                group = raw_value;
                continue;
            }
            let value = parse_value(&key, &raw_value, factor, number)?;
            match key.as_str() {
                "w" => w = Some(value.abs()),
                "h" => h = Some(value.abs()),
                "nwinc" => nwinc = Some(value as usize),
                "nhinc" => nhinc = Some(value as usize),
                "rw" => rw = Some(value),
                "rh" => rh = Some(value),
                "wx" | "wy" | "wz" => {
                    let axis = width_dir_axis(&key).expect("matched a width-direction key");
                    width_dir[axis] = Some(value);
                }
                // Already converted to S/m by parse_value either way.
                "sigma" | "rho" => sigma = Some(value),
                "x" | "y" | "z" => {
                    return Err(err(
                        number,
                        format!("'{key}' is a node field, not a segment field"),
                    ));
                }
                other => {
                    return Err(err(
                        number,
                        format!("unknown field '{other}' (supported: {SEGMENT_FIELDS})"),
                    ));
                }
            }
        }
        let (w, h, sigma) = match (w, h, sigma) {
            (Some(w), Some(h), Some(sigma)) => (w, h, sigma),
            (None, _, _) => {
                return Err(err(
                    number,
                    format!("segment '{head}' has no width: set w= here or in .default"),
                ));
            }
            (_, None, _) => {
                return Err(err(
                    number,
                    format!("segment '{head}' has no height: set h= here or in .default"),
                ));
            }
            (_, _, None) => {
                return Err(err(
                    number,
                    format!(
                        "segment '{head}' has no conductivity: set sigma= or rho= here or in .default"
                    ),
                ));
            }
        };
        let width_dir = resolve_width_dir(width_dir, head, number)?;
        if let Some(direction) = width_dir {
            // Parallel to the declared centreline: caught here, on this
            // line, by the library's own orientation check.
            let [pa, pb] = [self.positions[a], self.positions[b]];
            let probe = Segment::new(
                Node::new(pa[0], pa[1], pa[2]),
                Node::new(pb[0], pb[1], pb[2]),
                w,
                h,
                sigma,
            )
            .with_width_dir(direction);
            if let Err(SegmentError::InvalidWidthDirection) = probe.basis() {
                return Err(err(
                    number,
                    format!(
                        "segment '{head}': width direction (wx, wy, wz) = ({}, {}, {}) is parallel to the segment and fixes no orientation",
                        direction[0], direction[1], direction[2]
                    ),
                ));
            }
        }
        self.segment_defs.push(SegmentSpec {
            a,
            b,
            width: w,
            height: h,
            sigma,
            width_dir,
            line: number,
        });
        self.subdivisions.push(AxisGrading::new(
            nwinc.unwrap_or(1),
            nhinc.unwrap_or(1),
            rw.unwrap_or(1.0),
            rh.unwrap_or(1.0),
        ));
        self.segment_groups.push(group);
        Ok(())
    }

    /// A `G` statement, in either grammar. The two are told apart by the
    /// shape of the first token after the name: a bare number starts the
    /// extension form, anything else (`x1=…`) the FastHenry corner-point
    /// form. A deck may mix them freely.
    fn apply_ground_plane(
        &mut self,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        if self.unit.is_none() {
            return Err(err(
                number,
                "ground plane before .units (lengths need a unit)",
            ));
        }
        if tokens
            .get(1)
            .is_some_and(|token| token.parse::<f64>().is_err())
        {
            return self.apply_plane_corner_form(tokens, factor, number);
        }
        self.apply_plane_extent_form(tokens, factor, number)
    }

    /// The FastHenry corner-point `G` grammar: three corners, `thick=`,
    /// `seg1=`/`seg2=`, and the `N…` / `hole …` / `contact …` clauses
    /// [`parse_plane_statement`] reads.
    fn apply_plane_corner_form(
        &mut self,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        let head = tokens[0];
        one_conductivity(&tokens[1..], number)?;
        let (spec, plane_nodes_here) =
            parse_plane_statement(head, &tokens[1..].join(" "), factor, &self.defaults, number)?;
        let index = self.planes.len();
        for (name, position, equipotential) in plane_nodes_here {
            self.positions.push(position);
            self.names.define(&name, self.positions.len() - 1, number)?;
            self.plane_nodes.push(PlaneNodeSpec {
                slot: self.positions.len() - 1,
                plane: index,
                equipotential,
                position,
                line: number,
            });
        }
        self.planes.push(spec);
        Ok(())
    }

    /// The extension `G` grammar: `G<name> x1 y1 z1 x2 y2 z2 t [nx=] [ny=]
    /// [nhinc=] [sigma=|rho=]` — extent, top surface `z`, thickness down.
    fn apply_plane_extent_form(
        &mut self,
        tokens: &[&str],
        factor: f64,
        number: usize,
    ) -> Result<(), ParseError> {
        let head = tokens[0];
        if tokens.len() < 8 {
            return Err(err(
                number,
                "expected G<name> x1 y1 z1 x2 y2 z2 thickness [nx=…] [ny=…] [nhinc=…], or the FastHenry corner-point form G<name> x1=… y1=… z1=… x2=… y2=… z2=… x3=… y3=… z3=… thick=… seg1=… seg2=…",
            ));
        }
        let mut corners = [None; 7];
        for (slot, token) in corners.iter_mut().zip(&tokens[1..8]) {
            *slot = Some(
                parse_number(token, number).map_err(|mut error| {
                    error.line = number;
                    error
                })? * factor,
            );
        }
        let mut nx = 1usize;
        let mut ny = 1usize;
        let mut nhinc = 1usize;
        let mut sigma = self.defaults.sigma;
        one_conductivity(&tokens[8..], number)?;
        for token in &tokens[8..] {
            let (key, raw_value) = parse_field(token, number)?;
            let value = parse_value(&key, &raw_value, factor, number)?;
            match key.as_str() {
                "nx" => nx = value as usize,
                "ny" => ny = value as usize,
                "nhinc" => nhinc = value as usize,
                // Already converted to S/m by parse_value either way.
                "sigma" | "rho" => sigma = Some(value),
                other => {
                    return Err(err(
                        number,
                        format!("unknown G field '{other}' (supported: nx, ny, nhinc, sigma, rho)"),
                    ));
                }
            }
        }
        let corners: Vec<f64> = corners.into_iter().flatten().collect();
        let [x1, y1, z1, x2, y2, z2, thickness] = [
            corners[0], corners[1], corners[2], corners[3], corners[4], corners[5], corners[6],
        ];
        if nx < 1 || ny < 1 {
            return Err(err(
                number,
                format!("ground plane '{head}' needs nx and ny >= 1 (got nx={nx}, ny={ny})"),
            ));
        }
        self.planes.push(PlaneSpec {
            name: head.to_string(),
            plane: GroundPlane {
                lo: [x1.min(x2), y1.min(y2)],
                hi: [x1.max(x2), y1.max(y2)],
                z_top: z1.max(z2),
                thickness,
                nx,
                ny,
                sigma: sigma.ok_or_else(|| {
                    err(
                        number,
                        format!(
                            "ground plane '{head}' has no conductivity: set sigma= or rho= here or in .default"
                        ),
                    )
                })?,
                holes: Vec::new(),
                contacts: Vec::new(),
                equipotentials: Vec::new(),
            },
            nhinc,
        });
        Ok(())
    }

    /// The deck-level checks that belong to no single directive: the ones a
    /// deck fails by *omission*. A missing `.end` or `.units` is reported on
    /// the deck's last physical line, the rest on line 0.
    fn check_complete(&self, last_number: usize) -> Result<(), ParseError> {
        if !self.ended {
            return Err(err(last_number, "deck has no .end directive"));
        }
        if self.unit.is_none() {
            return Err(err(
                last_number,
                format!(
                    "deck has no .units directive (units are mandatory: .units <unit>, one of {UNITS})"
                ),
            ));
        }
        if self.ports.is_empty() {
            return Err(err(0, "deck has no .external port"));
        }
        if self.frequencies.is_empty() {
            return Err(err(0, "deck has no .freq sweep"));
        }
        if self.segment_defs.is_empty() && self.planes.is_empty() {
            return Err(err(
                0,
                "deck has no conductor: no E segment and no G ground plane",
            ));
        }
        Ok(())
    }

    /// Assembles the read deck into a [`Deck`]: the deck-level checks, then
    /// `.equiv` compaction, ground-plane meshes, endpoint snapping, node and
    /// segment assembly, the discretization, and the coupling.
    ///
    /// `last_number` is [`FoldedDeck::last_number`] — the line a missing
    /// `.end` or `.units` is reported on.
    fn finish(self, last_number: usize) -> Result<Deck, ParseError> {
        let parse_wall = std::time::Instant::now();
        self.check_complete(last_number)?;
        let DeckBuilder {
            title,
            names,
            positions,
            segment_defs,
            subdivisions,
            segment_groups,
            couples,
            couples_names,
            couples_line,
            named_couples,
            mut ports,
            frequencies,
            planes,
            plane_nodes,
            alias_lines,
            ..
        } = self;

        // Apply .equiv: compact aliased-away nodes out of the geometry and
        // remap every reference. Resolution follows the alias chain, so a
        // reference declared *before* the .equiv directive lands on the
        // canonical node too — the deck is one circuit, not a timeline.
        let compaction = names.compaction(positions.len());
        eprintln!(
            "deck: finish: equiv compaction done, wall {:.1} s",
            parse_wall.elapsed().as_secs_f64()
        );
        let resolve = |slot: usize| -> usize {
            let mut id = slot;
            while let Some(&target) = names.aliases.get(&id) {
                id = target;
            }
            compaction[id].expect("a canonical node is live by construction")
        };
        // Assembly: ground-plane meshes first (their cell nodes and bars), then
        // the declared nodes, then the segments — with every endpoint that
        // lands in a plane's footprint snapped to the nearest live cell node.
        let mut geometry = Geometry::new();
        let (plane_meshes, plane_subdivisions) = build_plane_meshes(&planes, &mut geometry)?;
        let plane_bars = geometry.segment_count();
        // A contact area that ties nothing is a deck mistake whether or not
        // anything references it — the rectangle names metal the mesh does
        // not have — so it is checked here rather than only where it is used.
        for &node in &plane_nodes {
            if let Some(index) = node.equipotential {
                contact_area_node(&plane_meshes, node, index)?;
            }
        }
        let plane_node_at = plane_node_owners(&plane_nodes, &planes, &alias_lines, resolve)?;
        // Endpoint resolution: follow `.equiv` aliases to the canonical slot,
        // take that slot's position, and if it lands in a plane's footprint,
        // snap to the nearest live cell node. Plane ids are final here; the
        // declared nodes that are still referenced as themselves get their ids
        // below (a snapped endpoint's declared node is dropped, not orphaned).
        let live_positions: Vec<[f64; 3]> = {
            let live_count = compaction.iter().filter(|slot| slot.is_some()).count();
            let mut live = vec![[0.0_f64; 3]; live_count];
            for (slot, &position) in positions.iter().enumerate() {
                if let Some(index) = compaction[slot] {
                    live[index] = position;
                }
            }
            live
        };
        let snap = |slot: usize| -> Result<Option<usize>, ParseError> {
            let live = resolve(slot);
            if let Some(&node) = plane_node_at.get(&live) {
                // A named contact area attaches to the patch's own tie node,
                // not to whatever cell centre happens to be nearest: the two
                // differ whenever the rectangle is not centred on a cell.
                if let Some(index) = node.equipotential {
                    return Ok(Some(contact_area_node(&plane_meshes, node, index)?.0));
                }
                let (spec, centres) = &plane_meshes[node.plane];
                return Ok(Some(
                    spec.plane
                        .attach(centres, node.position)
                        .map_err(|error| ParseError {
                            line: 0,
                            message: error.to_string(),
                        })?
                        .0,
                ));
            }
            let position = live_positions[live];
            for (spec, centres) in &plane_meshes {
                if spec.plane.contains(position, 0.0) {
                    return Ok(Some(
                        spec.plane
                            .attach(centres, position)
                            .map_err(|error| ParseError {
                                line: 0,
                                message: error.to_string(),
                            })?
                            .0,
                    ));
                }
            }
            Ok(None)
        };

        // One endpoint: the plane node it snapped to (final id), or the live
        // declared slot it stays on (id assigned below).
        let endpoint = |slot: usize| -> Result<(Option<usize>, usize), ParseError> {
            Ok((snap(slot)?, resolve(slot)))
        };
        type End = (Option<usize>, usize);
        let segment_ends: Vec<(End, End, &SegmentSpec)> = segment_defs
            .iter()
            .map(|spec| Ok((endpoint(spec.a)?, endpoint(spec.b)?, spec)))
            .collect::<Result<_, ParseError>>()?;
        eprintln!(
            "deck: finish: segment endpoints resolved ({} segments), wall {:.1} s",
            segment_defs.len(),
            parse_wall.elapsed().as_secs_f64()
        );
        let port_ends: Vec<(End, End)> = ports
            .iter()
            .map(|port| Ok((endpoint(port.positive.0)?, endpoint(port.negative.0)?)))
            .collect::<Result<_, ParseError>>()?;

        // Live declared slots still referenced as themselves keep their nodes.
        let mut used = vec![false; positions.len()];
        let mark = |endpoint: &(Option<usize>, usize), used: &mut Vec<bool>| {
            if endpoint.0.is_none() {
                used[endpoint.1] = true;
            }
        };
        for (a, b, _) in &segment_ends {
            mark(a, &mut used);
            mark(b, &mut used);
        }
        for (a, b) in &port_ends {
            mark(a, &mut used);
            mark(b, &mut used);
        }
        let mut live_to_id = vec![usize::MAX; positions.len()];
        let mut next_id = geometry.nodes().len();
        for (slot, &position) in positions.iter().enumerate() {
            let Some(live) = compaction[slot] else {
                continue; // aliased away by .equiv
            };
            if !used[live] {
                continue; // every reference snapped into a plane
            }
            geometry
                .add_node(Node::new(position[0], position[1], position[2]))
                .map_err(|error| ParseError {
                    line: 0,
                    message: error.to_string(),
                })?;
            live_to_id[live] = next_id;
            next_id += 1;
        }

        eprintln!(
            "deck: finish: geometry nodes added, wall {:.1} s",
            parse_wall.elapsed().as_secs_f64()
        );
        let final_id = |endpoint: &(Option<usize>, usize)| -> usize {
            endpoint.0.unwrap_or(live_to_id[endpoint.1])
        };
        for (a, b, spec) in &segment_ends {
            let mut def = SegmentDef::new(
                NodeId(final_id(a)),
                NodeId(final_id(b)),
                spec.width,
                spec.height,
                spec.sigma,
            );
            if let Some(direction) = spec.width_dir {
                def = def.with_width_dir(direction);
            }
            // A segment the geometry rejects is reported on its own `E` line.
            geometry.add_segment(def).map_err(|error| ParseError {
                line: spec.line,
                message: error.to_string(),
            })?;
        }
        for (port, (positive, negative)) in ports.iter_mut().zip(&port_ends) {
            port.positive = NodeId(final_id(positive));
            port.negative = NodeId(final_id(negative));
        }
        eprintln!(
            "deck: finish: segments added, wall {:.1} s",
            parse_wall.elapsed().as_secs_f64()
        );

        let discretization = discretization_of(plane_subdivisions, subdivisions);

        let coupling = if named_couples {
            truncated_coupling(
                plane_bars,
                segment_groups,
                couples,
                &couples_names,
                couples_line,
                geometry.segment_count(),
            )?
        } else {
            Coupling::all_pairs()
        };

        Ok(Deck {
            title,
            geometry,
            ports,
            discretization,
            coupling,
            frequencies,
        })
    }
}

/// A built ground plane: its statement, and the cell-centre nodes its mesh
/// produced — what an endpoint landing in its footprint snaps to.
type PlaneMesh<'a> = (&'a PlaneSpec, Vec<Vec<Option<NodeId>>>);

/// Builds every declared ground plane's mesh into `geometry`, in deck order
/// and ahead of the declared nodes and segments. Returns each plane with the
/// cell-centre nodes its mesh produced, and one [`Subdivision`] per plane bar
/// in geometry order — every bar of a plane carrying that plane's `nhinc`
/// filaments through the thickness.
fn build_plane_meshes<'a>(
    planes: &'a [PlaneSpec],
    geometry: &mut Geometry,
) -> Result<(Vec<PlaneMesh<'a>>, Vec<Subdivision>), ParseError> {
    let mut plane_meshes: Vec<PlaneMesh> = Vec::new();
    let mut plane_subdivisions: Vec<Subdivision> = Vec::new();
    for spec in planes {
        let centres = spec
            .plane
            .build_into(geometry)
            .map_err(|error| ParseError {
                line: 0,
                message: error.to_string(),
            })?;
        plane_subdivisions.resize(geometry.segment_count(), Subdivision::new(1, spec.nhinc));
        plane_meshes.push((spec, centres));
    }
    Ok((plane_meshes, plane_subdivisions))
}

/// The tie node of a named contact area (`contact equiv_rect` / `contact
/// connection`): the one node every live cell inside its rectangle shares,
/// as [`build_plane_meshes`] built it.
///
/// A rectangle catching no live cell centre ties nothing, which is a deck
/// mistake rather than an invitation to land on a neighbouring cell — the
/// clause named metal the mesh does not have.
fn contact_area_node(
    plane_meshes: &[PlaneMesh],
    node: PlaneNodeSpec,
    index: usize,
) -> Result<NodeId, ParseError> {
    let (spec, centres) = &plane_meshes[node.plane];
    spec.plane
        .equipotential_node(centres, index)
        .map_err(|error| ParseError {
            line: 0,
            message: error.to_string(),
        })?
        .ok_or_else(|| {
            err(
                node.line,
                format!(
                    "ground plane '{}': the contact area named here covers no live cell of the plane's mesh, so it ties nothing; widen it, or raise 'seg1'/'seg2' (or refine it with a 'contact decay_rect' over the same rectangle) so the mesh has a cell centre inside it",
                    spec.name
                ),
            )
        })
}

/// Which plane each in-plane node's `.equiv` class belongs to, keyed by the
/// class's canonical (live) slot.
///
/// An in-plane node's `.equiv` class attaches to *its* plane, wherever the
/// class's canonical node happens to sit: joining a via's segment node to an
/// in-plane node is how a deck wires into a plane, and the join must not
/// depend on which of the two the directive named first. A class that
/// reaches the in-plane nodes of two *different* planes is rejected, on the
/// `.equiv` line that joined them.
fn plane_node_owners(
    plane_nodes: &[PlaneNodeSpec],
    planes: &[PlaneSpec],
    alias_lines: &HashMap<usize, usize>,
    resolve: impl Fn(usize) -> usize,
) -> Result<HashMap<usize, PlaneNodeSpec>, ParseError> {
    let mut plane_node_at: HashMap<usize, PlaneNodeSpec> = HashMap::new();
    for &node in plane_nodes {
        let live = resolve(node.slot);
        match plane_node_at.get(&live) {
            Some(other) if other.plane != node.plane => {
                return Err(err(
                    alias_lines.get(&node.slot).copied().unwrap_or(node.line),
                    format!(
                        "'.equiv' joins in-plane nodes of ground planes '{}' and '{}'; joining two planes to each other is not supported (connect them with a segment)",
                        planes[other.plane].name, planes[node.plane].name
                    ),
                ));
            }
            // A set joining a named contact area to a plain in-plane node
            // of the same plane lands on the contact *area*: attaching to
            // the nearest single cell instead would silently throw away the
            // very thing the clause declared.
            Some(other) if other.equipotential.is_none() && node.equipotential.is_some() => {
                plane_node_at.insert(live, node);
            }
            Some(_) => {}
            None => {
                plane_node_at.insert(live, node);
            }
        }
    }
    Ok(plane_node_at)
}

/// The deck's filament discretization: the plane bars' grids (first, in
/// geometry order) followed by the `E` lines' own, narrowed to the least
/// specific form that still describes every segment.
fn discretization_of(
    plane_subdivisions: Vec<Subdivision>,
    subdivisions: Vec<AxisGrading>,
) -> Discretization {
    let mut all: Vec<AxisGrading> = plane_subdivisions
        .into_iter()
        .map(AxisGrading::from)
        .collect();
    all.extend(subdivisions);
    // A ratio only matters on an axis cut into more than one filament.
    let graded = all.iter().any(|grid| {
        (grid.nw > 1 && grid.width_ratio != 1.0) || (grid.nh > 1 && grid.height_ratio != 1.0)
    });
    if graded {
        return Discretization::PerSegmentGraded(all);
    }
    let subdivisions: Vec<Subdivision> = all
        .iter()
        .map(|grid| Subdivision::new(grid.nw, grid.nh))
        .collect();
    let first = subdivisions[0];
    if subdivisions.iter().all(|&sub| sub == first) {
        Discretization::Uniform(first)
    } else {
        Discretization::PerSegment(subdivisions)
    }
}

/// The coupling a deck that named groups on `.couples` asks for.
///
/// Ground-plane bars come first in the geometry and carry no group tag, so
/// they share the default group with any untagged segment. `couples_line` is
/// the first `.couples` directive's line, which a validation failure of the
/// declaration as a whole is reported on.
fn truncated_coupling(
    plane_bars: usize,
    segment_groups: Vec<String>,
    couples: Vec<[String; 2]>,
    couples_names: &[(usize, String)],
    couples_line: Option<usize>,
    segment_count: usize,
) -> Result<Coupling, ParseError> {
    let mut groups = vec![String::new(); plane_bars];
    groups.extend(segment_groups);
    // A name no segment carries is a typo, not a silent no-op — and a name
    // declared on its own reaches no pair, so it is checked here rather than
    // by `Coupling::validate` below.
    for (line, name) in couples_names {
        if !groups.iter().any(|group| group == name) {
            return Err(err(
                *line,
                format!("'.couples' names group '{name}', which no segment carries"),
            ));
        }
    }
    let mut coupling = Coupling::truncated(groups);
    for [a, b] in couples {
        coupling = coupling.coupled(a, b);
    }
    coupling
        .validate(segment_count)
        .map_err(|error| err(couples_line.unwrap_or(0), error.to_string()))?;
    Ok(coupling)
}

/// Sourced from `finish`, which prebuilds the per-live-slot position table
/// (`live_positions`) in O(nodes) once — this function's per-endpoint
/// rescan of the compaction array made 2 M-node decks quadratically slow
/// (5 M endpoints x 2 M slots); removed rather than re-tuned.

/// The `.freq` decade sweep; see the module documentation. Integer decades
/// use exact `powi` so round decimal frequencies round-trip bitwise.
fn frequency_sweep(fmin: f64, fmax: f64, ndec: usize, line: usize) -> Result<Vec<f64>, ParseError> {
    if !(fmin.is_finite() && fmin >= 0.0 && fmax.is_finite() && fmax >= 0.0) {
        return Err(err(line, "frequencies must be finite and ≥ 0"));
    }
    if fmax < fmin {
        return Err(err(line, format!("fmax ({fmax}) is below fmin ({fmin})")));
    }
    if fmin == fmax {
        return Ok(vec![fmin]);
    }
    if fmin == 0.0 {
        return Err(err(
            line,
            "fmin = 0 is only allowed as the single-frequency case (.freq fmin=0 fmax=0 runs the DC solve)",
        ));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(text: &str) -> Deck {
        parse(text).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn minimal_deck_parses() {
        let deck = parse_ok(
            "\
.title minimal
.units m
n1 x=0 y=0 z=0
n2 x=1e-3 y=0 z=0
e1 n1 n2 w=1e-3 h=1e-4 sigma=5.8e7
.external n1 n2
.freq fmin=1e3 fmax=1e6 ndec=10
.end
",
        );
        assert_eq!(deck.title.as_deref(), Some("minimal"));
        assert_eq!(deck.geometry.nodes().len(), 2);
        assert_eq!(deck.geometry.segment_count(), 1);
        assert_eq!(deck.ports.len(), 1);
        assert_eq!(deck.ports[0].name.as_deref(), Some("n1/n2"));
        assert_eq!(deck.frequencies.len(), 31);
        assert_eq!(deck.frequencies[0], 1e3);
        assert_eq!(deck.frequencies[30], 1e6);
        assert_eq!(
            deck.discretization,
            Discretization::Uniform(Subdivision::SINGLE)
        );
    }

    /// The body of a valid deck, with no title line of any kind.
    const COMPAT_BODY: &str = "\
.units m
n1 x=0 y=0 z=0
n2 x=1e-3 y=0 z=0
e1 n1 n2 w=1e-3 h=1e-4 sigma=5.8e7
.external n1 n2
.freq fmin=1e3 fmax=1e3 ndec=1
.end
";

    const COMPAT: ParseOptions = ParseOptions {
        fasthenry_compat: true,
    };

    fn parse_compat(text: &str) -> Result<Deck, ParseError> {
        parse_with_options(text, COMPAT)
    }

    #[test]
    fn default_mode_rejects_prose_first_line() {
        let error = parse(&format!("A two-node test trace\n{COMPAT_BODY}"))
            .expect_err("prose on line 1 is not a directive by default");
        assert_eq!(error.line, 1);
        assert!(error.message.contains("unrecognized line 'A'"), "{error}");
        // `parse` is exactly the default options.
        assert_eq!(
            parse(COMPAT_BODY).unwrap(),
            parse_with_options(COMPAT_BODY, ParseOptions::default()).unwrap()
        );
        // And line 1 still participates in normal parsing: no implicit title.
        assert_eq!(parse_ok(COMPAT_BODY).title, None);
    }

    #[test]
    fn compat_mode_ignores_prose_first_line() {
        let deck = parse_compat(&format!("A two-node   test trace\n{COMPAT_BODY}"))
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(deck.title.as_deref(), Some("A two-node test trace"));
        let plain = parse_ok(COMPAT_BODY);
        assert_eq!(deck.geometry, plain.geometry);
        assert_eq!(deck.ports, plain.ports);
        assert_eq!(deck.frequencies, plain.frequencies);
    }

    #[test]
    fn compat_mode_ignores_directive_like_first_line() {
        // Line 1 is skipped whatever it holds: a `.units` there is swallowed,
        // so the deck then lacks `.units` — the documented cost of the mode.
        let error = parse_compat(COMPAT_BODY).expect_err(".units on line 1 is the title");
        assert!(error.message.contains(".units"), "{error}");
        // A `*` comment or a `.title` on line 1 is likewise just title text.
        let deck = parse_compat(&format!("* comment-looking title\n{COMPAT_BODY}")).unwrap();
        assert_eq!(deck.title.as_deref(), Some("* comment-looking title"));
        let deck = parse_compat(&format!(".title x\n{COMPAT_BODY}")).unwrap();
        assert_eq!(deck.title.as_deref(), Some(".title x"));
    }

    #[test]
    fn compat_mode_honors_later_title_directive() {
        let body = COMPAT_BODY.replace(".units m", ".units m\n.title explicit  name");
        let deck = parse_compat(&format!("prose title\n{body}")).unwrap();
        assert_eq!(deck.title.as_deref(), Some("explicit name"));
    }

    #[test]
    fn compat_mode_blank_first_line_is_an_empty_title() {
        for first in ["", "   ", "\t "] {
            let deck = parse_compat(&format!("{first}\n{COMPAT_BODY}"))
                .unwrap_or_else(|error| panic!("{first:?}: {error}"));
            assert_eq!(deck.title, None, "{first:?}");
        }
    }

    #[test]
    fn compat_mode_keeps_physical_line_numbers() {
        let body = COMPAT_BODY.replace("n2 x=1e-3", "n2 bogus=1");
        let error = parse_compat(&format!("title\n{body}")).unwrap_err();
        assert_eq!(error.line, 4, "{error}");
        // A continuation cannot extend the title line.
        let error = parse_compat(&format!("title\n+ more\n{COMPAT_BODY}")).unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.message.contains("continuation"), "{error}");
    }

    #[test]
    fn compat_mode_title_only_deck_has_no_end() {
        for text in ["just a title", "just a title\n", "just a title\n\n* note\n"] {
            let error = parse_compat(text).expect_err("no .end");
            assert!(
                error.message.contains("no .end directive"),
                "{text:?}: {error}"
            );
        }
        let error = parse_compat("").expect_err("empty deck");
        assert!(error.message.contains("no .end directive"), "{error}");
    }

    #[test]
    fn units_scale_lengths_and_conductivity() {
        let deck = parse_ok(
            "\
.units um
n1 x=0 y=0 z=0
n2 x=1000 y=0 z=0
e1 n1 n2 w=100 h=10 sigma=1.0
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        let segment = deck.geometry.segment(0).unwrap();
        assert!((segment.length() - 1e-3).abs() < 1e-18);
        assert!((segment.width - 100e-6).abs() < 1e-18);
        // sigma=1.0 per um: 1.0 / 1e-6 = 1e6 S/m.
        assert!((segment.sigma - 1e6).abs() < 1e-9);
    }

    #[test]
    fn copper_in_mm_units() {
        let deck = parse_ok(
            "\
.units mm
.default z=0 sigma=5.8e4
n1 x=0 y=0
n2 x=10 y=0
e1 n1 n2 w=0.2 h=0.035
.external n1 n2
.freq fmin=1e6 fmax=1e6 ndec=1
.end
",
        );
        let segment = deck.geometry.segment(0).unwrap();
        assert!((segment.sigma - 5.8e7).abs() < 5.8e7 * 1e-15);
        assert_eq!(segment.a.z, 0.0);
    }

    #[test]
    fn defaults_and_continuation_lines() {
        let deck = parse_ok(
            "\
.units mm
* comment line
.default z=0 w=0.1 h=0.01 sigma=5.8e4 nhinc=2
n1 x=0 y=0
n2 x=10 y=0
e1 n1 n2 nwinc=3
+ sigma=1.0
.external n1 n2
.freq fmin=1e6 fmax=1e6 ndec=1
.end
",
        );
        let segment = deck.geometry.segment(0).unwrap();
        assert!((segment.width - 0.1e-3).abs() < 1e-18);
        assert!((segment.sigma - 1.0 / 1e-3).abs() < 1e-12);
        assert_eq!(
            deck.discretization,
            Discretization::Uniform(Subdivision::new(3, 2))
        );
    }

    #[test]
    fn equiv_compacts_the_aliased_node_away() {
        let deck = parse_ok(
            "\
.units m
n1 x=0 y=0 z=0
n1b x=5 y=5 z=5
n2 x=1e-3 y=0 z=0
e1 n1b n2 w=1e-3 h=1e-3 sigma=1.0
.equiv n1 n1b
e2 n1b n2 w=1e-3 h=1e-3 sigma=1.0
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // The aliased node disappears; every reference — declared before or
        // after .equiv — resolves to the canonical node n1 at the origin.
        assert_eq!(deck.geometry.nodes().len(), 2);
        assert_eq!(
            deck.geometry.segment(0).unwrap().a,
            Node::new(0e0, 0e0, 0e0)
        );
        assert_eq!(
            deck.geometry.segment(1).unwrap().a,
            Node::new(0e0, 0e0, 0e0)
        );
    }

    #[test]
    fn equiv_joins_more_than_two_nodes() {
        // `.equiv a b c` is `.equiv a b` followed by `.equiv a c`: every
        // later name aliases to the first, so three declared nodes compact
        // down to the one canonical node.
        let deck = parse_ok(
            "\
.units m
n1 x=0 y=0 z=0
n2 x=5 y=5 z=5
n3 x=7 y=7 z=7
n4 x=1e-3 y=0 z=0
e1 n2 n4 w=1e-3 h=1e-3 sigma=1.0
.equiv n1 n2 n3
e2 n3 n4 w=1e-3 h=1e-3 sigma=1.0
.external n1 n4
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        assert_eq!(deck.geometry.nodes().len(), 2);
        assert_eq!(
            deck.geometry.segment(0).unwrap().a,
            Node::new(0e0, 0e0, 0e0)
        );
        assert_eq!(
            deck.geometry.segment(1).unwrap().a,
            Node::new(0e0, 0e0, 0e0)
        );
    }

    #[test]
    fn equiv_chain_rejects_a_node_repeated_with_itself() {
        let error = parse(
            "\
.units m
n1 x=0 y=0 z=0
n2 x=1 y=0 z=0
e1 n1 n2 w=1 h=1 sigma=1
.equiv n1 n2 n1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        )
        .unwrap_err();
        assert!(error.message.contains("with itself"));
    }

    #[test]
    fn external_accepts_a_port_name() {
        let deck = parse_ok(
            "\
.units m
n1 x=0 y=0 z=0
n2 x=1 y=0 z=0
e1 n1 n2 w=1 h=1 sigma=1.0
.external n1 n2 trace
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        assert_eq!(deck.ports[0].name.as_deref(), Some("trace"));
    }

    #[test]
    fn dc_single_frequency() {
        let deck = parse_ok(
            "\
.units m
n1 x=0 y=0 z=0
n2 x=1 y=0 z=0
e1 n1 n2 w=1 h=1 sigma=1.0
.external n1 n2
.freq fmin=0 fmax=0 ndec=1
.end
",
        );
        assert_eq!(deck.frequencies, vec![0.0]);
    }

    #[test]
    fn g_ground_plane_builds_mesh_and_holes() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp 0 0 0 10 6 0 0.035 nx=5 ny=3
.hole Gp 4.9 2.9 5.1 3.1
n1 x=1 y=0 z=0
n2 x=9 y=0 z=0
e1 n1 n2 w=0.2 h=0.035
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // 15 cells minus the holed one; 22 bars minus its 4 incident,
        // plus the declared segment.
        assert_eq!(deck.geometry.nodes().len(), 14);
        assert_eq!(deck.geometry.segment_count(), 18 + 1);
        // The segment endpoints land in the plane footprint and are
        // snapped off the declared nodes onto cell centres.
        let end_a = deck.geometry.segment(18).unwrap().a;
        assert_ne!(end_a, Node::new(1e-3, 0.0, 0.0));
    }

    #[test]
    fn contact_regions_grade_the_plane_mesh() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp 0 0 0 10 6 0 0.035 nx=5 ny=3
.contact Gp 4.5 2.5 5.5 3.5 nx=4 ny=4 ratio=2
n1 x=5 y=3 z=0.5
n2 x=5 y=3 z=0
e1 n1 n2 w=0.2 h=0.035
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // The deck's plane is exactly the library plane with the region.
        let mesh = GroundPlane {
            lo: [0.0, 0.0],
            hi: [10e-3, 6e-3],
            z_top: 0.0,
            thickness: 0.035e-3,
            nx: 5,
            ny: 3,
            sigma: 5.8e7,
            holes: Vec::new(),
            equipotentials: Vec::new(),
            contacts: vec![ContactRegion::new(
                [4.5e-3, 2.5e-3],
                [5.5e-3, 3.5e-3],
                [4, 4],
                2.0,
            )],
        }
        .mesh()
        .unwrap();
        assert!(mesh.nx() > 5 && mesh.ny() > 3, "the deck plane is refined");
        // The plane's cells, plus the via's top node (the bottom one
        // snapped onto a plane cell).
        assert_eq!(deck.geometry.nodes().len(), mesh.nx() * mesh.ny() + 1);
        assert_eq!(deck.geometry.segment_count(), mesh.bars() + 1);
        // The via at (5, 3) mm lands on a refined cell centre: within half
        // a 0.25 mm fine cell, not half a 2 mm background cell.
        let landing = deck.geometry.segment(mesh.bars()).unwrap().b;
        assert!((landing.x - 5e-3).abs() <= 0.125e-3 + 1e-12);
        assert!((landing.y - 3e-3).abs() <= 0.125e-3 + 1e-12);
    }

    #[test]
    fn contact_defaults_and_field_errors() {
        // Without fields: 2 x 2 cells at ratio 2.
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp 0 0 0 10 6 0 0.035 nx=5 ny=3
.contact Gp 4 2 6 4
n1 x=5 y=3 z=0.5
n2 x=5 y=3 z=0
e1 n1 n2 w=0.2 h=0.035
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        let mesh = GroundPlane {
            lo: [0.0, 0.0],
            hi: [10e-3, 6e-3],
            z_top: 0.0,
            thickness: 0.035e-3,
            nx: 5,
            ny: 3,
            sigma: 5.8e7,
            holes: Vec::new(),
            equipotentials: Vec::new(),
            contacts: vec![ContactRegion::new([4e-3, 2e-3], [6e-3, 4e-3], [2, 2], 2.0)],
        }
        .mesh()
        .unwrap();
        // The plane's cells, plus the via's top node (the bottom one
        // snapped onto a plane cell).
        assert_eq!(deck.geometry.nodes().len(), mesh.nx() * mesh.ny() + 1);

        let bad = |line: &str| {
            parse(&format!(
                "\
.units mm
.default sigma=5.8e4
Gp 0 0 0 10 6 0 0.035 nx=5 ny=3
{line}
n1 x=5 y=3 z=0.5
n2 x=5 y=3 z=0
e1 n1 n2 w=0.2 h=0.035
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
"
            ))
            .unwrap_err()
        };
        assert!(bad(".contact Gq 4 2 6 4")
            .message
            .contains("'.contact' names unknown ground plane 'Gq'"));
        assert!(bad(".contact Gp 4 2").message.contains("expected .contact"));
        assert!(bad(".contact Gp 4 2 6 4 decay=2")
            .message
            .contains("unknown .contact field 'decay'"));
        assert!(bad(".contact Gp 4 2 6 4 ratio=0.5")
            .message
            .contains("ratio must be ≥ 1"));
        assert!(bad(".contact Gp 4 2 6 4 nx=0")
            .message
            .contains("nx must be ≥ 1"));
        // A region that misses the footprint is a plane-assembly error.
        assert!(bad(".contact Gp 40 2 60 4")
            .message
            .contains("outside the plane footprint"));
    }

    #[test]
    fn g_plane_errors_carry_lines() {
        let error = parse(
            "\
.units mm
.default sigma=5.8e4
.hole Gq 0 0 1 1
.end
",
        )
        .unwrap_err();
        assert!(error.message.contains("unknown ground plane 'Gq'"));
        let error = parse(
            "\
.units mm
Gp 0 0 0 10 6 0 0.035 nx=0
.end
",
        )
        .unwrap_err();
        assert!(error.message.contains("nx"));
        let error = parse(
            "\
.units mm
Gp 0 0 0 10 6 0 0.035
.end
",
        )
        .unwrap_err();
        assert!(error.message.contains("conductivity"));
    }

    /// A deck around one `G` statement (always on line 3, continuation
    /// lines included) with a via landing on the plane at (5, 3).
    fn plane_deck(statement: &str, extra: &str) -> String {
        format!(
            "\
.units mm
.default sigma=5.8e4
{statement}
Nt x=5 y=3 z=0.5
Nb x=5 y=3 z=0
Ev Nt Nb w=0.2 h=0.2
{extra}
.external Nt Nb
.freq fmin=1 fmax=1 ndec=1
.end
"
        )
    }

    /// The FastHenry corner-point form builds the same plane as the
    /// extension form: same mesh, same hole, same contact region, same
    /// snapped landing — and the mid-thickness/top-surface difference in
    /// how the two name the plane's z is the only coordinate that moves.
    #[test]
    fn fasthenry_plane_statement_equals_the_extension_form() {
        let fasthenry = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole rect (0.5, 4.5, 0, 1.5, 5.5, 0)
+ contact rect (4, 2, 0, 6, 4, 0)",
            "",
        ));
        let extension = parse_ok(&plane_deck(
            "Gp 0 0 0.02 10 6 0.02 0.04 nx=5 ny=3",
            "\
.hole Gp 0.5 4.5 1.5 5.5
.contact Gp 4 2 6 4",
        ));
        assert_eq!(fasthenry.geometry, extension.geometry);
        assert_eq!(fasthenry.ports, extension.ports);
        assert_eq!(fasthenry.discretization, extension.discretization);
    }

    /// `hole point (x, y, z)` removes exactly the one cell whose own
    /// extent contains the point (issue #98) — see
    /// [`fasterhenry::plane::Hole::Point`] for the rule.
    #[test]
    fn hole_point_clause_removes_exactly_the_cell_containing_it() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole point (5.5, 3.5, 0)
n1 x=50 y=50 z=0
n2 x=51 y=50 z=0
e1 n1 n2 w=1 h=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // (5.5, 3.5) mm lies inside cell (2, 1), spanning x in [4, 6] and
        // y in [2, 4] mm: 15 − 1 = 14 plane nodes, plus the unrelated
        // segment's own 2 nodes (its endpoints are far outside the plane's
        // footprint, so neither one snaps onto it).
        assert_eq!(deck.geometry.nodes().len(), 14 + 2);
    }

    /// A point exactly on a shared cell boundary is the documented tie:
    /// every cell touching that edge is removed, not an arbitrarily chosen
    /// one (issue #98).
    #[test]
    fn hole_point_clause_on_a_shared_edge_removes_every_touching_cell() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole point (4, 3.5, 0)
n1 x=50 y=50 z=0
n2 x=51 y=50 z=0
e1 n1 n2 w=1 h=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // (4, 3.5) mm sits exactly on the edge shared by cells (1, 1) and
        // (2, 1) (x = 4 mm): the tie removes both, 15 − 2 = 13 plane
        // nodes, plus the unrelated segment's own 2.
        assert_eq!(deck.geometry.nodes().len(), 13 + 2);
    }

    /// `hole circle (x, y, z, r)` removes every cell whose centre lies at
    /// or inside `r` of the circle's centre (issue #98) — see
    /// [`fasterhenry::plane::Hole::Circle`] for the rule.
    #[test]
    fn hole_circle_clause_removes_cells_whose_centre_is_inside() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole circle (5, 1, 0, 2.5)
n1 x=50 y=50 z=0
n2 x=51 y=50 z=0
e1 n1 n2 w=1 h=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // Cell centres sit on a 2 mm grid: a 2.5 mm circle around (5, 1)
        // mm reaches its three grid neighbours 2 mm away — (3, 1), (7, 1)
        // in x and (5, 3) in y — but not the diagonal ones (2.83 mm away)
        // or the row's far ends (4 mm away). 4 of 15 cells removed, 11
        // plane nodes, plus the unrelated segment's own 2.
        assert_eq!(deck.geometry.nodes().len(), 11 + 2);
    }

    /// A hole `point`/`circle` that removes no cells at all — one that
    /// falls entirely inside an existing rectangular hole, and a circle
    /// small enough to sit between cell centres — is not an error: it is
    /// simply a no-op hole (issue #98).
    #[test]
    fn hole_point_and_circle_may_remove_zero_cells() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole rect (4, 2, 0, 6, 4, 0)
+ hole point (5, 3.9, 0)
+ hole circle (2, 3, 0, 0.1)
n1 x=50 y=50 z=0
n2 x=51 y=50 z=0
e1 n1 n2 w=1 h=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // The point falls inside the rect hole's cell (already removed);
        // the circle (0.1 mm radius around (2, 3) mm) is far from every
        // cell centre (nearest is (3, 3) mm, 1 mm away) so it covers none.
        // Only the rect hole's one cell is gone: 15 − 1 = 14 plane nodes,
        // plus the unrelated segment's own 2.
        assert_eq!(deck.geometry.nodes().len(), 14 + 2);
    }

    /// A circle wide enough to cover the whole plane removes every cell —
    /// an edge case distinct from the rectangle hole's degenerate-extent
    /// rejection, since a circle has no "corners shared" failure mode
    /// (issue #98).
    #[test]
    fn hole_circle_covering_the_whole_plane_removes_every_cell() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole circle (5, 3, 0, 100)
n1 x=50 y=50 z=0
n2 x=51 y=50 z=0
e1 n1 n2 w=1 h=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // No plane node survives; only the unrelated segment's own 2.
        assert_eq!(deck.geometry.nodes().len(), 2);
    }

    /// A negative radius is rejected on the statement's own line rather
    /// than silently accepted as "removes nothing" (issue #98).
    #[test]
    fn hole_circle_rejects_a_negative_radius() {
        let error = parse(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole circle (5, 3, 0, -1)",
            "",
        ))
        .unwrap_err();
        assert!(error.message.contains("r=-0.001"), "{error}");
        assert!(error.message.contains(">= 0"), "{error}");
    }

    /// `contact decay_rect` names the same rectangle as `contact rect`
    /// when its numbers say so: centred at (5, 3) mm, 2 mm across each
    /// way, cells no larger than 1 mm, decaying to the background cell.
    /// That is 2 × 2 fine cells at ratio 1/(1 − 1/2) = 2 — exactly the
    /// region `contact rect` and a bare `.contact` build.
    #[test]
    fn decay_rect_equals_the_rect_form_at_the_same_numbers() {
        let decay = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ contact decay_rect (5, 3, 0, 2, 2, 1, 1, -1, -1)",
            "",
        ));
        let rect = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ contact rect (4, 2, 0, 6, 4, 0)",
            "",
        ));
        assert_eq!(decay.geometry, rect.geometry);
        assert_eq!(decay.ports, rect.ports);
    }

    /// The fine cell count and the decay ratio are both derived per axis,
    /// so an anisotropic `decay_rect` grades x and y at different rates. A
    /// 2 × 2 mm rectangle asking for 1 mm cells across x and 0.1 mm across
    /// y is 2 × 20 fine cells decaying at 1/(1 − 1/2) = 2 across x and
    /// 1/(1 − 0.1/2) = 20/19 across y.
    #[test]
    fn decay_rect_derives_cells_and_ratio_per_axis() {
        let deck = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.035 seg1=5 seg2=3
+ contact decay_rect (5, 3, 0, 2, 2, 1, 0.1, -1, -1)",
            "",
        ));
        let plane = GroundPlane {
            lo: [0.0, 0.0],
            hi: [10e-3, 6e-3],
            z_top: 0.035e-3 / 2.0,
            thickness: 0.035e-3,
            nx: 5,
            ny: 3,
            sigma: 5.8e7,
            holes: Vec::new(),
            equipotentials: Vec::new(),
            contacts: vec![ContactRegion::graded_per_axis(
                [4e-3, 2e-3],
                [6e-3, 4e-3],
                [2, 20],
                [1.0 / (1.0 - 1e-3 / 2e-3), 1.0 / (1.0 - 0.1e-3 / 2e-3)],
            )],
        };
        let mesh = plane.mesh().unwrap();
        // The deck's plane is that plane: same graded mesh, same landing.
        assert_eq!(deck.geometry.nodes().len(), mesh.nx() * mesh.ny() + 1);
        assert_eq!(deck.geometry.segment_count(), mesh.bars() + 1);
        // …and it is genuinely anisotropic: putting x's ratio on both axes
        // keeps x as it was and coarsens y far sooner, so a single ratio
        // cannot stand in for the pair.
        let one_ratio = GroundPlane {
            contacts: vec![ContactRegion::new(
                [4e-3, 2e-3],
                [6e-3, 4e-3],
                [2, 20],
                plane.contacts[0].ratio[0],
            )],
            ..plane.clone()
        }
        .mesh()
        .unwrap();
        assert_eq!(mesh.nx(), one_ratio.nx(), "x decays at the same ratio");
        assert!(
            mesh.ny() > one_ratio.ny(),
            "y decays more gently on its own ratio: {} vs {}",
            mesh.ny(),
            one_ratio.ny()
        );
    }

    /// Every `decay_rect` parameter the reader cannot honour is rejected by
    /// name on the statement's own line; nothing is quietly approximated.
    #[test]
    fn decay_rect_parameter_errors() {
        let bad = |clause: &str| -> ParseError {
            parse(&plane_deck(
                &format!(
                    "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ {clause}"
                ),
                "",
            ))
            .unwrap_err()
        };
        for (clause, expected) in [
            // Eight values is the `rect`-plus-two shape a reader that
            // guessed the argument list would accept: it is not this one.
            (
                "contact decay_rect (4, 2, 0, 6, 4, 0, 2, 2)",
                "takes 9 values",
            ),
            ("contact decay_rect (5, 3, 0, 0, 2, 1, 1, -1, -1)", "xwidth"),
            ("contact decay_rect (5, 3, 0, 2, 2, 2, 1, -1, -1)", "xcell"),
            ("contact decay_rect (5, 3, 0, 2, 2, 1, 0, -1, -1)", "ycell"),
            (
                "contact decay_rect (5, 3, 0, 2, 2, 1, 1, 0, -1)",
                "xmaxcell=0",
            ),
            // The plane's background cell is 2 mm each way; a limit finer
            // than that asks for a mesh this engine grades toward, not
            // below.
            (
                "contact decay_rect (5, 3, 0, 2, 2, 1, 1, 1.5, -1)",
                "finer than ground plane 'Gp's own background cell",
            ),
            // The z is checked against the plane's slab, as `rect`'s is.
            (
                "contact decay_rect (5, 3, 3, 2, 2, 1, 1, -1, -1)",
                "is not in ground plane 'Gp'",
            ),
        ] {
            let error = bad(clause);
            assert_eq!(error.line, 3, "'{clause}' reports the statement's line");
            assert!(
                error.message.contains(expected),
                "'{clause}' must name what it rejects, got: {}",
                error.message
            );
        }
        // A limit at or above the background cell never binds.
        parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ contact decay_rect (5, 3, 0, 2, 2, 1, 1, 2, 2)",
            "",
        ));
    }

    /// The **documented** seven-value `contact rect (x, y, z, xwidth,
    /// ywidth, xcell, ycell)` is exactly `contact decay_rect` without its
    /// two outward limits — that is, with both `maxcell`s at the negative
    /// "no limit" sentinel. Issue #95; the decision is in
    /// `docs/fasthenry-compat.md`.
    #[test]
    fn contact_rect_is_a_decay_rect_without_the_limits() {
        // Deliberately anisotropic and not the corner form's default, so
        // the identity is not a coincidence of round numbers: 4 × 2 mm
        // about (5, 3), cells 2 mm across x and 0.5 mm across y.
        let rect = parse_ok(&refine_deck("contact rect (5, 3, 0, 4, 2, 2, 0.5)"));
        let decay = parse_ok(&refine_deck(
            "contact decay_rect (5, 3, 0, 4, 2, 2, 0.5, -1, -1)",
        ));
        assert_eq!(rect.geometry, decay.geometry);
        assert_eq!(rect.ports, decay.ports);
    }

    /// The two spellings are one clause: the six-value corner form's
    /// default 2 × 2 cells at ratio 2 *is* the seven-value form at
    /// `xcell = xwidth/2`, `ycell = ywidth/2`, so `contact rect (4, 2, 0,
    /// 6, 4, 0)` and `contact rect (5, 3, 0, 2, 2, 1, 1)` name the same
    /// region. Issue #95.
    #[test]
    fn contact_rect_two_spellings_are_one_region() {
        let centred = parse_ok(&refine_deck("contact rect (5, 3, 0, 2, 2, 1, 1)"));
        let corners = parse_ok(&refine_deck("contact rect (4, 2, 0, 6, 4, 0)"));
        assert_eq!(centred.geometry, corners.geometry);
        assert_eq!(centred.ports, corners.ports);
    }

    /// What the documented form buys over the corner form: the cell size
    /// itself. A 2 × 2 mm rectangle asking for 1 mm cells across x and 0.1
    /// mm across y is 2 × 20 fine cells decaying at 1/(1 − 1/2) = 2 across
    /// x and 1/(1 − 0.1/2) = 20/19 across y — a mesh the corner form,
    /// fixed at 2 × 2 cells at ratio 2, cannot ask for at all. Issue #95.
    #[test]
    fn contact_rect_seven_values_choose_the_cell_per_axis() {
        let deck = parse_ok(&refine_deck("contact rect (5, 3, 0, 2, 2, 1, 0.1)"));
        let plane = refine_test_plane(vec![ContactRegion::graded_per_axis(
            [4e-3, 2e-3],
            [6e-3, 4e-3],
            [2, 20],
            [1.0 / (1.0 - 1e-3 / 2e-3), 1.0 / (1.0 - 0.1e-3 / 2e-3)],
        )]);
        let mesh = plane.mesh().unwrap();
        assert_eq!(deck.geometry.nodes().len(), mesh.nx() * mesh.ny() + 1);
        assert_eq!(deck.geometry.segment_count(), mesh.bars() + 1);
        // The corner form over the same rectangle is coarser across y,
        // which is the whole point of reading the documented spelling.
        let corners = parse_ok(&refine_deck("contact rect (4, 2, 0, 6, 4, 0)"));
        assert!(
            deck.geometry.nodes().len() > corners.geometry.nodes().len(),
            "the chosen 0.1 mm y cell must refine further than the corner form's default: {} vs {}",
            deck.geometry.nodes().len(),
            corners.geometry.nodes().len()
        );
    }

    /// A `contact rect` value list that is neither six nor seven long is a
    /// line-numbered error naming **both** spellings — the cost of telling
    /// them apart by arity is paid here rather than by nudging the deck
    /// toward one of them. The seven-value form's own parameter errors name
    /// `'contact rect'`, not the `decay_rect` it is read as. Issue #95.
    #[test]
    fn contact_rect_arity_and_parameter_errors() {
        for (clause, expected) in [
            // Neither arity: the error names both.
            ("contact rect (5, 3, 0, 2, 2)", "takes 7 values"),
            (
                "contact rect (5, 3, 0, 2, 2)",
                "or 6 (x1, y1, z1, x2, y2, z2)",
            ),
            // Eight values is `decay_rect`'s list one short, not a `rect`.
            ("contact rect (5, 3, 0, 2, 2, 1, 1, -1)", "takes 7 values"),
            ("contact rect (5, 3, 0, 2, 2, 1, 1, -1)", "got 8"),
            // The seven-value form's widths are widths, not a corner…
            (
                "contact rect (5, 3, 0, 0, 2, 1, 1)",
                "'contact rect': xwidth=0 must be > 0",
            ),
            (
                "contact rect (5, 3, 0, 2, -1, 1, 1)",
                "'contact rect': ywidth=-0.001 must be > 0",
            ),
            // …and its cell must be smaller than its width, as
            // `decay_rect`'s is: the decay ratio is 1/(1 − cell/width).
            (
                "contact rect (5, 3, 0, 2, 2, 2, 1)",
                "'contact rect': xcell=0.002",
            ),
            (
                "contact rect (5, 3, 0, 2, 2, 1, 0)",
                "'contact rect': ycell=0",
            ),
            // The z is checked against the plane's slab, as the corner
            // form's is, and the error names this clause.
            (
                "contact rect (5, 3, 3, 2, 2, 1, 1)",
                "'contact rect': z=0.003",
            ),
            (
                "contact rect (5, 3, 3, 2, 2, 1, 1)",
                "is not in ground plane 'Gp'",
            ),
            // The six-value form still reports its own six-value list.
            ("contact rect (4, 2, 0, 6, 4)", "takes 7 values"),
            // Neither spelling names a node — the seven-value one no more
            // than the corner form the existing rejection test covers.
            (
                "contact rect Npad (5, 3, 0, 2, 2, 1, 1)",
                "'contact rect' takes no node name",
            ),
        ] {
            let error = parse(&refine_deck(clause)).unwrap_err();
            assert_eq!(error.line, 3, "'{clause}' reports the statement's line");
            assert!(
                error.message.contains(expected),
                "'{clause}' must name {expected}, got: {}",
                error.message
            );
        }
    }

    /// The unsupported-contact-shape message is user-facing documentation of
    /// the grammars this reader accepts, so the `contact rect` spelling it
    /// advertises has to be one the reader actually reads. Asserted by
    /// lifting the advertised clause straight out of the message and parsing
    /// it, which is what keeps the string from drifting away from
    /// `apply_clause` again (issue #95).
    #[test]
    fn the_unsupported_shape_message_advertises_a_readable_contact_rect() {
        let error = parse(&refine_deck("contact wibble (5, 3, 0)")).unwrap_err();
        assert!(
            error.message.contains("'contact wibble' is not supported"),
            "an unsupported shape names itself, got: {}",
            error.message
        );
        let advertised = "contact rect (x, y, z, xwidth, ywidth, xcell, ycell)";
        assert!(
            error.message.contains(advertised),
            "the message must advertise the documented seven-value spelling, got: {}",
            error.message
        );
        // The advertised parameter names, filled in, must parse — and mean
        // the documented centre-and-widths rectangle, not the corners.
        let filled = advertised
            .replace("x, y, z", "5, 3, 0")
            .replace("xwidth, ywidth", "2, 2")
            .replace("xcell, ycell", "1, 1");
        assert_eq!(filled, "contact rect (5, 3, 0, 2, 2, 1, 1)");
        assert_eq!(
            parse_ok(&refine_deck(&filled)).geometry,
            parse_ok(&refine_deck("contact rect (4, 2, 0, 6, 4, 0)")).geometry,
        );
    }

    /// The corner-point plane every `contact point` / `contact line` test
    /// below uses: 10 × 6 mm, 2 mm background cells each way, as a library
    /// value carrying the given contacts.
    fn refine_test_plane(contacts: Vec<ContactRegion>) -> GroundPlane {
        GroundPlane {
            lo: [0.0, 0.0],
            hi: [10e-3, 6e-3],
            z_top: 0.02e-3,
            thickness: 0.04e-3,
            nx: 5,
            ny: 3,
            sigma: 5.8e7,
            holes: Vec::new(),
            contacts,
            equipotentials: Vec::new(),
        }
    }

    /// The index of the cell whose extent (edges included) holds `value`.
    fn cell_holding(edges: &[f64], value: f64) -> usize {
        edges
            .windows(2)
            .position(|pair| pair[0] - 1e-15 <= value && value <= pair[1] + 1e-15)
            .expect("the value lies on the plane")
    }

    const REFINE_PLANE: &str = "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3";

    /// A deck whose plane is `REFINE_PLANE` carrying one extra clause.
    fn refine_deck(clause: &str) -> String {
        plane_deck(&format!("{REFINE_PLANE}\n+ {clause}"), "")
    }

    /// `contact point (x, y, z, xcell, ycell)` is one fine cell of exactly
    /// `xcell × ycell` centred on the point, graded outward at ratio 2 —
    /// the same region a `.contact` directive names by its corners. The
    /// point is then that cell's centre, so the via landing on it (the
    /// deck's `Nb` at (5, 3)) snaps to it exactly.
    #[test]
    fn contact_point_is_one_fine_cell_centred_on_the_point() {
        let point = parse_ok(&refine_deck("contact point (5, 3, 0, 0.5, 0.25)"));
        let directive = parse_ok(&plane_deck(
            REFINE_PLANE,
            ".contact Gp 4.75 2.875 5.25 3.125 nx=1 ny=1 ratio=2",
        ));
        assert_eq!(point.geometry, directive.geometry);
        assert_eq!(point.ports, directive.ports);

        let mesh = refine_test_plane(vec![ContactRegion::new(
            [4.75e-3, 2.875e-3],
            [5.25e-3, 3.125e-3],
            [1, 1],
            2.0,
        )])
        .mesh()
        .unwrap();
        let i = cell_holding(mesh.x_edges(), 5e-3);
        let j = cell_holding(mesh.y_edges(), 3e-3);
        assert!(mesh.dx(i) <= 0.5e-3 + 1e-15, "{}", mesh.dx(i));
        assert!(mesh.dy(j) <= 0.25e-3 + 1e-15, "{}", mesh.dy(j));
        let centre = mesh.centre(i, j);
        assert!((centre[0] - 5e-3).abs() < 1e-12 && (centre[1] - 3e-3).abs() < 1e-12);
        assert!(
            point
                .geometry
                .nodes()
                .iter()
                .any(|n| (n.x - 5e-3).abs() < 1e-12 && (n.y - 3e-3).abs() < 1e-12),
            "the landing snaps onto the refined cell's centre"
        );
    }

    /// A requested cell at or above the plane's background cell is
    /// already met by the background mesh, so that axis is clamped to the
    /// background cell rather than coarsened — and a point met on both
    /// axes changes nothing at all.
    ///
    /// A clamped axis spans the whole plane at the plane's own cell count
    /// (issue #116): the axis is left exactly as the plane meshes it, so
    /// the region a half-met point names is the plane's full 0 … 10 mm on
    /// x cut into its own 5 background cells, not a band around the point.
    #[test]
    fn contact_point_never_coarsens_the_background() {
        let plain = parse_ok(&plane_deck(REFINE_PLANE, ""));
        let met = parse_ok(&refine_deck("contact point (5, 3, 0, 2, 3)"));
        assert_eq!(met.geometry, plain.geometry);

        // x asks for 4 mm (met by the 2 mm background: clamped, so x spans
        // the whole plane at its own nx=5 cells), y for 0.5 mm.
        let half = parse_ok(&refine_deck("contact point (5, 3, 0, 4, 0.5)"));
        let directive = parse_ok(&plane_deck(
            REFINE_PLANE,
            ".contact Gp 0 2.75 10 3.25 nx=5 ny=1 ratio=2",
        ));
        assert_eq!(half.geometry, directive.geometry);
    }

    /// The x coordinates the deck's nodes sit at, deduplicated and sorted:
    /// a plane's x cell edges show up in its node positions, so two decks
    /// whose planes mesh x identically have the same ones.
    fn node_axis(deck: &Deck, axis: usize) -> Vec<f64> {
        let mut values: Vec<f64> = deck
            .geometry
            .nodes()
            .iter()
            .map(|n| if axis == 0 { n.x } else { n.y })
            .collect();
        values.sort_by(f64::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        values
    }

    /// A `contact point` met on one axis leaves that axis alone even when
    /// its coordinate does *not* land on a background grid line. x = 5.3 mm
    /// on this 2 mm-cell plane once cut a 4.3 … 6.3 mm band into x, moving
    /// every x edge off 0/2/4/6/8/10 mm for a request x had already met;
    /// now the clamped axis spans the plane and its edges are the plain
    /// plane's, to the last bit. Issue #116.
    #[test]
    fn contact_point_met_on_one_axis_leaves_an_unaligned_axis_alone() {
        let plain = parse_ok(&plane_deck(REFINE_PLANE, ""));
        let unaligned = parse_ok(&refine_deck("contact point (5.3, 3, 0, 4, 0.5)"));
        assert_eq!(node_axis(&unaligned, 0), node_axis(&plain, 0));
        // y *is* refined, so the two differ there — the assertion above is
        // about x being untouched, not about the clause doing nothing.
        assert_ne!(node_axis(&unaligned, 1), node_axis(&plain, 1));

        // The mesh itself: the clamped axis's edges are the background's.
        let plain_mesh = refine_test_plane(Vec::new()).mesh().unwrap();
        let refined_mesh = refine_test_plane(vec![ContactRegion::new(
            [0.0, 2.75e-3],
            [10e-3, 3.25e-3],
            [5, 1],
            2.0,
        )])
        .mesh()
        .unwrap();
        assert_eq!(refined_mesh.x_edges().len(), plain_mesh.x_edges().len());
        for (a, b) in refined_mesh.x_edges().iter().zip(plain_mesh.x_edges()) {
            assert!((a - b).abs() < 1e-15, "x edge {a} vs {b}");
        }
        // The point's own cell still honours the y cell it asked for.
        let j = cell_holding(refined_mesh.y_edges(), 3e-3);
        assert!(
            refined_mesh.dy(j) <= 0.5e-3 + 1e-15,
            "{}",
            refined_mesh.dy(j)
        );
    }

    /// A met axis costs nothing even when the deck refines that same axis
    /// somewhere else. The clamped band spans the whole plane, so it overlaps
    /// every other band on its axis, and the mesher's merge rule keeps the
    /// finer of two overlapping bands across the union of their extents: a
    /// met-on-x `contact point` beside a 0.1 mm region near (1, 1) mm once
    /// refined the entire 10 mm x axis to 0.1 mm — 100 cells, against the 11
    /// that region costs on its own. Issue #124.
    #[test]
    fn a_met_axis_costs_nothing_beside_a_finer_refinement_on_it() {
        let fine = ".contact Gp 0.95 0.95 1.05 1.05 nx=1 ny=1 ratio=2";
        let both = parse_ok(&plane_deck(
            &format!("{REFINE_PLANE}\n+ contact point (5.3, 3, 0, 4, 0.5)"),
            fine,
        ));
        let alone = parse_ok(&plane_deck(REFINE_PLANE, fine));
        // x: the met axis is meshed exactly as the 0.1 mm region alone
        // meshes it — the clause adds no x cells at all.
        assert_eq!(node_axis(&both, 0), node_axis(&alone, 0));
        // y: the cell the point did ask for is still there.
        assert_ne!(node_axis(&both, 1), node_axis(&alone, 1));

        // The same, counted on the mesh the two regions build.
        let fine_region = ContactRegion::new([0.95e-3, 0.95e-3], [1.05e-3, 1.05e-3], [1, 1], 2.0);
        // The x band `contact point (5.3, 3, 0, 4, 0.5)` clamps to.
        let met_on_x = ContactRegion::new([0.0, 2.75e-3], [10e-3, 3.25e-3], [5, 1], 2.0);
        let both_mesh = refine_test_plane(vec![met_on_x, fine_region])
            .mesh()
            .unwrap();
        let alone_mesh = refine_test_plane(vec![fine_region]).mesh().unwrap();
        assert_eq!(alone_mesh.nx(), 11);
        assert_eq!(
            both_mesh.nx(),
            alone_mesh.nx(),
            "not the 100 of a 0.1 mm x axis"
        );
    }

    /// The same for `contact line`, which shares the per-axis loop: a strip
    /// along x whose y cell is met by the background leaves y's edges as
    /// they are even though neither end sits on a y grid line, and a strip
    /// along y whose x cell is met leaves x's alone. Issue #116.
    #[test]
    fn contact_line_met_on_one_axis_leaves_an_unaligned_axis_alone() {
        let plain = parse_ok(&plane_deck(REFINE_PLANE, ""));
        // y = 3.1 mm is not a 2 mm y grid line (0/2/4/6), and ycell=3 mm is
        // met by the 2 mm background, so y is clamped; x is refined.
        let strip = parse_ok(&refine_deck("contact line (2, 3.1, 0, 8, 3.1, 0, 0.5, 3)"));
        let directive = parse_ok(&plane_deck(
            REFINE_PLANE,
            ".contact Gp 1.75 0 8.25 6 nx=13 ny=3 ratio=2",
        ));
        assert_eq!(strip.geometry, directive.geometry);
        assert_eq!(node_axis(&strip, 1), node_axis(&plain, 1));
        assert_ne!(node_axis(&strip, 0), node_axis(&plain, 0));

        // …and the other way round: a strip along y with x already met.
        let column = parse_ok(&refine_deck("contact line (5.3, 1, 0, 5.3, 5, 0, 4, 0.5)"));
        let column_directive = parse_ok(&plane_deck(
            REFINE_PLANE,
            ".contact Gp 0 0.75 10 5.25 nx=5 ny=9 ratio=2",
        ));
        assert_eq!(column.geometry, column_directive.geometry);
        assert_eq!(node_axis(&column, 0), node_axis(&plain, 0));
    }

    /// A point near the plane's edge: its region is clipped to the plane
    /// and widened over the sliver it leaves, and the widened band is cut
    /// into cells no coarser than the one requested — the cell holding the
    /// point still honours `xcell`.
    #[test]
    fn contact_point_at_the_plane_edge_keeps_its_cell_size() {
        let edge = parse_ok(&refine_deck("contact point (0.8, 3, 0, 1, 1)"));
        let directive = parse_ok(&plane_deck(
            REFINE_PLANE,
            ".contact Gp 0.3 2.5 1.3 3.5 nx=1 ny=1 ratio=2",
        ));
        // The same mesh, up to the last bit of `0.8 − 0.5` against `0.3`.
        let (a, b) = (edge.geometry.nodes(), directive.geometry.nodes());
        assert_eq!(a.len(), b.len());
        assert_eq!(
            edge.geometry.segment_count(),
            directive.geometry.segment_count()
        );
        for (p, q) in a.iter().zip(b) {
            assert!((p.x - q.x).abs() < 1e-15 && (p.y - q.y).abs() < 1e-15 && p.z == q.z);
        }
        let mesh = refine_test_plane(vec![ContactRegion::new(
            [0.3e-3, 2.5e-3],
            [1.3e-3, 3.5e-3],
            [1, 1],
            2.0,
        )])
        .mesh()
        .unwrap();
        let i = cell_holding(mesh.x_edges(), 0.8e-3);
        assert!(mesh.dx(i) <= 1e-3 + 1e-15, "{}", mesh.dx(i));
        // Exactly on the plane's corner is still on the plane.
        parse_ok(&refine_deck("contact point (0, 0, 0, 0.5, 0.5)"));
        parse_ok(&refine_deck("contact point (10, 6, 0, 0.5, 0.5)"));
    }

    /// `contact line` along x refines a one-cell-high strip covering the
    /// segment plus half a cell past each end; a zero-length line is
    /// exactly the `contact point` at its ends.
    #[test]
    fn contact_line_along_an_axis_is_a_padded_strip() {
        let strip = parse_ok(&refine_deck("contact line (2, 3, 0, 8, 3, 0, 0.5, 0.25)"));
        // x: 1.75 … 8.25 is 13 cells of 0.5 mm; y: one 0.25 mm cell.
        let directive = parse_ok(&plane_deck(
            REFINE_PLANE,
            ".contact Gp 1.75 2.875 8.25 3.125 nx=13 ny=1 ratio=2",
        ));
        assert_eq!(strip.geometry, directive.geometry);
        // The ends may be given either way round.
        let reversed = parse_ok(&refine_deck("contact line (8, 3, 0, 2, 3, 0, 0.5, 0.25)"));
        assert_eq!(reversed.geometry, strip.geometry);

        let degenerate = parse_ok(&refine_deck("contact line (5, 3, 0, 5, 3, 0, 0.5, 0.25)"));
        let point = parse_ok(&refine_deck("contact point (5, 3, 0, 0.5, 0.25)"));
        assert_eq!(degenerate.geometry, point.geometry);
    }

    /// A diagonal `contact line` refines its whole (padded) bounding box.
    /// On this engine's tensor-product mesh that is not a compromise: a
    /// refined band on one axis spans the plane on the other, so any
    /// refinement that covers the line — a chain of small rectangles
    /// included — has these same x- and y-bands, and their crossing is the
    /// box. The cost is stated: 13 × 9 fine cells here, where a mesh free
    /// to follow the diagonal would need about 13 + 9.
    #[test]
    fn contact_line_on_a_diagonal_refines_its_bounding_box() {
        let diagonal = parse_ok(&refine_deck("contact line (2, 1, 0, 8, 5, 0, 0.5, 0.5)"));
        let directive = parse_ok(&plane_deck(
            REFINE_PLANE,
            ".contact Gp 1.75 0.75 8.25 5.25 nx=13 ny=9 ratio=2",
        ));
        assert_eq!(diagonal.geometry, directive.geometry);

        // Every cell the line passes through honours the requested size.
        let mesh = refine_test_plane(vec![ContactRegion::new(
            [1.75e-3, 0.75e-3],
            [8.25e-3, 5.25e-3],
            [13, 9],
            2.0,
        )])
        .mesh()
        .unwrap();
        for step in 0..=600 {
            let t = f64::from(step) / 600.0;
            let (x, y) = (2e-3 + t * 6e-3, 1e-3 + t * 4e-3);
            let i = cell_holding(mesh.x_edges(), x);
            let j = cell_holding(mesh.y_edges(), y);
            assert!(mesh.dx(i) <= 0.5e-3 + 1e-15, "dx {} at x={x}", mesh.dx(i));
            assert!(mesh.dy(j) <= 0.5e-3 + 1e-15, "dy {} at y={y}", mesh.dy(j));
        }
    }

    /// Every `contact point` / `contact line` value the reader cannot
    /// honour is rejected by name on the statement's own line.
    #[test]
    fn contact_point_and_line_parameter_errors() {
        for (clause, expected) in [
            // The pre-#100 three-value form is a point with no cell size:
            // not guessed at.
            ("contact point (5, 3, 0)", "takes 5 values"),
            ("contact point (5, 3, 0, 1, 1, 1)", "takes 5 values"),
            ("contact line (1, 1, 0, 9, 5, 0, 0.2)", "takes 8 values"),
            ("contact point (5, 3, 0, 0, 1)", "xcell=0"),
            ("contact point (5, 3, 0, 1, -1)", "ycell=-0.001"),
            ("contact line (1, 1, 0, 9, 5, 0, 0.2, 0)", "ycell=0"),
            // z is checked against the plane's slab, as every clause's is.
            (
                "contact point (5, 3, 3, 1, 1)",
                "is not in ground plane 'Gp'",
            ),
            (
                "contact line (1, 1, 0, 9, 5, 3, 0.2, 0.2)",
                "is not in ground plane 'Gp'",
            ),
            // …and x/y against its footprint.
            (
                "contact point (11, 3, 0, 1, 1)",
                "is outside ground plane 'Gp'",
            ),
            (
                "contact line (1, 1, 0, 9, 7, 0, 0.2, 0.2)",
                "is outside ground plane 'Gp'",
            ),
        ] {
            let error = parse(&refine_deck(clause)).unwrap_err();
            assert_eq!(error.line, 3, "'{clause}' reports the statement's line");
            assert!(
                error.message.contains(expected),
                "'{clause}' must name what it rejects, got: {}",
                error.message
            );
        }
    }

    /// `contact circle` is rejected by name on the statement's own line,
    /// whatever value list it carries — there is no arity to get right,
    /// because no contact shape of that name is documented (`circle` is a
    /// *hole* shape). The error says that, and names the bounding-square
    /// alternative rather than leaving the reader to guess. Issue #109.
    #[test]
    fn contact_circle_is_rejected_by_name_and_names_the_bounding_square() {
        for clause in [
            // The shape as a `hole circle`'s values would spell it…
            "contact circle (5, 3, 0, 1)",
            // …and as the other `contact` shapes' cell sizes would.
            "contact circle (5, 3, 0, 1, 0.5, 0.5)",
            "contact circle (5, 3, 0, 1, 0.5)",
            "contact circle (5, 3, 1)",
        ] {
            let error = parse(&refine_deck(clause)).unwrap_err();
            assert_eq!(error.line, 3, "'{clause}' reports the statement's line");
            for expected in [
                "'contact circle' is not supported",
                "no contact shape of that name is documented",
                "hole circle (x, y, z, r)",
                "contact decay_rect (x, y, z, 2r, 2r, xcell, ycell, xmaxcell, ymaxcell)",
                "contact point (x, y, z, xcell, ycell)",
            ] {
                assert!(
                    error.message.contains(expected),
                    "'{clause}' must name {expected}, got: {}",
                    error.message
                );
            }
        }
    }

    /// The alternative that error names is the whole of what a disc could
    /// mean on this mesh, which is why nothing is lost by rejecting the
    /// shape: `contact decay_rect` over the disc's bounding square refines
    /// every cell the disc touches — and, because the mesh is a tensor
    /// product, the square's corners with it. That over-refinement is the
    /// disc's cost here (4/π of its area), not an approximation of the
    /// request: a refined x-band spans the plane in y, so any refinement
    /// covering the disc has these same bands and their crossing is the
    /// square. Issue #109, the same argument as the diagonal `contact
    /// line`'s.
    #[test]
    fn a_discs_bounding_square_is_what_refining_a_disc_costs() {
        // A 1 mm disc at the plane's centre, asking for 0.5 mm cells:
        // centre (5, 3), full widths 2r = 2 mm each way, 4 × 4 fine cells.
        let deck = parse_ok(&refine_deck(
            "contact decay_rect (5, 3, 0, 2, 2, 0.5, 0.5, -1, -1)",
        ));
        let ratio = 1.0 / (1.0 - 0.5 / 2.0);
        let plane = refine_test_plane(vec![ContactRegion::graded_per_axis(
            [4e-3, 2e-3],
            [6e-3, 4e-3],
            [4, 4],
            [ratio, ratio],
        )]);
        let mesh = plane.mesh().unwrap();
        assert_eq!(deck.geometry.nodes().len(), mesh.nx() * mesh.ny() + 1);
        assert_eq!(deck.geometry.segment_count(), mesh.bars() + 1);

        // Every cell holding a point inside the disc honours the request…
        let fine_at = |mesh: &fasterhenry::plane::PlaneMesh, x: f64, y: f64| {
            let i = cell_holding(mesh.x_edges(), x);
            let j = cell_holding(mesh.y_edges(), y);
            mesh.dx(i) <= 0.5e-3 + 1e-15 && mesh.dy(j) <= 0.5e-3 + 1e-15
        };
        for step in 0..720 {
            let angle = f64::from(step) * std::f64::consts::TAU / 720.0;
            for radius in [0.0, 0.5e-3, 0.999e-3] {
                let (x, y) = (5e-3 + radius * angle.cos(), 3e-3 + radius * angle.sin());
                assert!(fine_at(&mesh, x, y), "coarse cell at ({x}, {y})");
            }
        }
        // …and so does the bounding square's corner, which the disc misses
        // by a quarter of its own radius: that is the stated cost, 4/π of
        // the disc's area refined instead of the disc.
        let corner = [4.25e-3f64, 2.25e-3f64];
        assert!(
            (corner[0] - 5e-3).hypot(corner[1] - 3e-3) > 1e-3,
            "the sampled corner is outside the disc"
        );
        assert!(fine_at(&mesh, corner[0], corner[1]));

        // The rim itself lies on the square's edge, so a cell outside may
        // graze it — pad each width by one cell (as `contact point` pads by
        // half a cell on each side) and the rim's own cells are fine too.
        let padded = refine_test_plane(vec![ContactRegion::graded_per_axis(
            [3.75e-3, 1.75e-3],
            [6.25e-3, 4.25e-3],
            [5, 5],
            [1.0 / (1.0 - 0.5 / 2.5); 2],
        )])
        .mesh()
        .unwrap();
        let deck = parse_ok(&refine_deck(
            "contact decay_rect (5, 3, 0, 2.5, 2.5, 0.5, 0.5, -1, -1)",
        ));
        assert_eq!(deck.geometry.nodes().len(), padded.nx() * padded.ny() + 1);
        for step in 0..720 {
            let angle = f64::from(step) * std::f64::consts::TAU / 720.0;
            let (x, y) = (5e-3 + 1e-3 * angle.cos(), 3e-3 + 1e-3 * angle.sin());
            assert!(
                fine_at(&padded, x, y),
                "coarse cell on the rim at ({x}, {y})"
            );
        }
    }

    /// The same mesh up to the last bit of a coordinate: a trace's side
    /// lines sit at `y ± offset`, which need not round to the same double
    /// as the offset position written out in the deck.
    fn assert_same_mesh(a: &Deck, b: &Deck) {
        let (p, q) = (a.geometry.nodes(), b.geometry.nodes());
        assert_eq!(p.len(), q.len());
        assert_eq!(a.geometry.segment_count(), b.geometry.segment_count());
        for (m, n) in p.iter().zip(q) {
            assert!(
                (m.x - n.x).abs() < 1e-15 && (m.y - n.y).abs() < 1e-15 && m.z == n.z,
                "{m:?} vs {n:?}"
            );
        }
    }

    /// `contact trace` along x is exactly the documented five `contact
    /// line`s: under the trace (its centre line and both edges) cells no
    /// wider than `trace_width/2` across it, the lines `1.5 · trace_width`
    /// either side no wider than `trace_width`, and the trace's own length
    /// along it (no refinement there once it reaches the background cell).
    /// `scale_factor` has no effect on an axis-aligned trace, and the ends
    /// may be given either way round.
    #[test]
    fn contact_trace_along_x_is_the_documented_five_lines() {
        let trace = parse_ok(&refine_deck("contact trace (2, 3, 0, 8, 3, 0, 0.4, 1)"));
        let lines = parse_ok(&refine_deck(
            "contact line (2, 3, 0, 8, 3, 0, 6, 0.2)\n\
             + contact line (2, 3.2, 0, 8, 3.2, 0, 6, 0.2)\n\
             + contact line (2, 2.8, 0, 8, 2.8, 0, 6, 0.2)\n\
             + contact line (2, 3.6, 0, 8, 3.6, 0, 6, 0.4)\n\
             + contact line (2, 2.4, 0, 8, 2.4, 0, 6, 0.4)",
        ));
        assert_same_mesh(&trace, &lines);
        let plain = parse_ok(&plane_deck(REFINE_PLANE, ""));
        assert_ne!(
            trace.geometry, plain.geometry,
            "the trace refines the plane"
        );

        let scaled = parse_ok(&refine_deck("contact trace (2, 3, 0, 8, 3, 0, 0.4, 7)"));
        assert_eq!(
            scaled.geometry, trace.geometry,
            "scale_factor has no effect"
        );
        let reversed = parse_ok(&refine_deck("contact trace (8, 3, 0, 2, 3, 0, 0.4, 1)"));
        assert_eq!(reversed.geometry, trace.geometry);
    }

    /// The same along y, checked on the mesh itself: every cell under the
    /// trace is no wider (in x) than `trace_width/2`, every cell out to
    /// `1.5 · trace_width` either side no wider than `trace_width`, and a
    /// trace shorter than the background cell refines along its length to
    /// that length.
    #[test]
    fn contact_trace_along_y_refines_across_it_only() {
        let trace = parse_ok(&refine_deck("contact trace (5, 1, 0, 5, 5, 0, 0.4, 3)"));
        let lines = parse_ok(&refine_deck(
            "contact line (5, 1, 0, 5, 5, 0, 0.2, 4)\n\
             + contact line (5.2, 1, 0, 5.2, 5, 0, 0.2, 4)\n\
             + contact line (4.8, 1, 0, 4.8, 5, 0, 0.2, 4)\n\
             + contact line (5.6, 1, 0, 5.6, 5, 0, 0.4, 4)\n\
             + contact line (4.4, 1, 0, 4.4, 5, 0, 0.4, 4)",
        ));
        assert_same_mesh(&trace, &lines);

        // The same five regions through the library (y along the trace is
        // 4 mm, at or above the 2 mm background cell, so it is clamped).
        let regions = [(5.0, 0.2), (5.2, 0.2), (4.8, 0.2), (5.6, 0.4), (4.4, 0.4)]
            .iter()
            .map(|&(x, cell): &(f64, f64)| {
                ContactRegion::new(
                    [(x - cell / 2.0) * 1e-3, 0.0],
                    [(x + cell / 2.0) * 1e-3, 6e-3],
                    [1, 3],
                    2.0,
                )
            })
            .collect();
        let mesh = refine_test_plane(regions).mesh().unwrap();
        for step in 0..=240 {
            let x = 5e-3 + (f64::from(step) / 240.0 - 0.5) * 1.2e-3;
            let i = cell_holding(mesh.x_edges(), x);
            let limit = if (x - 5e-3).abs() <= 0.2e-3 + 1e-12 {
                0.2e-3
            } else {
                0.4e-3
            };
            assert!(mesh.dx(i) <= limit + 1e-15, "dx {} at x={x}", mesh.dx(i));
        }

        // A 1 mm trace against a 2 mm background: 1 mm cells along it.
        let short = parse_ok(&refine_deck("contact trace (5, 2, 0, 5, 3, 0, 0.4, 1)"));
        let short_lines = parse_ok(&refine_deck(
            "contact line (5, 2, 0, 5, 3, 0, 0.2, 1)\n\
             + contact line (5.2, 2, 0, 5.2, 3, 0, 0.2, 1)\n\
             + contact line (4.8, 2, 0, 4.8, 3, 0, 0.2, 1)\n\
             + contact line (5.6, 2, 0, 5.6, 3, 0, 0.4, 1)\n\
             + contact line (4.4, 2, 0, 4.4, 3, 0, 0.4, 1)",
        ));
        assert_same_mesh(&short, &short_lines);
    }

    /// Only the trace's own ends are checked against the plane's footprint:
    /// the refining lines beside a trace running along the plane's edge
    /// fall partly off it, and are clipped like any other region.
    #[test]
    fn contact_trace_along_the_plane_edge_is_accepted() {
        let edge = parse_ok(&refine_deck("contact trace (1, 0.1, 0, 9, 0.1, 0, 0.4, 1)"));
        let plain = parse_ok(&plane_deck(REFINE_PLANE, ""));
        assert_ne!(edge.geometry, plain.geometry);
        // Cells no taller than 0.2 mm under the trace put a cell centre
        // within 0.1 mm of y = 0.1 mm, which the 2 mm background (centres
        // at y = 1, 3, 5 mm) never has; the via's nodes sit at y = 3 mm.
        assert!(
            edge.geometry
                .nodes()
                .iter()
                .any(|n| (n.y - 0.1e-3).abs() <= 0.1e-3 + 1e-15),
            "a refined cell lies under the trace"
        );
        // Written out by hand, the line at y = −0.5 mm is off the plane.
        let error = parse(&refine_deck(
            "contact line (1, -0.5, 0, 9, -0.5, 0, 8, 0.4)",
        ))
        .unwrap_err();
        assert!(error.message.contains("is outside ground plane 'Gp'"));
    }

    /// A trace not parallel to x or y is rejected by name on the
    /// statement's own line: the public description of how `scale_factor`
    /// applies to it does not determine the cell size (see
    /// `docs/fasthenry-compat.md`, "Decision: a diagonal `contact trace`").
    /// The error names the explicit alternative, `contact line`.
    #[test]
    fn a_diagonal_contact_trace_is_rejected_and_names_the_alternative() {
        for clause in [
            "contact trace (1, 1, 0, 9, 5, 0, 0.2, 1)",
            "contact trace (1, 1, 0, 5, 5, 0, 0.2, 10)",
            "contact trace (5, 1, 0, 5.1, 5, 0, 0.2, 1)",
        ] {
            let error = parse(&refine_deck(clause)).unwrap_err();
            assert_eq!(error.line, 3, "'{clause}' reports the statement's line");
            for expected in [
                "'contact trace'",
                "is not parallel to x or y",
                "a diagonal 'contact trace' is not supported",
                "scale_factor",
                "contact line (x0, y0, z0, x1, y1, z1, xcell, ycell)",
            ] {
                assert!(
                    error.message.contains(expected),
                    "'{clause}' must name {expected}, got: {}",
                    error.message
                );
            }
        }
    }

    /// Every `contact trace` value the reader cannot honour is rejected by
    /// name on the statement's own line.
    #[test]
    fn contact_trace_parameter_errors() {
        for (clause, expected) in [
            ("contact trace (1, 3, 0, 9, 3, 0, 0.2)", "takes 8 values"),
            (
                "contact trace (1, 3, 0, 9, 3, 0, 0.2, 1, 1)",
                "takes 8 values",
            ),
            ("contact trace (1, 3, 0, 9, 3, 0, 0, 1)", "trace_width=0"),
            (
                "contact trace (1, 3, 0, 9, 3, 0, -0.2, 1)",
                "trace_width=-0.0002",
            ),
            ("contact trace (1, 3, 0, 9, 3, 0, 0.2, 0)", "scale_factor=0"),
            (
                "contact trace (1, 3, 0, 9, 3, 0, 0.2, -1)",
                "scale_factor=-1",
            ),
            ("contact trace (5, 3, 0, 5, 3, 0, 0.2, 1)", "zero length"),
            // z is checked against the plane's slab, as every clause's is.
            (
                "contact trace (1, 3, 0, 9, 3, 3, 0.2, 1)",
                "is not in ground plane 'Gp'",
            ),
            // …and x/y of the trace's own ends against its footprint.
            (
                "contact trace (1, 3, 0, 11, 3, 0, 0.2, 1)",
                "is outside ground plane 'Gp'",
            ),
        ] {
            let error = parse(&refine_deck(clause)).unwrap_err();
            assert_eq!(error.line, 3, "'{clause}' reports the statement's line");
            assert!(
                error.message.contains(expected),
                "'{clause}' must name what it rejects, got: {}",
                error.message
            );
        }
    }

    /// A deck that names a contact *area*: `contact equiv_rect` ties every
    /// cell centre inside its rectangle to the one node it names, and a
    /// reference to that node — here through `.equiv` — lands on the tie
    /// rather than on the nearest single cell centre.
    #[test]
    fn equiv_rect_ties_its_rectangle_to_the_node_it_names() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.035 seg1=5 seg2=3
+ contact equiv_rect Npad (5, 3, 0, 6, 3)
Nt x=5 y=3 z=0.5
Nb x=5 y=3 z=0.2
Ev Nt Nb w=0.2 h=0.2
.equiv Nb Npad
.external Nt Nb
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // The same plane through the library: cell centres sit at
        // x = 1, 3, 5, 7, 9 and y = 1, 3, 5 mm, so the 6 × 3 mm rectangle
        // about (5, 3) ties the three cells of the middle row with
        // x ∈ {3, 5, 7}.
        let plane = GroundPlane {
            lo: [0.0, 0.0],
            hi: [10e-3, 6e-3],
            z_top: 0.035e-3 / 2.0,
            thickness: 0.035e-3,
            nx: 5,
            ny: 3,
            sigma: 5.8e7,
            holes: Vec::new(),
            contacts: Vec::new(),
            equipotentials: vec![Equipotential::centred([5e-3, 3e-3], [6e-3, 3e-3])],
        };
        let mut geometry = Geometry::new();
        let centres = plane.build_into(&mut geometry).unwrap();
        let tie = plane.equipotential_node(&centres, 0).unwrap().unwrap();
        for (i, column) in centres.iter().enumerate().skip(1).take(3) {
            assert_eq!(column[1], Some(tie), "cell ({i}, 1) joins the tie");
        }
        // 15 cells become 13 nodes (three tied into one), plus the via's
        // top node; the two bars inside the patch are gone, and the via is
        // the one segment the deck adds.
        assert_eq!(deck.geometry.nodes().len(), 13 + 1);
        assert_eq!(deck.geometry.segment_count(), 22 - 2 + 1);
        assert_eq!(deck.geometry.nodes().len(), geometry.nodes().len() + 1);
        assert_eq!(deck.geometry.segment_count(), geometry.segment_count() + 1);
        // The via's lower end is the patch's tie node — the plane is built
        // first, so the library's node ids are the deck's.
        let via = deck.geometry.segment_defs()[deck.geometry.segment_count() - 1];
        assert_eq!(via.b, tie, "the via lands on the contact area");
        let node = deck.geometry.nodes()[tie.0];
        assert!((node.x - 5e-3).abs() < 1e-12, "{}", node.x);
        assert!((node.y - 3e-3).abs() < 1e-12, "{}", node.y);
    }

    /// The named node is an ordinary deck node: a port may drive it
    /// directly, without `.equiv`.
    #[test]
    fn a_contact_area_node_can_be_named_by_a_port() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.035 seg1=5 seg2=3
+ contact equiv_rect Npad (5, 3, 0, 6, 3)
+ Nfar (9, 5, 0)
.external Npad Nfar
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        assert_eq!(deck.geometry.nodes().len(), 13);
        assert_eq!(deck.ports.len(), 1);
        // The port's positive end is the tie node, its negative the nearest
        // live centre to the plain in-plane node — two different nodes.
        assert_ne!(deck.ports[0].positive, deck.ports[0].negative);
    }

    /// `contact connection` is the documented shorthand: the same rectangle
    /// tied by an `equiv_rect` *and* refined by a `decay_rect` whose cells
    /// are the widths divided by `ratio`, with no decay limit. Writing the
    /// two clauses out by hand must give exactly the same deck.
    #[test]
    fn connection_is_an_equiv_rect_plus_a_decay_rect() {
        let deck = |clauses: &str| {
            parse_ok(&format!(
                "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.035 seg1=5 seg2=3
+ {clauses}
Nt x=5 y=3 z=0.5
Nb x=5 y=3 z=0.2
Ev Nt Nb w=0.2 h=0.2
.equiv Nb Npad
.external Nt Nb
.freq fmin=1 fmax=1 ndec=1
.end
"
            ))
        };
        let shorthand = deck("contact connection Npad (5, 3, 0, 2, 2, 4)");
        let written_out = deck(
            "contact equiv_rect Npad (5, 3, 0, 2, 2)\n\
             + contact decay_rect (5, 3, 0, 2, 2, 0.5, 0.5, -1, -1)",
        );
        assert_eq!(shorthand.geometry, written_out.geometry);
        assert_eq!(shorthand.ports, written_out.ports);
        // …and the pairing does its job: the refined rectangle is cut into
        // `ratio` cells per axis, every one of them tied.
        let plane = GroundPlane {
            lo: [0.0, 0.0],
            hi: [10e-3, 6e-3],
            z_top: 0.035e-3 / 2.0,
            thickness: 0.035e-3,
            nx: 5,
            ny: 3,
            sigma: 5.8e7,
            holes: Vec::new(),
            contacts: vec![ContactRegion::graded_per_axis(
                [4e-3, 2e-3],
                [6e-3, 4e-3],
                [4, 4],
                [1.0 / (1.0 - 0.25), 1.0 / (1.0 - 0.25)],
            )],
            equipotentials: vec![Equipotential::centred([5e-3, 3e-3], [2e-3, 2e-3])],
        };
        let mut geometry = Geometry::new();
        let centres = plane.build_into(&mut geometry).unwrap();
        let tie = plane.equipotential_node(&centres, 0).unwrap().unwrap();
        let mesh = plane.mesh().unwrap();
        let tied = (0..mesh.nx())
            .flat_map(|i| (0..mesh.ny()).map(move |j| (i, j)))
            .filter(|&(i, j)| centres[i][j] == Some(tie))
            .count();
        assert_eq!(tied, 16, "the 4 × 4 fine cells of the rectangle are tied");
        assert_eq!(shorthand.geometry.nodes().len(), geometry.nodes().len() + 1);
        assert_eq!(
            shorthand.geometry.segment_count(),
            geometry.segment_count() + 1
        );
    }

    /// Every way a named contact area can be written wrong is rejected on
    /// the statement's own line, naming what it rejects.
    #[test]
    fn equiv_rect_and_connection_parameter_errors() {
        let bad = |clause: &str| -> ParseError {
            let deck = plane_deck(
                &format!(
                    "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ {clause}"
                ),
                "",
            );
            match parse(&deck) {
                Ok(_) => panic!("'{clause}' was accepted"),
                Err(error) => error,
            }
        };
        for (clause, expected) in [
            // The node name is the point of the clause, so its absence is
            // an error rather than an anonymous contact area.
            (
                "contact equiv_rect (5, 3, 0, 2, 2)",
                "names the node it ties",
            ),
            (
                "contact connection (5, 3, 0, 2, 2, 2)",
                "names the node it ties",
            ),
            (
                "contact equiv_rect Xpad (5, 3, 0, 2, 2)",
                "is not a node name",
            ),
            // The value list is the centre-and-widths spelling, not corners.
            (
                "contact equiv_rect Npad (4, 2, 0, 6, 4, 0)",
                "takes 5 values",
            ),
            (
                "contact equiv_rect Npad (5, 3, 0, 0, 2)",
                "xwidth=0 must be > 0",
            ),
            (
                "contact equiv_rect Npad (5, 3, 0, 2, -1)",
                "ywidth=-0.001 must be > 0",
            ),
            // The z is checked against the plane's slab, as every other
            // clause's is.
            (
                "contact equiv_rect Npad (5, 3, 3, 2, 2)",
                "is not in ground plane 'Gp'",
            ),
            // …and the rectangle's centre against its footprint.
            (
                "contact equiv_rect Npad (50, 3, 0, 2, 2)",
                "is outside ground plane 'Gp'",
            ),
            ("contact connection Npad (5, 3, 0, 2, 2)", "takes 6 values"),
            (
                "contact connection Npad (5, 3, 0, 2, 2, 1)",
                "ratio=1 must be > 1",
            ),
            // A node name on a clause that names no node is a mistake, not
            // a token to drop.
            (
                "hole rect Npad (0.5, 4.5, 0, 1.5, 5.5, 0)",
                "'hole rect' takes no node name",
            ),
            (
                "contact rect Npad (4, 2, 0, 6, 4, 0)",
                "'contact rect' takes no node name",
            ),
        ] {
            let error = bad(clause);
            assert_eq!(error.line, 3, "'{clause}' reports the statement's line");
            assert!(
                error.message.contains(expected),
                "'{clause}' must name what it rejects, got: {}",
                error.message
            );
        }

        // A name the deck already uses is the ordinary duplicate-node error.
        assert!(bad("contact equiv_rect Nt (5, 3, 0, 2, 2)")
            .message
            .contains("duplicate node name 'Nt'"));

        // A rectangle small enough to fall between two cell centres ties
        // nothing, and the deck is told so rather than silently landing on
        // a neighbouring cell.
        let error = parse(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ contact equiv_rect Npad (2, 2, 0, 0.5, 0.5)",
            "",
        ))
        .unwrap_err();
        assert_eq!(error.line, 3);
        assert!(
            error.message.contains("covers no live cell"),
            "got: {}",
            error.message
        );
    }

    /// `contact initial_grid (n1, n2)` is `seg1=n1 seg2=n2` — the
    /// equivalence the clause's own public description states — and the
    /// plane it builds is **not** the transposed one. The plane here is 10 ×
    /// 6 mm and the counts 5 × 3, so a transposition would be visible
    /// (issue #113).
    #[test]
    fn the_initial_grid_is_seg1_and_seg2_and_is_not_transposed() {
        let grid = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 contact initial_grid (5, 3)",
            "",
        ));
        let segments = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3",
            "",
        ));
        assert_eq!(grid.geometry, segments.geometry);
        let transposed = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=3 seg2=5",
            "",
        ));
        assert_ne!(
            grid.geometry, transposed.geometry,
            "5 × 3 and 3 × 5 must not be the same mesh, or this test proves nothing"
        );

        // Like `seg1`/`seg2`, the counts follow the *edges*: p1 → p2 runs
        // along y here, so the first value is the y count.
        let rotated = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=0 y2=6 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 contact initial_grid (3, 5)",
            "",
        ));
        assert_eq!(rotated.geometry, segments.geometry);
    }

    /// `contact initial_mesh_grid (n1, n2)` is that same initial grid plus
    /// the documented checkerboard: every cell whose indices, counted from
    /// the plane's own origin, are both even. Odd counts here (5 × 3), so
    /// the holes are the single interior column pair of an odd grid.
    #[test]
    fn the_meshed_initial_grid_holes_the_even_indexed_cells() {
        // 10 × 6 mm in 5 × 3 cells of 2 × 2 mm: the cells with 1-based
        // indices (2, 2) and (4, 2), centred at (3, 3) and (7, 3).
        let meshed = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 contact initial_mesh_grid (5, 3)",
            "",
        ));
        let by_hand = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole point (3, 3, 0)
+ hole point (7, 3, 0)",
            "",
        ));
        assert_eq!(meshed.geometry, by_hand.geometry);
        // The unmeshed form of the same grid keeps those cells.
        let solid = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 contact initial_grid (5, 3)",
            "",
        ));
        assert!(solid.geometry.segment_count() > meshed.geometry.segment_count());

        // An odd count is symmetric about the plane's centre, so numbering
        // it from either end selects the same cells: reversing the corner
        // order cannot move these holes.
        let reversed = parse_ok(&plane_deck(
            "\
Gp x1=10 y1=6 z1=0 x2=0 y2=6 z2=0 x3=0 y3=0 z3=0
+ thick=0.04 contact initial_mesh_grid (5, 3)",
            "",
        ));
        assert_eq!(reversed.geometry, meshed.geometry);
    }

    /// With an **even** count the checkerboard is no longer symmetric, and
    /// which end the cells are numbered from decides it: the numbering runs
    /// from the plane's own origin, its p1 corner.
    #[test]
    fn the_meshed_initial_grid_numbers_cells_from_the_p1_corner() {
        // 10 × 6 mm in 4 × 2 cells of 2.5 × 3 mm. Counted from p1 at the
        // origin, the holes are the 1-based (2, 2) and (4, 2) cells,
        // centred at (3.75, 4.5) and (8.75, 4.5).
        let meshed = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 contact initial_mesh_grid (4, 2)",
            "",
        ));
        let by_hand = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=4 seg2=2
+ hole point (3.75, 4.5, 0)
+ hole point (8.75, 4.5, 0)",
            "",
        ));
        assert_eq!(meshed.geometry, by_hand.geometry);

        // The same plane with p1 at the opposite corner: the same grid,
        // numbered from the other end, so the holes mirror to the 1-based
        // (2, 2) and (4, 2) cells counted from (10, 6) — centred at (6.25,
        // 1.5) and (1.25, 1.5).
        let reversed = parse_ok(&plane_deck(
            "\
Gp x1=10 y1=6 z1=0 x2=0 y2=6 z2=0 x3=0 y3=0 z3=0
+ thick=0.04 contact initial_mesh_grid (4, 2)",
            "",
        ));
        let reversed_by_hand = parse_ok(&plane_deck(
            "\
Gp x1=10 y1=6 z1=0 x2=0 y2=6 z2=0 x3=0 y3=0 z3=0
+ thick=0.04 seg1=4 seg2=2
+ hole point (6.25, 1.5, 0)
+ hole point (1.25, 1.5, 0)",
            "",
        ));
        assert_eq!(reversed.geometry, reversed_by_hand.geometry);
        assert_ne!(
            reversed.geometry, meshed.geometry,
            "an even count's checkerboard must move when the numbering does"
        );
    }

    /// A meshed grid one cell wide has no even index on that axis, and so
    /// no hole: the checkerboard needs a cell on either side of each hole.
    #[test]
    fn the_meshed_initial_grid_of_a_single_row_has_no_holes() {
        let meshed = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 contact initial_mesh_grid (5, 1)",
            "",
        ));
        let solid = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=1",
            "",
        ));
        assert_eq!(meshed.geometry, solid.geometry);
    }

    /// The initial grid and `seg1`/`seg2` are the same statement, so a
    /// plane that gives both — in either order — is a line-numbered error
    /// rather than a silent race between them, and so is a second initial
    /// grid or a malformed value list.
    #[test]
    fn the_initial_grid_and_seg1_seg2_are_one_statement() {
        let bad = |body: &str| -> ParseError {
            parse(&plane_deck(
                &format!(
                    "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 {body}"
                ),
                "",
            ))
            .unwrap_err()
        };
        for (body, expected) in [
            (
                "seg1=5 seg2=3 contact initial_grid (5, 3)",
                "both set this plane's cell counts",
            ),
            (
                "contact initial_grid (5, 3) seg1=5",
                "both set this plane's cell counts",
            ),
            (
                "contact initial_mesh_grid (5, 3) seg2=3",
                "both set this plane's cell counts",
            ),
            (
                "contact initial_grid (5, 3) contact initial_mesh_grid (5, 3)",
                "has already set on this statement",
            ),
            ("contact initial_grid (5)", "takes 2 values"),
            ("contact initial_grid (5, 3, 1)", "takes 2 values"),
            ("contact initial_mesh_grid (5, 0)", "must be ≥ 1"),
            ("contact initial_grid (5, two)", "cell count along p2→p3"),
            // The clause names no node, like every clause but the two
            // contact-area ones.
            (
                "contact initial_grid Npad (5, 3)",
                "'contact initial_grid' takes no node name",
            ),
        ] {
            let error = bad(body);
            assert_eq!(error.line, 3, "'{body}' reports the statement's line");
            assert!(
                error.message.contains(expected),
                "'{body}' must name what it rejects, got: {}",
                error.message
            );
        }
        // A plane whose only cell counts come from the initial grid needs
        // no `seg1`/`seg2` at all — that is the whole point of the clause.
        assert!(parse(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04",
            ""
        ))
        .unwrap_err()
        .message
        .contains("has no 'seg1'"));
    }

    /// `file=` names the plane's nonuniform-discretization *hierarchy* file —
    /// an input, not a dump of the finished mesh — and `file=NONE` says there
    /// is no such file, so the hierarchy is a single root cell discretized at
    /// run time from the statement's own clauses. That is exactly what this
    /// reader always does, so `file=NONE` is accepted as the no-op it is and
    /// the documented spelling of an initial grid reads; a *named* file is
    /// still rejected, as the input this reader does not read (issue #122).
    #[test]
    fn file_none_is_the_documented_no_op_and_a_named_hierarchy_file_is_rejected() {
        // The public description's own equivalence, written out in full:
        // `seg1=10 seg2=12` "could be replaced with `file=NONE contact
        // initial_grid (10,12)`".
        let documented = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 file=NONE contact initial_grid (10, 12)",
            "",
        ));
        let segments = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=10 seg2=12",
            "",
        ));
        assert_eq!(documented.geometry, segments.geometry);

        // Accepting it drops nothing: the token says "no hierarchy file",
        // which is the only case this reader has, so a plane carrying it is
        // the same plane. Spelled case-insensitively, as every other token
        // this reader interprets is.
        let with_marker = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 file=none",
            "",
        ));
        let without = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3",
            "",
        ));
        assert_eq!(with_marker.geometry, without.geometry);

        // A named file is a discretization hierarchy this reader cannot
        // read, and the error says so — naming the file, the accepted
        // alternative, and never calling it an output this engine lacks.
        let error = parse(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 file=plane.mat",
            "",
        ))
        .unwrap_err();
        assert_eq!(error.line, 3);
        assert!(
            error.message.contains("'file=plane.mat'"),
            "the rejection names the file it will not read, got: {}",
            error.message
        );
        assert!(
            error.message.contains("file=NONE"),
            "the rejection names the accepted spelling, got: {}",
            error.message
        );
        assert!(
            !error.message.contains("output"),
            "'file' is an input, not an output option, got: {}",
            error.message
        );
    }

    /// `seg1` counts cells along `p1 → p2` and `seg2` along `p2 → p3`,
    /// whichever axis each of those edges runs along.
    #[test]
    fn seg1_and_seg2_follow_the_edges_not_the_axes() {
        // p1 → p2 runs along y here, so seg1 is the y count.
        let rotated_corners = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=0 y2=6 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=3 seg2=5",
            "",
        ));
        let extension = parse_ok(&plane_deck("Gp 0 0 0.02 10 6 0.02 0.04 nx=5 ny=3", ""));
        assert_eq!(rotated_corners.geometry, extension.geometry);
    }

    /// The same 10 × 6 mm plane, 5 × 3 cells, written with its first edge
    /// along global x and along global y. `seg1`/`seg2` are swapped to match,
    /// so the two statements describe one physical plane (that much is
    /// `seg1_and_seg2_follow_the_edges_not_the_axes`).
    fn oriented_plane_decks(x_clause: &str, y_clause: &str) -> (Deck, Deck) {
        let deck = |statement: &str, clause: &str| {
            parse_ok(&plane_deck(&format!("{statement}\n+ {clause}"), ""))
        };
        (
            deck(
                "Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0\n+ thick=0.04 seg1=5 seg2=3",
                x_clause,
            ),
            deck(
                "Gp x1=0 y1=0 z1=0 x2=0 y2=6 z2=0 x3=10 y3=6 z3=0\n+ thick=0.04 seg1=3 seg2=5",
                y_clause,
            ),
        )
    }

    /// Every `contact` clause carrying an `x…`/`y…` pair of lengths states
    /// that pair in the **plane's own** coordinate system — index 0 along
    /// `p1 → p2`, index 1 along `p2 → p3` — which is what the public memo
    /// says of `contact point` ("the width of the cell in the plane
    /// coordinate system's x-direction") and which `line`, `decay_rect`,
    /// `equiv_rect` and `connection` inherit from it (issue #118), as does
    /// the documented seven-value `contact rect`, which is exactly a
    /// `decay_rect` without the outward limits (issue #95) and so turns
    /// with the plane through that same code. So on a plane whose first
    /// edge runs along global **y**, the pair is transposed relative to
    /// global x/y.
    ///
    /// Each case writes one physical plane twice — first edge along x, then
    /// along y with the clause's pair swapped — and the two decks must build
    /// the same geometry. Every pair here is anisotropic, so leaving it
    /// *unswapped* must build a different one: without that second
    /// assertion the test would pass on a reader that ignored the plane's
    /// axes entirely.
    #[test]
    fn contact_cell_sizes_follow_the_plane_axes_not_the_global_ones() {
        for (x_clause, y_clause) in [
            // A point's two cell sizes.
            (
                "contact point (5, 3, 0, 0.5, 1.5)",
                "contact point (5, 3, 0, 1.5, 0.5)",
            ),
            // A line's — its ends stay global, only the pair turns.
            (
                "contact line (1, 3, 0, 9, 3, 0, 0.5, 1.5)",
                "contact line (1, 3, 0, 9, 3, 0, 1.5, 0.5)",
            ),
            // `decay_rect`'s widths and its cells, both anisotropic.
            (
                "contact decay_rect (5, 3, 0, 4, 2, 1, 0.25, -1, -1)",
                "contact decay_rect (5, 3, 0, 2, 4, 0.25, 1, -1, -1)",
            ),
            // The documented seven-value `contact rect` (issue #95) is
            // `decay_rect` without the outward limits, so it turns with the
            // plane through the same code, not a special case of its own.
            (
                "contact rect (5, 3, 0, 4, 2, 1, 0.25)",
                "contact rect (5, 3, 0, 2, 4, 0.25, 1)",
            ),
            // A contact area's widths: 10 × 2 mm ties the five cells of one
            // row, 2 × 10 mm the three of one column, so a transposition
            // changes the node count and not merely which cells are tied.
            (
                "contact equiv_rect Npad (5, 3, 0, 10, 2)",
                "contact equiv_rect Npad (5, 3, 0, 2, 10)",
            ),
            // …and the grouped clause built on both halves at once.
            (
                "contact connection Npad (5, 3, 0, 4, 2, 4)",
                "contact connection Npad (5, 3, 0, 2, 4, 4)",
            ),
        ] {
            let (along_x, along_y) = oriented_plane_decks(x_clause, y_clause);
            assert_eq!(
                along_x.geometry, along_y.geometry,
                "'{y_clause}' on a plane whose first edge runs along y must \
                 mesh like '{x_clause}' on the same plane written along x"
            );
            let (_, unswapped) = oriented_plane_decks(x_clause, x_clause);
            assert_ne!(
                along_x.geometry, unswapped.geometry,
                "'{x_clause}' is anisotropic, so reading its pair along \
                 global x/y on a plane whose first edge runs along y would \
                 be a visible transposition"
            );
        }
    }

    /// An isotropic request cannot tell the two readings apart, and must
    /// come out identical either way round — the mapping is a permutation,
    /// not an extra refinement.
    #[test]
    fn a_square_contact_request_is_orientation_independent() {
        for clause in [
            "contact point (5, 3, 0, 0.5, 0.5)",
            "contact decay_rect (5, 3, 0, 2, 2, 0.5, 0.5, -1, -1)",
            "contact rect (5, 3, 0, 2, 2, 0.5, 0.5)",
            "contact equiv_rect Npad (5, 3, 0, 4, 4)",
            "contact connection Npad (5, 3, 0, 2, 2, 4)",
            // `contact trace` carries no `x…`/`y…` pair at all — one
            // `trace_width`, and a direction read from its own global ends —
            // and its expansion is symmetric in the two axes, so the plane's
            // orientation cannot transpose it either (issue #118).
            "contact trace (1, 3, 0, 9, 3, 0, 1, 1)",
        ] {
            let (along_x, along_y) = oriented_plane_decks(clause, clause);
            assert_eq!(
                along_x.geometry, along_y.geometry,
                "'{clause}' asks for the same mesh whichever edge is first"
            );
        }
    }

    /// The per-axis errors name the axis **the deck** wrote, not the global
    /// one it lands on: on a plane whose first edge runs along global y, the
    /// deck's `xmaxcell` is measured along y and is what `seg1` counts.
    /// Here the plane is 10 × 6 mm with `seg1=3` (along y: a 2 mm
    /// background cell) and `seg2=10` (along x: 1 mm), so `xmaxcell=1.5`
    /// is finer than its own axis's background cell while `ymaxcell=9`
    /// binds nothing.
    #[test]
    fn a_rotated_planes_decay_limit_error_names_the_decks_own_axis() {
        let error = parse(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=0 y2=6 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=3 seg2=10
+ contact decay_rect (5, 3, 0, 4, 2, 1, 0.5, 1.5, 9)",
            "",
        ))
        .unwrap_err();
        assert_eq!(error.line, 3);
        assert!(
            error.message.contains("xmaxcell=") && error.message.contains("'seg1'"),
            "got: {}",
            error.message
        );
    }

    /// An in-plane node wires a segment into the plane whichever side of
    /// `.equiv` it is named on — the join is a circuit, not a timeline.
    #[test]
    fn in_plane_nodes_join_by_equiv_either_way() {
        let statement = "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ Nland (5, 3, 0)";
        let direct = parse_ok(&plane_deck(statement, ""));
        // The via's own foot node, joined to the in-plane node each way
        // round: both must land on the plane, not at the foot node.
        let plane_first = parse_ok(&format!(
            "\
.units mm
.default sigma=5.8e4
{statement}
Nt x=5 y=3 z=0.5
Nb x=5 y=3 z=0
Ev Nt Nb w=0.2 h=0.2
.equiv Nland Nb
.external Nt Nb
.freq fmin=1 fmax=1 ndec=1
.end
"
        ));
        let node_first = parse_ok(&format!(
            "\
.units mm
.default sigma=5.8e4
{statement}
Nt x=5 y=3 z=0.5
Nb x=5 y=3 z=0
Ev Nt Nb w=0.2 h=0.2
.equiv Nb Nland
.external Nt Nb
.freq fmin=1 fmax=1 ndec=1
.end
"
        ));
        assert_eq!(plane_first.geometry, direct.geometry);
        assert_eq!(node_first.geometry, direct.geometry);
        assert_eq!(node_first.ports, direct.ports);
    }

    /// `.equiv` cannot be used to weld two planes together: the connection
    /// would silently attach to one of them only.
    #[test]
    fn equiv_across_two_planes_is_rejected() {
        let error = parse(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 Na (5, 3, 0)
Gq x1=0 y1=10 z1=0 x2=10 y2=10 z2=0 x3=10 y3=16 z3=0
+ thick=0.04 seg1=5 seg2=3 Nb (5, 13, 0)
Nt x=5 y=3 z=0.5
Ev Nt Na w=0.2 h=0.2
.equiv Na Nb
.external Nt Na
.freq fmin=1 fmax=1 ndec=1
.end
",
        )
        .unwrap_err();
        assert_eq!(error.line, 9, "{}", error.message);
        assert!(error.message.contains("joining two planes"), "{error}");
    }

    /// `nhinc` on a plane cuts every bar of *that* plane through the
    /// thickness, and nothing else.
    #[test]
    fn plane_nhinc_subdivides_only_the_plane_bars() {
        let deck = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 nhinc=3",
            "",
        ));
        // 5 × 3 cells: 2·15 − 5 − 3 = 22 bars, then the via.
        let Discretization::PerSegment(subdivisions) = &deck.discretization else {
            panic!(
                "a plane with nhinc differs from its segments: {:?}",
                deck.discretization
            );
        };
        assert_eq!(subdivisions.len(), 22 + 1);
        assert!(subdivisions[..22]
            .iter()
            .all(|sub| *sub == Subdivision::new(1, 3)));
        assert_eq!(subdivisions[22], Subdivision::SINGLE);
        // The extension form takes it too.
        let extension = parse_ok(&plane_deck(
            "Gp 0 0 0.02 10 6 0.02 0.04 nx=5 ny=3 nhinc=3",
            "",
        ));
        assert_eq!(extension.discretization, deck.discretization);
    }

    /// The two grammars mix freely inside one deck, and `.hole` /
    /// `.contact` reach a plane declared either way.
    #[test]
    fn both_g_grammars_mix_in_one_deck() {
        let deck = parse_ok(
            "\
.units mm
.default sigma=5.8e4
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
Gq 0 10 0.02 10 16 0.02 0.04 nx=5 ny=3
.hole Gp 4.9 2.9 5.1 3.1
.contact Gq 4 12 6 14 nx=2 ny=2 ratio=2
Nt x=5 y=3 z=0.5
Nb x=5 y=3 z=0
Nu x=5 y=13 z=0
Ev Nt Nb w=0.2 h=0.2
Ew Nt Nu w=0.2 h=0.2
.external Nb Nu
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
        // Gp: 15 cells less the holed one; Gq graded by its contact.
        assert!(deck.geometry.segment_count() > 22);
        assert_eq!(deck.ports.len(), 1);
    }

    /// The statement scanner takes the layouts the documented format
    /// allows: whitespace around `=`, and value lists separated by commas,
    /// whitespace, or both.
    #[test]
    fn plane_statement_layout_is_flexible() {
        let spaced = parse_ok(&plane_deck(
            "\
Gp x1 = 0 y1 = 0 z1 = 0 x2 = 10 y2 = 0 z2 = 0 x3 = 10 y3 = 6 z3 = 0
+ thick = 0.04 seg1 = 5 seg2 = 3
+ hole rect ( 0.5 4.5 0 1.5 5.5 0 )",
            "",
        ));
        let tight = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3
+ hole rect (0.5,4.5,0,1.5,5.5,0)",
            "",
        ));
        assert_eq!(spaced.geometry, tight.geometry);
    }

    /// Every documented plane parameter this engine cannot represent is
    /// rejected by name, on the statement's own line — never ignored.
    #[test]
    fn documented_plane_parameters_are_supported_or_named_in_the_error() {
        let bad = |extra: &str| -> ParseError {
            parse(&plane_deck(
                &format!(
                    "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 {extra}"
                ),
                "",
            ))
            .unwrap_err()
        };
        for (extra, expected) in [
            ("nhinc=3 rh=2", "'rh'"),
            ("segwid1=0.5", "'segwid1'"),
            ("segwid2=0.5", "'segwid2'"),
            ("relx=1", "'relx'"),
            ("rely=1", "'rely'"),
            ("relz=1", "'relz'"),
            // Only a *named* discretization hierarchy is rejected; the
            // documented `file=NONE` is accepted as the no-op it is — see
            // `file_none_is_the_documented_no_op_and_a_named_hierarchy_file_is_rejected`.
            ("file=plane.mat", "'file=plane.mat'"),
            ("nx=5", "seg1"),
            ("ny=3", "seg1"),
            ("wibble=1", "unknown ground-plane parameter 'wibble'"),
            ("hole user1 (5, 3, 0)", "'hole user1' is not supported"),
            // A diagonal trace stays rejected (issue #110): its cell size
            // is not determined by the public description.
            (
                "contact trace (1, 1, 0, 9, 5, 0, 0.2, 1)",
                "a diagonal 'contact trace' is not supported",
            ),
        ] {
            let error = bad(extra);
            assert_eq!(error.line, 3, "'{extra}' reports the statement's line");
            assert!(
                error.message.contains(expected),
                "'{extra}' must name what it rejects, got: {}",
                error.message
            );
        }
    }

    /// All seven user-defined holes are rejected on the statement's own
    /// line, each naming itself, saying the rejection is permanent, and
    /// naming both alternatives: the declarative shapes, and the library
    /// path (`GroundPlane::mesh` + `Hole::Point`) for a shape none of them
    /// describe. Issue #99.
    #[test]
    fn user_defined_holes_are_rejected_permanently_and_name_the_alternative() {
        for n in 1..=7 {
            let error = parse(&plane_deck(
                &format!(
                    "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 hole user{n} (5, 3, 0, 1, 2)"
                ),
                "",
            ))
            .unwrap_err();
            assert_eq!(error.line, 3, "'hole user{n}' reports the statement's line");
            for expected in [
                &format!("'hole user{n}' is not supported, and will not be"),
                "hole rect (x1, y1, z1, x2, y2, z2)",
                "hole point (x, y, z)",
                "hole circle (x, y, z, r)",
                "fasterhenry::plane::GroundPlane::mesh",
                "fasterhenry::plane::Hole::Point",
            ] {
                assert!(
                    error.message.contains(expected),
                    "'hole user{n}' must name {expected}, got: {}",
                    error.message
                );
            }
        }

        // The spelling is exact: `user8` and `usery` are not user-defined
        // hole names, so they fall to the generic unknown-shape arm, which
        // makes no claim about permanence.
        for other in ["user8", "user0", "usery", "user"] {
            let error = parse(&plane_deck(
                &format!(
                    "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 hole {other} (5, 3, 0)"
                ),
                "",
            ))
            .unwrap_err();
            assert_eq!(error.line, 3);
            assert!(
                error
                    .message
                    .contains(&format!("'hole {other}' is not supported"))
                    && !error.message.contains("will not be"),
                "'hole {other}' takes the generic rejection, got: {}",
                error.message
            );
        }
    }

    /// The corner points must describe an axis-aligned rectangle parallel
    /// to xy; anything else is named, not squared off.
    #[test]
    fn plane_corner_points_are_validated() {
        let bad = |corners: &str, tail: &str| -> ParseError {
            parse(&plane_deck(&format!("Gp {corners}\n+ {tail}"), "")).unwrap_err()
        };
        let square = "thick=0.04 seg1=5 seg2=3";
        // Tilted out of the xy plane.
        let error = bad("x1=0 y1=0 z1=0 x2=10 y2=0 z2=1 x3=10 y3=6 z3=1", square);
        assert_eq!(error.line, 3);
        assert!(
            error.message.contains("not parallel to the xy plane"),
            "{error}"
        );
        // Rotated in the xy plane.
        let error = bad("x1=0 y1=0 z1=0 x2=10 y2=1 z2=0 x3=9 y3=7 z3=0", square);
        assert!(error.message.contains("axis-aligned rectangle"), "{error}");
        // Two edges along the same axis: not a rectangle.
        let error = bad("x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=4 y3=0 z3=0", square);
        assert!(error.message.contains("axis-aligned rectangle"), "{error}");
        // Missing pieces are named individually.
        for (corners, tail, expected) in [
            ("x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6", square, "'z3'"),
            (
                "x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0",
                "seg1=5 seg2=3",
                "'thick'",
            ),
            (
                "x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0",
                "thick=0.04 seg2=3",
                "'seg1'",
            ),
            (
                "x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0",
                "thick=0.04 seg1=5",
                "'seg2'",
            ),
        ] {
            let error = bad(corners, tail);
            assert_eq!(error.line, 3);
            assert!(error.message.contains(expected), "{error}");
        }
    }

    /// A hole, contact or in-plane node the deck put somewhere other than
    /// this plane is a mistake, not a silently relocated feature.
    #[test]
    fn plane_features_must_lie_in_the_plane() {
        let bad = |extra: &str| -> ParseError {
            parse(&plane_deck(
                &format!(
                    "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 {extra}"
                ),
                "",
            ))
            .unwrap_err()
        };
        // A z far from this plane's slab.
        let error = bad("hole rect (0.5, 4.5, 3, 1.5, 5.5, 3)");
        assert_eq!(error.line, 3);
        assert!(
            error.message.contains("is not in ground plane 'Gp'"),
            "{error}"
        );
        // A hole with no area removes nothing, so it is rejected.
        let error = bad("hole rect (0.5, 4.5, 0, 0.5, 5.5, 0)");
        assert!(error.message.contains("degenerate"), "{error}");
        // An in-plane node outside the footprint.
        let error = bad("Nfar (50, 3, 0)");
        assert!(
            error.message.contains("outside ground plane 'Gp'"),
            "{error}"
        );
        // Wrong arity in a value list.
        let error = bad("Nland (5, 3)");
        assert!(error.message.contains("takes 3 values"), "{error}");
        let error = bad("hole rect (5, 3, 0, 6)");
        assert!(error.message.contains("takes 6 values"), "{error}");
        // Junk instead of a parameter.
        let error = bad("wibble");
        assert!(
            error.message.contains("is not a ground-plane parameter"),
            "{error}"
        );
        let error = bad("hole");
        assert!(error.message.contains("needs a shape name"), "{error}");
        let error = bad("hole rect (5, 3, 0");
        assert!(error.message.contains("unterminated"), "{error}");
    }

    #[test]
    fn deck_with_no_conductor_is_an_error_not_a_panic() {
        // Nodes and a port, but no `E` segment and no `G` ground plane: there
        // is nothing to discretize, and it must be a `ParseError`, not the
        // `subdivisions[0]` index-out-of-bounds panic this regresses.
        let error = parse(
            "\
.units mm
N1 x=0 y=0 z=0
N2 x=1 y=0 z=0
.external N1 N2
.freq fmin=1 fmax=1 ndec=1
.end
",
        )
        .unwrap_err();
        assert!(error.message.contains("deck has no conductor"), "{error}");
        assert!(error.message.contains('E'), "{error}");
        assert!(error.message.contains('G'), "{error}");
        assert_eq!(error.line, 0);

        // Edge case: an `E` segment with no plane is unaffected.
        parse_ok(
            "\
.units mm
N1 x=0 y=0 z=0
N2 x=1 y=0 z=0
E1 N1 N2 w=1 h=1 sigma=5.8e4
.external N1 N2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );

        // Edge case: a `G` plane with no `E` segment is unaffected.
        parse_ok(
            "\
.units mm
Gp 0 0 0 10 6 0 0.035 nx=5 ny=3 sigma=5.8e4
N1 x=0 y=0 z=0
N2 x=1 y=0 z=0
.external N1 N2
.freq fmin=1 fmax=1 ndec=1
.end
",
        );
    }

    #[test]
    fn missing_units_missing_end_and_unknown_directive() {
        let error = parse("n1 x=0 y=0 z=0\n.end\n").unwrap_err();
        assert!(error.message.contains(".units"));
        let valid_prefix = "\
.units m
n1 x=0 y=0 z=0
n2 x=1 y=0 z=0
e1 n1 n2 w=1 h=1 sigma=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
";
        let error = parse(valid_prefix).unwrap_err();
        assert!(error.message.contains(".end"));
        let error = parse(".units m\n.cparams tol=1e-3\n.end\n").unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.message.contains("deck subset"));
    }

    /// A one-segment deck under `unit` whose conductivity comes from
    /// `default_fields` (spliced into `.default`) and `segment_fields`
    /// (spliced onto the `E` line).
    fn conductivity_deck(unit: &str, default_fields: &str, segment_fields: &str) -> String {
        format!(
            "\
.units {unit}
.default z=0 w=1 h=1 {default_fields}
n1 x=0 y=0
n2 x=10 y=0
e1 n1 n2 {segment_fields}
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
"
        )
    }

    #[test]
    fn rho_on_a_segment_is_one_over_sigma_per_deck_unit() {
        for (unit, factor) in [("m", 1.0), ("mm", 1e-3), ("um", 1e-6)] {
            let from_rho = parse_ok(&conductivity_deck(unit, "", "rho=0.5"));
            let from_sigma = parse_ok(&conductivity_deck(unit, "", "sigma=2"));
            let sigma = from_rho.geometry.segment(0).unwrap().sigma;
            // rho=0.5 ohm·unit is sigma=2 S/unit = 2/factor S/m, exactly.
            assert_eq!(sigma, 2.0 / factor, "{unit}");
            assert_eq!(from_rho, from_sigma, "{unit}");
        }
        // Copper by resistivity in mm: 1.7241e-5 ohm·mm = 1.7241e-8 ohm·m.
        let copper = parse_ok(&conductivity_deck("mm", "", "rho=1.7241e-5"));
        let sigma = copper.geometry.segment(0).unwrap().sigma;
        assert!((sigma - 1.0 / 1.7241e-8).abs() < 1e-12 * sigma);
    }

    #[test]
    fn rho_on_default_and_overrides() {
        let from_default = parse_ok(&conductivity_deck("mm", "rho=0.25", ""));
        assert_eq!(from_default.geometry.segment(0).unwrap().sigma, 4.0 / 1e-3);
        // A per-line rho overrides a default sigma, and vice versa.
        let overridden = parse_ok(&conductivity_deck("mm", "sigma=1", "rho=0.5"));
        assert_eq!(overridden.geometry.segment(0).unwrap().sigma, 2.0 / 1e-3);
        let overridden = parse_ok(&conductivity_deck("mm", "rho=0.5", "sigma=8"));
        assert_eq!(overridden.geometry.segment(0).unwrap().sigma, 8.0 / 1e-3);
        // Separate .default lines: the later one wins.
        let later = parse_ok(
            &conductivity_deck("mm", "sigma=1", "")
                .replace(".default z=0", ".default rho=0.5\n.default z=0"),
        );
        assert_eq!(later.geometry.segment(0).unwrap().sigma, 1.0 / 1e-3);
    }

    #[test]
    fn rho_and_sigma_on_one_line_is_an_error() {
        for (default_fields, segment_fields, line) in [
            ("", "sigma=2 rho=0.5", 5),
            ("", "rho=0.5 sigma=2", 5),
            ("sigma=2 rho=0.5", "", 2),
        ] {
            let error =
                parse(&conductivity_deck("mm", default_fields, segment_fields)).unwrap_err();
            assert_eq!(error.line, line, "{error}");
            assert!(error.message.contains("both"), "{error}");
        }
        // Across a continuation line it is still one line.
        let error =
            parse(&conductivity_deck("mm", "", "sigma=2").replace("sigma=2", "sigma=2\n+ rho=0.5"))
                .unwrap_err();
        assert_eq!(error.line, 5);
        // On a G line too.
        let error = parse(".units mm\nGp 0 0 0 10 6 0 0.035 sigma=1 rho=1\n.end\n").unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.message.contains("both"), "{error}");
    }

    #[test]
    fn rho_must_be_positive() {
        for bad in ["rho=0", "rho=-1", "rho=0.0"] {
            let error = parse(&conductivity_deck("mm", "", bad)).unwrap_err();
            assert_eq!(error.line, 5, "{bad}");
            assert!(error.message.contains("rho"), "{error}");
        }
        let error = parse(&conductivity_deck("mm", "rho=0", "")).unwrap_err();
        assert_eq!(error.line, 2);
    }

    #[test]
    fn g_plane_takes_sigma_or_rho_on_its_line() {
        let plane_sigma = |text: &str| -> f64 {
            // The first plane bar carries the plane's conductivity.
            parse_ok(text).geometry.segment(0).unwrap().sigma
        };
        let deck = |g_fields: &str, default_fields: &str| {
            format!(
                "\
.units mm
.default {default_fields}
Gp 0 0 0 10 6 0 0.035 nx=5 ny=3 {g_fields}
n1 x=1 y=0 z=0
n2 x=9 y=0 z=0
e1 n1 n2 w=0.2 h=0.035 sigma=5.8e4
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
"
            )
        };
        let by_default = plane_sigma(&deck("", "sigma=2"));
        assert_eq!(by_default, 2.0 / 1e-3);
        assert_eq!(plane_sigma(&deck("", "rho=0.5")), by_default);
        assert_eq!(plane_sigma(&deck("rho=0.5", "z=0")), by_default);
        assert_eq!(plane_sigma(&deck("sigma=2", "z=0")), by_default);
        // A per-line value overrides the default.
        assert_eq!(plane_sigma(&deck("rho=0.5", "sigma=100")), by_default);
        // The whole deck is the same either way.
        assert_eq!(
            parse_ok(&deck("rho=0.5", "z=0")),
            parse_ok(&deck("", "sigma=2"))
        );
    }

    /// Conductivity composes with the other per-plane fields: on either
    /// grammar `sigma=`/`rho=` sit alongside the cell counts and `nhinc=` in
    /// any order, and both grammars build the same plane given equivalent
    /// conductivities — and a `.default rho=` reaches a corner-point plane
    /// that names neither.
    #[test]
    fn plane_conductivity_composes_with_nhinc_and_both_grammars() {
        let corner = parse_ok(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 sigma=2 nhinc=3",
            "",
        ));
        for fields in [
            "nx=5 ny=3 nhinc=3 rho=0.5",
            "rho=0.5 nx=5 ny=3 nhinc=3",
            "nx=5 nhinc=3 sigma=2 ny=3",
            "nx=5 ny=3\n+ nhinc=3\n+ rho=0.5",
        ] {
            let extension = parse_ok(&plane_deck(
                &format!("Gp 0 0 0.02 10 6 0.02 0.04 {fields}"),
                "",
            ));
            assert_eq!(extension, corner, "{fields}");
            // The plane bars carry the per-plane value, not the default...
            assert_eq!(extension.geometry.segment(0).unwrap().sigma, 2.0 / 1e-3);
            // ...and nhinc still subdivides them.
            let Discretization::PerSegment(subdivisions) = &extension.discretization else {
                panic!("{fields}: {:?}", extension.discretization);
            };
            assert_eq!(subdivisions[0], Subdivision::new(1, 3), "{fields}");
        }
        // sigma and rho together is still an error next to nhinc, even
        // split across continuation lines.
        let error = parse(&plane_deck(
            "Gp 0 0 0.02 10 6 0.02 0.04 nx=5 ny=3 sigma=2\n+ nhinc=3 rho=0.5",
            "",
        ))
        .unwrap_err();
        assert_eq!(error.line, 3, "{error}");
        assert!(error.message.contains("both"), "{error}");

        // The corner-point form takes rho= on the statement itself, in any
        // order and across continuation lines, and builds the same deck as
        // the equivalent sigma=.
        for fields in [
            "thick=0.04 seg1=5 seg2=3 rho=0.5 nhinc=3",
            "rho=0.5 thick=0.04 seg1=5 seg2=3 nhinc=3",
            "thick=0.04 seg1=5 seg2=3 nhinc=3\n+ rho=0.5",
        ] {
            let by_rho = parse_ok(&plane_deck(
                &format!("Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0\n+ {fields}"),
                "",
            ));
            assert_eq!(by_rho, corner, "{fields}");
            assert_eq!(
                by_rho.geometry.segment(0).unwrap().sigma,
                2.0 / 1e-3,
                "{fields}"
            );
        }
        // sigma= and rho= together on one corner-point statement is the same
        // line-numbered error as everywhere else, continuations included.
        for fields in [
            "thick=0.04 seg1=5 seg2=3 sigma=2 rho=0.5",
            "thick=0.04 seg1=5 seg2=3 rho=0.5 sigma=2",
            "thick=0.04 seg1=5 seg2=3 sigma=2\n+ nhinc=3 rho=0.5",
        ] {
            let error = parse(&plane_deck(
                &format!("Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0\n+ {fields}"),
                "",
            ))
            .unwrap_err();
            assert_eq!(error.line, 3, "{error}");
            assert!(error.message.contains("both"), "{error}");
        }
        // A non-positive rho is rejected here as it is elsewhere.
        let error = parse(&plane_deck(
            "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 rho=0",
            "",
        ))
        .unwrap_err();
        assert_eq!(error.line, 3, "{error}");
        assert!(error.message.contains("rho"), "{error}");
        // A `.default rho=` is the deck's conductivity, and reaches a
        // corner-point plane that names neither.
        let corner_plane = "\
Gp x1=0 y1=0 z1=0 x2=10 y2=0 z2=0 x3=10 y3=6 z3=0
+ thick=0.04 seg1=5 seg2=3 nhinc=3";
        let by_rho = parse_ok(
            &plane_deck(corner_plane, "").replace(".default sigma=5.8e4", ".default rho=0.5"),
        );
        let by_sigma = parse_ok(
            &plane_deck(corner_plane, "").replace(".default sigma=5.8e4", ".default sigma=2"),
        );
        assert_eq!(by_rho, by_sigma);
        assert_eq!(by_rho.geometry.segment(0).unwrap().sigma, 2.0 / 1e-3);
    }

    #[test]
    fn every_documented_unit_scales_lengths_and_conductivity() {
        for (unit, factor) in [
            ("km", 1e3),
            ("m", 1.0),
            ("cm", 1e-2),
            ("mm", 1e-3),
            ("um", 1e-6),
            ("in", 0.0254),
            ("mils", 25.4e-6),
            ("mil", 25.4e-6),
            // Directives are case-insensitive, and so is the unit.
            ("MM", 1e-3),
            ("In", 0.0254),
            ("MILS", 25.4e-6),
        ] {
            let deck = parse_ok(&conductivity_deck(unit, "", "sigma=3"));
            let segment = deck.geometry.segment(0).unwrap();
            assert_eq!(segment.length(), 10.0 * factor, "{unit}");
            assert_eq!(segment.width, factor, "{unit}");
            assert_eq!(segment.sigma, 3.0 / factor, "{unit}");
        }
        let error = parse(&conductivity_deck("furlong", "", "sigma=1")).unwrap_err();
        assert_eq!(error.line, 1);
        for unit in ["km", "in", "mils"] {
            assert!(error.message.contains(unit), "{error}");
        }
    }

    #[test]
    fn line_numbers_count_comments_and_blanks() {
        let error = parse(
            "\
* leading comment

.units m
n1 x=0 y=0 z=0
n2 x=1 y=0 z=0
e1 n1 n9 w=1 h=1 sigma=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        )
        .unwrap_err();
        assert_eq!(error.line, 6);
        assert!(error.message.contains("n9"));
    }

    #[test]
    fn missing_field_names_the_segment_and_the_field() {
        let error = parse(
            "\
.units m
n1 x=0 y=0 z=0
n2 x=1 y=0 z=0
e1 n1 n2 h=1 sigma=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        )
        .unwrap_err();
        assert_eq!(error.line, 4);
        assert!(error.message.contains("'e1'"));
        assert!(error.message.contains("width"));
    }

    #[test]
    fn unset_node_coordinate_points_at_the_default() {
        let error = parse(
            "\
.units m
n1 x=0
n2 x=1 y=0 z=0
e1 n1 n2 w=1 h=1 sigma=1
.external n1 n2
.freq fmin=1 fmax=1 ndec=1
.end
",
        )
        .unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.message.contains(".default"));
    }

    #[test]
    fn freq_validation() {
        let template = "\
.units m
n1 x=0 y=0 z=0
n2 x=1 y=0 z=0
e1 n1 n2 w=1 h=1 sigma=1
.external n1 n2
.freq {SWEEP}
.end
";
        for (sweep, what) in [
            ("fmin=1e6 fmax=1e3 ndec=10", "fmax < fmin"),
            ("fmin=0 fmax=1e6 ndec=10", "log sweep from zero"),
            ("fmin=-1 fmax=1e6 ndec=10", "negative"),
            ("fmin=1 fmax=1e6", "missing ndec"),
        ] {
            let error = parse(&template.replace("{SWEEP}", sweep)).unwrap_err();
            assert_eq!(error.line, 6, "{what}: {}", error.message);
        }
    }

    #[test]
    fn external_needs_known_nodes() {
        let error = parse(
            "\
.units m
n1 x=0 y=0 z=0
n2 x=1 y=0 z=0
e1 n1 n2 w=1 h=1 sigma=1
.external n1 n7
.freq fmin=1 fmax=1 ndec=1
.end
",
        )
        .unwrap_err();
        assert_eq!(error.line, 5);
        assert!(error.message.contains("n7"));
    }

    #[test]
    fn case_insensitive_directives_and_field_keys() {
        let deck = parse_ok(
            "\
.UNITS m
n1 X=0 Y=0 Z=0
n2 x=1 y=0 z=0
e1 n1 n2 W=1 H=1 SIGMA=1
.External n1 n2
.FREQ FMIN=1 FMAX=1 NDEC=1
.END
",
        );
        assert_eq!(deck.geometry.segment_count(), 1);
        assert_eq!(deck.ports.len(), 1);
    }

    /// A go-and-return loop of two long bars (y = 0 and y = 3 mm) joined by a
    /// square rung, driven at its open end. `{BARS}` is spliced into both
    /// long-bar `E` lines, so it sets their cross-section and orientation;
    /// their mutual coupling makes the impedance depend on which way the
    /// cross-section faces.
    fn loop_deck(defaults: &str, bars: &str) -> String {
        format!(
            "\
.units mm
.default z=0 sigma=5.8e4 {defaults}
n1 x=0 y=0
n2 x=10 y=0
n3 x=10 y=3
n4 x=0 y=3
e1 n1 n2 {bars}
e2 n2 n3 w=0.5 h=0.5 wx=1 wy=0 wz=0 nwinc=1 nhinc=1 rw=1 rh=1
e3 n3 n4 {bars}
.external n1 n4
.freq fmin=1e8 fmax=1e8 ndec=1
.end
"
        )
    }

    /// The loop's `Z` as `(R, X)`.
    fn loop_impedance(deck: &Deck) -> (f64, f64) {
        let result = fasterhenry::solve::solve(
            &deck.geometry,
            &deck.ports,
            &deck.discretization,
            &deck.frequencies,
        )
        .unwrap();
        let z = result.impedance_ohm[0][(0, 0)];
        (z.re, z.im)
    }

    fn z_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - b.0).hypot(a.1 - b.1) / b.0.hypot(b.1)
    }

    fn assert_same_z(a: (f64, f64), b: (f64, f64)) {
        assert!(z_distance(a, b) <= 1e-9, "{a:?} vs {b:?}");
    }

    /// `wx/wy/wz` rotates the cross-section: a flat 1 mm × 0.1 mm bar turned
    /// on edge by pointing its width along z is the same conductor as a
    /// 0.1 mm × 1 mm bar in the default orientation (width in the x–y
    /// plane), so the two decks give the same `Z` — and a different one
    /// from the flat bar. The rung's explicit `wx=1` (perpendicular to it,
    /// along y) is the default orientation spelled out.
    #[test]
    fn width_direction_rotates_the_cross_section() {
        let on_edge = parse_ok(&loop_deck("", "w=1 h=0.1 nwinc=3 nhinc=2 wx=0 wy=0 wz=1"));
        let tall = parse_ok(&loop_deck("", "w=0.1 h=1 nwinc=2 nhinc=3"));
        let flat = parse_ok(&loop_deck("", "w=1 h=0.1 nwinc=3 nhinc=2"));

        let segment = on_edge.geometry.segment(0).unwrap();
        assert_eq!(segment.width_dir, Some([0.0, 0.0, 1.0]));
        let basis = segment.basis().unwrap();
        assert!(basis.width.x.abs() + basis.width.y.abs() + (basis.width.z - 1.0).abs() < 1e-15);
        assert_eq!(tall.geometry.segment(0).unwrap().width_dir, None);

        let z_on_edge = loop_impedance(&on_edge);
        assert_same_z(z_on_edge, loop_impedance(&tall));
        let z_flat = loop_impedance(&flat);
        assert!(
            z_distance(z_on_edge, z_flat) > 1e-4,
            "orientation must matter: {z_on_edge:?} vs {z_flat:?}"
        );

        // The same through `.default`, with only one component given (the
        // others are 0) — and a line value overriding its default.
        let via_default = parse_ok(&loop_deck("wz=1", "w=1 h=0.1 nwinc=3 nhinc=2"));
        assert_eq!(
            via_default.geometry.segment(0).unwrap().width_dir,
            Some([0.0, 0.0, 1.0])
        );
        assert_same_z(loop_impedance(&via_default), z_on_edge);
        let overridden = parse_ok(&loop_deck("wz=1", "w=1 h=0.1 nwinc=3 nhinc=2 wy=1 wz=0"));
        assert_eq!(
            overridden.geometry.segment(0).unwrap().width_dir,
            Some([0.0, 1.0, 0.0])
        );
        assert_same_z(loop_impedance(&overridden), z_flat);
    }

    /// `rw`/`rh` produce the graded grid of the library's graded
    /// discretization, per segment and per axis, on `E` lines and through
    /// `.default` alike.
    #[test]
    fn filament_ratios_grade_the_grid_like_the_library() {
        let bars = "w=1 h=0.1 nwinc=5 nhinc=4 rw=2 rh=3";
        let deck = parse_ok(&loop_deck("", bars));
        let rung = AxisGrading::new(1, 1, 1.0, 1.0);
        let bar = AxisGrading::new(5, 4, 2.0, 3.0);
        assert_eq!(
            deck.discretization,
            Discretization::PerSegmentGraded(vec![bar, rung, bar])
        );
        // The bar's filaments are exactly the library's graded grid called
        // directly: widths 1:2:4:2:1 across the width, 1:3:3:1 across the
        // height.
        let segment = deck.geometry.segment(0).unwrap();
        let filaments = fasterhenry::discretize_graded_per_axis(&segment, 5, 4, 2.0, 3.0).unwrap();
        let widths = fasterhenry::filament::graded_extents(1e-3, 5, 2.0).unwrap();
        let heights = fasterhenry::filament::graded_extents(0.1e-3, 4, 3.0).unwrap();
        assert!((widths[0] - 1e-3 / 10.0).abs() < 1e-18);
        assert!((heights[0] - 0.1e-3 / 8.0).abs() < 1e-18);
        for (index, filament) in filaments.iter().enumerate() {
            assert!((filament.width() - widths[index % 5]).abs() < 1e-18);
            assert!((filament.height() - heights[index / 5]).abs() < 1e-18);
        }

        // Equal ratios solve exactly as the library's shared-ratio
        // `Discretization::graded`, called directly on the same geometry.
        let shared = parse_ok(
            "\
.units mm
n1 x=0 y=0 z=0
n2 x=10 y=0 z=0
e1 n1 n2 w=1 h=0.1 sigma=5.8e4 nwinc=3 nhinc=3 rw=2 rh=2
.external n1 n2
.freq fmin=1e8 fmax=1e8 ndec=1
.end
",
        );
        let direct = fasterhenry::solve::solve(
            &shared.geometry,
            &shared.ports,
            &Discretization::graded(3, 3, 2.0),
            &shared.frequencies,
        )
        .unwrap()
        .impedance_ohm[0][(0, 0)];
        assert_eq!(loop_impedance(&shared), (direct.re, direct.im));
        let uniform = fasterhenry::solve::solve(
            &shared.geometry,
            &shared.ports,
            &Discretization::uniform(3, 3),
            &shared.frequencies,
        )
        .unwrap()
        .impedance_ohm[0][(0, 0)];
        assert_ne!(loop_impedance(&shared), (uniform.re, uniform.im));

        // Through `.default`, with a line value overriding one ratio.
        let defaulted = parse_ok(&loop_deck("rw=2 rh=3", "w=1 h=0.1 nwinc=5 nhinc=4"));
        assert_eq!(defaulted.discretization, deck.discretization);
        let overridden = parse_ok(&loop_deck("rw=2 rh=3", "w=1 h=0.1 nwinc=5 nhinc=4 rh=1"));
        assert_eq!(
            overridden.discretization,
            Discretization::PerSegmentGraded(vec![
                AxisGrading::new(5, 4, 2.0, 1.0),
                rung,
                AxisGrading::new(5, 4, 2.0, 1.0),
            ])
        );

        // A ratio on single-filament axes changes nothing, so the deck stays
        // on the uniform discretization it always had.
        let inert = parse_ok(&loop_deck("rw=2 rh=2", "w=1 h=0.1"));
        assert_eq!(
            inert.discretization,
            Discretization::Uniform(Subdivision::SINGLE)
        );
    }

    #[test]
    fn invalid_ratios_and_width_directions_carry_their_line() {
        // (deck defaults, bar fields, expected line, message fragment)
        for (defaults, bars, line, fragment) in [
            ("rw=0", "w=1 h=0.1", 2, "'rw' must be positive"),
            ("", "w=1 h=0.1 rh=-2", 7, "'rh' must be positive"),
            ("", "w=1 h=0.1 rw=0.5", 7, "'rw' must be ≥ 1"),
            ("", "w=1 h=0.1 rw=abc", 7, "not a number"),
            ("", "w=1 h=0.1 wx=0 wy=0 wz=0", 7, "zero vector"),
            ("wx=0", "w=1 h=0.1", 7, "zero vector"),
            // Both long bars run along x, so an x width direction is parallel.
            ("", "w=1 h=0.1 wx=1", 7, "parallel to the segment"),
            ("wx=-2.5", "w=1 h=0.1", 7, "parallel to the segment"),
        ] {
            let error = parse(&loop_deck(defaults, bars)).unwrap_err();
            assert_eq!(error.line, line, "{defaults} / {bars}: {}", error.message);
            assert!(
                error.message.contains(fragment),
                "{defaults} / {bars}: {}",
                error.message
            );
        }
    }

    #[test]
    fn decade_frequencies_hit_round_values_exactly() {
        let frequencies = frequency_sweep(1e6, 1e9, 1, 6).unwrap();
        assert_eq!(frequencies, vec![1e6, 1e7, 1e8, 1e9]);
        let frequencies = frequency_sweep(1e3, 1e6, 10, 6).unwrap();
        assert_eq!(frequencies.len(), 31);
        assert_eq!(frequencies[30], 1e6);
    }
}
