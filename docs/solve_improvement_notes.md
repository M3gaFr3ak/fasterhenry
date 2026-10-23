# Solve improvement notes (precond / GMRES / subdivision)

Backlog captured from the 2026-10-22/23 exercise: extracting real board-level
loop inductance with the iterative path (`--solver iterative`, cubel-global
precond), on a stripped inverter board. Written so a future fork session can
pick any item up without re-deriving the context. Numbers below are measured,
from `~/peec_scratch` logs (the DC-bus runs) and the local decks.

## Context: what the exercise hit

The target is the **DC-bus commutation loop** of a 6-layer inverter
(VBUS plane -> power module -> GNDPWR plane), and, as the stress case, the
**OUT_U switch node** (a via-studded, shunt-funnelled copper mass).

Pipeline: board stackup -> `geometry.json` (105 um copper, sigma 5.97e7 S/m)
-> trim -> sizing field -> quadtree -> bar map -> reduce -> a FastHenry `.inp`
deck (no filament subdivision on the deck path; `nwinc`/`nhinc` ride the deck)
-> the fork's iterative solver with `FASTERHENRY_PRECOND=cubel-global`.

The precond is a **radius-limited near-pattern** of the loop-space matrix
`M Z M^T` (M the loop-branch incidence, `Z = R + j w L`), factored **exactly**
by faer's sparse `Lu` (`cubel.rs::CubelGlobal::build` -> `factor`). Measured
behaviour:

| case | rows | branches | precond CSC nnz | factor hwm | result |
|---|---|---|---|---|---|
| DC-bus, nwinc/nhinc = 1 | 14,422 | 70,672 | 9.78 M | 5.3 GB | converged, **90 iters** at `CUBEL_RADIUS_MULT=4` (1576 at mult 1) |
| OUT_U, 1x1 | 232,357 | 931,088 | 748 M | 43.7 GB | symbolic LU FAILED, RSS governor tripped at 57 GB (exit 134) |
| DC-bus, nwinc/nhinc = 2 | 226,438 | 282,688 | - | - | CSC assembly OOMed > 58 GB |

The two costs that matter:
1. **The factor's fill** - faer's `Lu` grows its factor in RAM; there is no
   out-of-core LU. The fill is the peak, not the input pattern.
2. **The retained CSC** - `CubelGlobal` keeps `row_idx`/`re_part`/`im_part`
   for the whole solve, though their only consumer is building the
   per-frequency `values`; after the factor they are dead weight.

And the multiplier: `nwinc`/`nhinc` subdivision multiplies the branch count
(nwinc=nhinc=2 -> 4x branches, 15.7x loops on this board), so the loop-space
matrix - and its fill - blow up fast. That is the wall.

## Items

### A. GMRES warm start / subspace reuse between frequencies and ports

**What.** Feed a previous solution (or Krylov subspace) as the initial guess
instead of zero.

**Context.** The ports loop (`schur_complement_gmres`, `iterative.rs`) is
sequential and passes `x = vec![0; internal]` per port; the GMRES *does* use an
incoming `x` (`residual_of(apply, b, x)`), so it is warm-start-capable and just
unfed. The precond/operator are already shared across ports (built once per
omega). The frequency sweep is `par_iter` (parallel within a memory budget
chunk), so a sequential omega_k -> omega_k+1 continuation conflicts with that
parallelism.

**Verdict.** Optional. Ports: trivial and no-risk (a closer `x0` can only
reduce the initial residual; a distant one is neutral). Frequencies: only if
the sweep is serialized first. Suggested refinement: the *first* port/frequency
that converges seeds the others *only when a solve is stalling*.

**Effort.** Small (ports). The stronger form - Krylov recycling (GMRES-DR /
GCRO-DR) reuses the subspace, not just `x0` - is new solver code.

### B. Drop the precond after a solve

**What.** Free the CSC (and the factor) once no longer needed.

**Context.** The factor is per-omega and reused across *ports*; the CSC is
omega-independent and reused across the *sweep* (it exists precisely so a
frequency change does not re-assemble `M Z M^T`). The `values` array is already
a transient (freed after the factor); the *source* CSC is not.

**Verdict.** The retention-drop is real but **does not help an OOM** - the peak
is *during the factor*, so freeing the CSC afterwards only lowers steady-state
RSS, not the max. Defer as a cleanliness/footprint item (and it only applies
when no further frequency needs the pattern).

**Effort.** Small.

### C. Rayon the `cubel` (block-local) pattern build

**What.** Parallelize `CubelPattern::build`'s exact-mutual loop.

**Context.** `cubel.rs` builds the partner list with a plain serial `for a in
0..branches` (the cubel-*global* equivalent uses `into_par_iter`), so it is
both serial and un-metered - on the 282,688-branch nw2 mesh it looks hung.

**Verdict.** Defer: rayon-ing makes a **bad precond faster**. Measured
block-local behaviour was weak - `184 cubes, worst dim 7637, pattern dim-sum
752,573,768`, 364 s build, `block fallback engaged (identity rows on failed
cubes)`, then ~2700 iters to ~1e-3 before cancel. So parallelize it only if
the block-local path is revived; today cubel-global is the precond that works.

**Effort.** Small if wanted (copy the cubel-global progress pattern).

### D. Subdivision precond: coarse/fine factorization on the incidence

**What.** Precondition the *subdivided* system with a precond built on the
*unsubdivided* (coarse) mesh: restrict -> coarse solve -> prolongate, with a
coarse `CUBEL_RADIUS_MULT` solve that is known to work (90 iters, 8 GB).

**Why this shape.** The loop counts make the structure explicit: for
nwinc/nhinc = 2, `loops_fine - loops_coarse = filaments_fine - filaments_coarse
= 212,016` - the extra DOFs are exactly the intra-segment redistribution loops.
Coarse and fine describe the same copper, differing only in that local
distribution, so the coarse precond captures the *global* loop coupling (where
the hard modes live) and the local DOFs converge fast.

#### Survey: what the fork already has

- **The incidence `M`** - `MeshMatrix::row(i) -> &[(usize, f64)]` is public
  (`mesh.rs:356`), with `internal_count()` / `branch_count()` / `loop_count()`
  / `port_count()`. The loop<->branch map is directly usable.
- **The per-segment grouping** - computed in `IterativeSystem::assemble` as
  `let (filaments, per_segment) = discretize_geometry(...)` (`iterative.rs:169`),
  right beside the mesh build. A local today, but also derivable from the
  filaments (strands sharing `endpoints` - the `fh_ir` "siblings" grouping).
- **The coarse mesh** - `MeshMatrix::from_branches(branches, node_count, ports)`
  is public (`mesh.rs:215`); the coarse branches are the segments' `(a, b)`.
- **The exact near-pair mutuals** - `near_pair_data(mesh, filaments, internal,
  near_radius)`, the builder the cubel-global already uses.
- **The precond hook** - `Precond::run(&self, v: &[T]) -> Vec<T>`
  (`iterative.rs:954`); a new kind is one enum arm + one builder arm.

#### Survey: what needs adding

1. **Store (or re-derive) `per_segment`** on `IterativeSystem` - trivial.
2. **The intra/forest loop split.** The rows are ordered *intra-segment first,
   then forest* (`mesh.rs:274-317`); the split point is `sum(per_segment_i - 1)`
   - derive it from `per_segment`, or expose `link_count`.
3. **The coarse operator** - the bulk. Two routes:
   - *Galerkin*: `Z_c = S Z_f S^T` (aggregate the fine near-pair mutuals with
     `S` = the per-segment all-ones), then `A_c = M_c Z_c M_c^T`. No second
     mesh; reuses `near_pair_data`.
   - *Re-mesh*: build one coarse `Filament` per segment and run the existing
     `CubelGlobal::build` on it - less new math, needs a coarse-filament ctor.
4. **`R`/`P`** = the incidence-composed restriction `R = M_c^T S M_f` (and
   `P = R^T`). Derivable from `mesh.row` + `S`, but the cycle-space bookkeeping
   (the loop bases are not nested) is the delicate part.
5. **The 2-level apply arm** - `z = P (A_c^-1) (R v)` (+ optionally a fine
   smoother = the existing block-local `cubel`, the natural local half).

#### Memory in the apply (the theory)

Persistent, per frequency:

- **The coarse precond** (the CSC + the LU fill), built on the **coarse/forest
  loops**. The forest-loop count is **independent of the subdivision** (nwinc
  adds intra-segment loops, not forest loops), so the coarse precond is exactly
  the *un-subdivided* cubel-global's footprint (DC-bus: CSC 9.8 M nnz ~ 0.2 GB
  + fill ~ 1.5 GB) - which fits. **This is the whole win**: the subdivided
  solve's precond footprint stops tracking the mesh size.
- **The adapters `R`/`P`** - sparse (fine-loops x coarse-loops); each fine loop
  touches only a few coarse loops, so `nnz(R) ~ fine_loops x ~2-5` ~ 1 M ~
  ~10-20 MB. Small. (Sparse, so the apply is a sparse matvec `O(nnz)`, not a
  dense fine x coarse product.)
- **The vectors** - the fine residual (226k) + the coarse (14k), a few MB.

Per apply: `z = P (A_c^-1) (R v)` = two sparse matvecs + one small triangular
solve. The coarse factors are per-frequency (reused across applies); the
adapters are frequency-independent (built once).

So the precond no longer scales with the subdivision - only the adapters (small)
and the fine vectors (which the operator already holds) do.

#### Open design questions

- **`R`/`P` exact form** - the incidence route (no mesh-generator change, but
  the loop-basis care) vs enforcing a nested loop basis (a mesh-code change).
- **Fine smoother or not** - pure coarse correction (`P A_c^-1 R`) vs coarse +
  the block-local smoother.
- **Only for n > 1** - with nwinc=nhinc=1 there is no coarse/fine split; D is a
  no-op and today's cubel-global is already the coarse solve.

#### Implementation plan (locked 2026-10-23)

Decisions:
- **Coarse operator**: Galerkin `Z_c = S Z_f S^T` from `near_pair_data` - no
  re-mesh (inferred from the parameters; the flux-linkage identity makes the
  aggregate equal the un-subdivided operator).
- **`S`**: built from the two `n` params (`nw`, `nh`) directly. A non-uniform
  per-segment deck is a **user error**: `cubel-global-sub` returns
  `MeshError::NonUniformSubdivision`, not an assert.
- **Coarse precond radius**: reuse `CUBEL_RADIUS_MULT` (one knob for both).
- **Nested loop basis**: deferred (it is the mesher's; use the incidence route).
- **Smoother**: deferred until the rest works (the block-local `cubel` is the
  natural smoother).
- **New kind**: `cubel-global-sub` (selector
  `FASTERHENRY_PRECOND=cubel-global-sub`).
- **Diagnostics**: print the `R`/`P` dims + nnz + an `R P ~ I` sanity, and the
  coarse operator's symmetry/diagonals, so the construction is inspectable.

Phases:
0. Store `per_segment` on `IterativeSystem`; add the intra/forest split helper.
1. Coarse system: the forest rows + the Galerkin `Z_c`; run the existing
   `CubelGlobal::build` / `factor` on it (the coarse precond).
2. Adapters `R = M_c^T S M_f`, `P = R^T`; the `cubel-global-sub` kind + the
   apply `z = P (A_c^-1) (R v)`.
3. Validate: DC-bus nw2 @ mult 4, 5 MHz (must converge with the coarse-sized
   precond; L ~ the nw1's 7.09 nH); the nwinc=1 no-op (must equal
   `cubel-global`); print the matrices for inspection.

**Verdict.** The real lever for the subdivided case: it makes subdivision
"free" (the coarse factor stays un-subdivided-sized), i.e. it multiplies the
solvable mesh size. New code, but the fork's own structures carry the two
things that looked hardest (the incidence and the per-segment grouping). The
existing `PrecondKind::TwoLevel` is a *layer* split with a block Gauss-Seidel
(frozen-prone on cross-heavy patterns) - a different construction; this must be
a new kind.

**Effort.** Moderate-to-large: 1 enum arm + 1 builder (mechanical), the coarse
operator (the bulk), and `R`/`P` (the delicate part).

### E. f32 the precond

**What.** Store the CSC and factor in single precision.

**Context.** CSC entry = `row_idx: u32` (4 B) + `re`/`im` (8 B each) = 20 B;
the index does not shrink, so f32 gives 20 -> 12 B (**1.67x**, not 2x) on the
CSC, and roughly halves the factor fill. The precond apply runs in the GMRES
scalar `T` (= c64); a c32 factor needs a `c64<->c32` cast at the apply boundary
- one O(n) vector conversion per apply, negligible next to the triangular
solve. `FieldParts` already encodes the boundary-conversion pattern.

**Verdict (measured 2026-10-23).** Tried - not terrible, not good. DC-bus nw1
@ 5 MHz, mult 4: memory hwm 7922 -> 5948 MiB (-25%; the pFFT operator floors
it), but iterations 91 -> 267 (~3x) and wall 191 -> 275 s; the L is
bit-identical (7.0902 nH, R 1.3151 mOhm). The iteration cost is real, not an
estimate artifact: the c32 precond leaves the *true* residual at iter 91 at
2.4e-2 vs the f64's 2.5e-9, and the GMRES residual *estimate* pins at ~e-9
(the target) every cycle while the true walks down ~3 orders per cycle ->
spurious restarts. So it is a genuine memory-for-iterations trade, and a bad
one unless memory is the hard wall. **Shipped as a compile-time feature
(`--features precond-f32`); it costs nothing when off.**

**Wholesale c32?** The GMRES tolerance (1e-10 relative) sits below the c32
roundoff floor (~1e-7), so a wholesale-c32 solve would stall on the estimate
(the same pinning, now for the whole solve) unless the tolerance is relaxed
to ~1e-6; the result would still be ~1e-7 accurate (fine for an L), and
f16/bf16 (~1e-3) would be a disaster. Plumbing: the GMRES scalar is pinned
to `ComplexField<RealField = f64>` across the stack, so wholesale c32 is a
real refactor, not a flag. Not worth it for the ~2x it would buy.

**Effort.** Small-moderate (storage types + a `FieldParts` impl + the apply
cast).

### F. cubel-global "to disk"

**What.** Persist the CSC to disk (construct-to-disk writes each chunk in
order) and/or factor by streaming.

**Context.** Construct-to-disk is straightforward: the assembly is already
ordered contiguous chunks. But the *factor* is faer's in-RAM `Lu` (no
out-of-core), and the fill is the peak - and it is at least as large as the
CSC. mmap-ing the input still faults pages into RSS.

**Verdict.** Buys (almost) nothing for a fill-bound case. Not worth it as a
memory fix.

**Effort.** N/A.

### G. pFFT radius: memory and accuracy - the error ceiling

**What.** The *operator* near radius (`FASTERHENRY_PFFT_RADIUS`) trades memory
(~r^2 near pairs) for operator fidelity; the *precond* radius
(`CUBEL_RADIUS_MULT`) is memory-only (precond is safety-only). Two different
radii.

**Context.** The pFFT operator is exact near-field + interpolated far-field, so
its error is a bounded interpolation error whose ceiling propagates to the
result via the perturbation bound `dZ/Z <= kappa(A) * dA/A + residual`. So a
provable, data-dependent (worst-case, likely loose) ceiling exists - and is
**not computed** anywhere today.

**Verdict.** G is just a knob *unless* the ceiling is calculated. Adding a
"ceiling report" (the interpolation/grid error bound + the perturbation term)
would turn it into a knob with a known error budget. Nice-to-have.

**Effort.** Small-moderate (analysis + a reporting hook).

### H. cubel-global symmetric factorization

**What.** Exploit that `M Z M^T` is **complex-symmetric** (`Z = R + j w L`
symmetric by reciprocity).

**Context.** Today it is stored full and factored with faer's **general** `Lu`.
A triangular CSC + a symmetric factorization would ~halve the CSC *and* the
fill. But faer's simplicial `Ldlt` is *Hermitian* (real pivots); the
complex-symmetric case is the Bunch-Kaufman `LBL^T`, exposed only via faer's
supernodal low-level API. So it is a factorizer swap, not a drop-in.

**Verdict.** Doable; a real memory lever (2x) if the factorizer can be swapped
cleanly. After E/D.

**Effort.** Moderate.

## Recommended order

1. **D** (the coarse/fine subdivision precond) - the structural fix that makes
   subdivision affordable. (**E done**: the f32 precond shipped as a compile-time
   feature, measured -25% memory / +3x iters - not the lever, see E.)
2. **H** (symmetry) - the other 2x on the factor.
3. **G** - keep as a knob; add the error ceiling when wanted.
4. **Deferred, record-and-revisit:** A (warm start; ports first), B (retention
   drop - peak unaffected), C (rayon the block-local cubel - only if that path
   is revived).

## Open questions / caveats

- The subdivided global precond (cubel-global, mult 4) OOMed at 226k rows /
  283k branches on 58 GB; the coarse/fine precond (D) is the intended route past that.
- `FASTERHENRY_CUBEL_RADIUS_MULT=4` was the single biggest practical win on the
  solvent case (1576 -> 90 iters); the radius, not the frequency, was the
  cause (5 MHz and 100 MHz agree to 1 iteration).
- `bytes_per_solve` in the sweep's concurrency estimate omits the precond
  (CSC + fill), the dominant per-solve memory, so raising
  `SWEEP_MEMORY_BUDGET_BYTES` (a hardcoded 2 GiB) without fixing that estimate
  would overcommit and trip the RSS governor.

## F: cubel-global-sub - measured (2026-10-23)

Shipped and run on the server (192.168.1.26, 16 cores / 59 GB). The kind
factors the **coarse** (1-filament-per-segment) twin and applies
`z = P A_c^-1 R v`, `R`/`P` the loop-space selection of the fine conductor
block (the bundle rows restrict to zero).

Commands: `FASTERHENRY_PRECOND=cubel-global-sub FASTERHENRY_CUBEL_RADIUS_MULT=4
FASTERHENRY_MEMORY_BUDGET_GB=58`, 5 MHz, `--solver iterative`. Logs
`logs/dcbus_sub_nw1.log`, `logs/dcbus_sub_nw2.log`; deck inputs
`deck_dcbus_v05{,_nw2}.inp`.

**Local unit tests** (a 2x2-subdivided square ring): the identity
`M_c^T (R x) == S (M_f^T x)` holds to `worst |diff| = 0` over all unit loop
currents + a general current; fine conductor loops == coarse loops.

**nw1 no-op (un-subdivided deck).** `rc=0`, **91 iters**, `L = 7.0902 nH`,
`R = 1.3151 mOhm` - a bit-match to the locked `cubel-global` baseline (the
coarse twin is `None`, so it falls through). The no-op contract holds.

**nw2 (226,438 loops / 282,688 branches).**
- **Memory: solved.** The factor is on the coarse twin (`n = 14,422`, CSC
  17.3 M nnz, hwm **15.1 GB**, RSS ~16 GB) where `cubel-global` on the fine
  mesh OOMed at 57 GB. The D thesis - factor the un-subdivided size - is
  confirmed.
- **Mapping: verified.** `nesting: 14,422 coarse loops, 0 mismatched`;
  identity `worst |diff| = 0` (unit + general); `r_off = 212,016`, exactly
  `filaments_fine - filaments_coarse`.
- **Convergence: does NOT converge.** GMRES falls 44 -> 7.1e-2 over the first
  ~100 iters, then **hard-stalls** at `|r| = 7.13e-2` across restarts
  (cycle 1 end == cycle 2 end). Killed at iter 252.
- **Convergence failure root-caused: `M = P A_c^-1 R` is rank-deficient.**
  Its rank is the number of conductor loops (`col(M)` = the fine conductor
  block: 14,422 here) while the system has 226,438 unknowns; the 212,016
  bundle (intra-segment) rows map to **zero**, so they are structurally
  unreachable by right-preconditioned GMRES. The signature is unmistakable in
  the log: `est |r|` falls to ~1e-17 (GMRES thinks it converged) while the
  `true |r|` hard-stalls at 7.1e-2 across restarts - the residual component
  orthogonal to `col(A M)` can never be reduced, and a singular right
  preconditioner makes the internal estimate invalid.

  Reproduced at 16-filament scale (`/tmp/opencode/subdiag/small_nw2.inp`, a
  dcbus-dimensioned 4-segment ring at `nwinc=nhinc=2`), where the bug is
  unambiguous because the other preconds all converge in a handful of iters:

  | precond | small deck (nw2) | big nw2 (dcbus) |
  |---|---|---|
  | none | 12 iters | - |
  | jacobi | 10 iters | - |
  | cubel (block-local) | 11 iters | ~9.6e-2 after 2000 iters, still creeping (separate, non-stall) |
  | cubel-global | 2 iters | OOM (57 GB) |
  | **cubel-global-sub** | **fails: 5000 iters, `true` 1e-1 / `est` 1e-17** | **hard-stall 7.1e-2** |

  (The earlier draft of this entry claimed the stall was not the precond's
  fault; the small-deck reproducer refutes that. The big-nw2 block-local creep
  is a *separate* conditioning/scale question, not a hard stall - its residual
  keeps decreasing.)

  **Fix direction (not yet built):** the preconditioner must be full rank -
  compose the coarse correction with a full-rank bundle treatment, e.g.
  additive `M = M_bundle + P A_c^-1 R` or the multiplicative two-level
  `z = M_bundle^-1 v + P A_c^-1 R (v - A M_bundle^-1 v)`, with `M_bundle` a
  Jacobi or block-local solve on the intra rows. This is the deferred
  "smoother", now proven *required*, not optional.
