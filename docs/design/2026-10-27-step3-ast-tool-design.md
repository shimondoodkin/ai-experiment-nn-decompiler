# Design: step 3, the AST heuristics tool

Date: 2026-10-27
Status: draft for approval
Parent design: `docs/design/2026-10-27-nn-decompiler-design.md` (§5 step 3)

## 1. Intent

Steps 1–2 showed that Claude Code, working blind, turns a decompiled ReLU net into a clean program
for 26 of 29 step-1 models. Step 3 asks: **how much of that judgment is mechanical?** We read the
heuristics Claude wrote down, implement the recurring ones as a deterministic tool (no LLM), run it
on the same dumps, and measure how often it reproduces Claude's result. The headline number is
measured on models the tool's author never studied (held-out seed 2).

The result is interesting in both directions:
- if the tool reproduces most of Claude's refactorings, decompiling small nets is largely a
  mechanical compiler pass
- whatever the tool cannot reproduce marks where the LLM's judgment really matters

## 2. Held-out discipline

- **Distillation set:** the 23 runs that are handcrafted, seed 0 or seed 1. Only their sections of
  `HEURISTICS.md` were read when writing this design.
- **Held-out set:** the 6 seed-2 runs: absdiff, neareq, window, median3, tax, xor.
  - Nobody looks at their dumps, notes or results while building the tool.
  - Their `HEURISTICS.md` sections are not read.
  - The tool is run on them once, at the end, after the code is frozen. That run is recorded in
    the report as the headline, and the commit hash of the frozen tool is recorded with it.
- **Two distillation runs are audit-flagged** (neareq seeds 0 and 1 wrote scratch files to
  `/tmp`). Their notes were still read, so they count as in-sample. Their audit status is reported.

## 3. What the tool sees and may use

The same blind inputs as Claude Code:
- the dump program
- the blind spec (neutral input names, types, bounds, output kind)
- the checker against the original net

It never sees the task name, the target, or the training data. The checker is allowed because
Claude used it too. Candidate selection uses only the checker and program size, never the target.

## 4. Representation

The tool parses the program with `ast` (via `nndc.lang`) into a small **symbolic net**:
- hidden units, each `relu(affine)` over the previous layer's variables
- output affine form(s)
- an affine form = coefficients per variable plus a constant

The tool's input is always a `dump.py`, which has exactly this shape. Heuristics run as passes over
the symbolic net and over the emitted expression tree. The final program is emitted as source in
the program language, so the checker, the report and the later C backend treat it like any other
version.

*Deviation from the parent design, my call:* the parent says "one Python-AST rewrite per
heuristic". Most of Claude's heuristics are about neurons (dead units, directions, kinks), not
about syntax. So they operate on the symbolic net recovered from the AST rather than on raw AST
nodes. There is still one module per heuristic.

## 5. Heuristics → passes

Each pass names the `HEURISTICS.md` entries it comes from (run ids in brackets) and whether it is
exact (algebraic identity) or approximate (needs the checker).

| # | Pass | Source heuristics | Kind |
|---|---|---|---|
| P1 | **Bounds: dead / always-on units.** Interval-propagate the input box. A unit never > 0 is deleted. A unit never < 0 loses its relu and is inlined as affine. | deadredundant/hand, tax/seed0–1, window/seed0–1, absdiff/seed1, neareq/seed0–1, sum8/hand, xor/seed0, ramp/hand | exact |
| P2 | **Normalize and merge.** Rewrite each unit as `s·relu(û·x − t)` with ‖û‖₁ = 1. Merge units with the same direction and threshold by adding their output coefficients (s·c). Merging near-identical ones (direction within ε, threshold within δ of the input range) is approximate. | absdiff/seed0–1, neareq/seed0–1, deadredundant/hand, median3/seed1, xor/seed1, window/seed0 | exact for identical, approximate for near |
| P3 | **Snap.** Snap coefficients to 0, ±1, small integers or simple fractions, and thresholds to round values or half-steps between integers. Tolerance is relative to the input scale. Small terms are dropped only if their **combined** effect `Σ|wᵢ|·max|xᵢ|` is negligible. In 1-D, snap segment slopes rather than individual increments. | absdiff_perturbed/hand, sparse8/hand, sum8/hand, tax/seed0, xor/seed0–1, absdiff/seed0–1 | approximate |
| P4 | **Pair and motif identities** on the expression tree: `c·relu(z)+c·relu(−z) → c·abs(z)`; `c·relu(z)−c·relu(−z) → c·z`; `a+relu(b−a) → max(a,b)`; `a−relu(a−b) → min(a,b)`; `relu(z)−relu(z−k) → min(max(z,0),k)`; the hat `(+a,−2a,+a)` on equally spaced knots `→ a·d·max(0, 1−abs(x−k)/d)`; equal coefficients `→ c·sum(…)`. | max/hand, deadredundant/hand, absdiff/hand, ramp/hand, triangle/hand, window/hand, neareq/hand, sum8/hand, xor/hand | exact |
| P5 | **1-D spline reading.** If the output depends on one variable (an input or a derived direction from P2), read it as kinks and slope changes. Merge near kinks, drop tiny or edge kinks. Write convex shapes as `max(lines…)` and concave ones as `min(lines…)`. | tax/seed0–1, triangle/hand, ramp/hand, window/seed0, absdiff/seed0 | approximate |
| P6 | **Binary boundary.** For binary outputs whose units all lie on one direction d, find the score's zero crossings along d, snap them, and emit `abs(d − m) < r` (two symmetric crossings), `lo < d < hi`, or `d > t`. The score's magnitude is dropped. | neareq/seed0–1, window/seed0–1, window/hand, neareq/hand, xor/seed1 | approximate |
| P7 | **Truth table.** For binary outputs over a discrete domain of at most 4096 points: enumerate the net's decisions and match them against a library (each single input and its negation; all 2-input boolean functions over each pair; all/any/parity over all inputs; `sum(x) ≥ k`). | xor/seed0–1, xor/hand | exact over the domain |
| P8 | **Candidate fit** (black box). For regression with ≤ 4 inputs, sample the box and fit `y ≈ a·f + b` for each feature in a library: each input, sum, pairwise and all-input min/max, median of 3, `abs(xᵢ−xⱼ)`, `relu(xᵢ−xⱼ)`. If R² > 0.999, snap a and b and emit. | median3/seed0–1 | approximate |
| P9 | **Cleanup and emission.** Fold constants, inline identity layers and single-use units, factor equal weights, name variables (`d0 = x0 - x1` for derived directions), and emit readable code. | sparse8/hand, sum8/hand, all | exact |

Heuristics deliberately **not** implemented: Claude's "read the raw weights file" (the tool always
uses float32-exact constants) and "try the simplest candidate early" (this is the selection rule
in §6).

## 6. Search and selection

1. Build candidates along several routes:
   - **structural:** P1 → P2 → P4 → P9 (exact), then the same with P3 snapping (approximate), then
     + P5 where it applies
   - **boundary:** P6, for binary outputs
   - **truth table:** P7, for small discrete binary domains
   - **fit:** P8, for small regression problems
2. Check every candidate with the checker in snap mode.
3. Select among passing candidates by fewest AST nodes, with `PASS (exact)` as the tiebreak. The
   target is never consulted. *Changed during implementation:* the original order, exact first,
   meant the structural-exact route always won, because it is exact by construction, so a snapped
   program could never be chosen. "Simplest formula that passes" is also Claude's own heuristic
   (absdiff/seed0, xor/seed0).
4. If no candidate passes, fall back to the best exact structural result. That is always valid,
   because P1/P2/P4 are exact.

Every run writes `ast.py` and `ast_trace.json`. The trace lists the candidates tried, the passes
applied and their effect, each checker verdict, and which route won.

## 7. Measuring reproduction

For each model, compare `ast.py` with Claude's best valid attempt (`agent_n.py`):
- **passes check** (exact or snap)
- **recovered the expected formula** (task grading, report only)
- **size** in nodes, compared with Claude's
- **reproduces Claude:** passes check, has the same "recovered" status as Claude's attempt, and
  has at most 1.5× Claude's nodes (minimum allowance of 5 nodes)
- **route:** which route won

The report adds:
- an `ast` row per model in the main table
- a reproduction summary, with **held-out (headline) and in-sample shown separately**
- a count of how often each pass contributed to a winning candidate
- a list of the models where Claude succeeded and the tool did not. This list is the answer to
  "what needs an LLM".

## 8. Goals, non-goals, anti-goals

- **Goals:**
  - a deterministic, LLM-free tool (`nndc ast`) that works on all 29 step-1 dumps
  - an honest reproduction score
  - a per-pass trace that explains each rewrite
- **Non-goals:**
  - deep nets in general (median3's 2×16 layers are handled by P1/P2 where possible and otherwise
    by P8)
  - byte/fixture patterns (step 6)
  - speed (step 5)
- **Anti-goals:**
  - tuning the tool on held-out runs
  - choosing candidates by the target
  - claiming reproduction when the tool merely passes the check with a bigger program than Claude's

## 9. Left to the builder

Exact ε/δ values for near-merging and snapping (fixed before the held-out run and recorded in the
report), the boolean and feature library details within the lists above, internal data
structures, and the trace format.
