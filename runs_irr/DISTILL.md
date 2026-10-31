# Distillation: irrational-function agent pass (runs_irr, agent_1)

Sources: the 108 `runs_irr/<variant>/<task>/<seed>/agent_1.py` programs with their `results.json` verdicts, and the 107
sections of `runs_irr/HEURISTICS.md` (variant from `blind_index.json`). relu_wide/bump/seed1 is excluded (its heuristics
are in `HEURISTICS_excluded.md`, which was not read); its program is counted separately below. ast5 status: the code of
`families.py`, `conic.py`, `kinks.py`, `sigmoid.py`, `ast_tool.py` and the `ast5_trace.json` files present (2-D runs
only). For the 60 1-input runs, which have no ast5 output yet, the family route was run in memory on the blind spec
(scratch script, nothing written to `runs_irr/`). Counts in section 2 are from reading every section; a section counts
once per cluster.

Verdicts: 108/108 agents pass (3 exact, 105 snap). The 3 exact ones are all ReLU 1-D min/max-of-lines or a single line.

## 1. Final program forms (agent_1.py)

Codes: **F** plain formula (`exp(x0)`, `x0*x0`, `sqrt(x0*x0+x1*x1)`, `4 - r2`, incl. a snapped scale/offset);
**C** formula + edge clamp (`max(sqrt(x0), c)`, `log(max(x0, c))`, `max(log(x0), c)`, one C1 tangent-line patch);
**S** formula with a smoothing constant (`sqrt(r2 + c)`); **K** formula + correction term (odd/even shrink `p - p³/500`,
`x² - x⁶/1000`, clip `min(max(p, -3.8), 3.8)`, corner hinges, formula + residual units/patches);
**L** min/max of lines or one line (no relu); **H** line + 1–2 hinges; **N** pruned/snapped net, no formula
(`x` = exact).

| variant | exp | log | sqrt | square | recip | circle (bin) | dist | product | bump |
|---|---|---|---|---|---|---|---|---|---|
| relu (8/16) | F2 L1 | L3 | L2 (2x) H1 | F2 H1 | L2 (1x) H1 | F3 | F2 S1 | K3 | N3 |
| relu_wide (32/64) | F3 | F3 | F1 C2 | F3 | F3 | F3 | F3 | F3 | N1 K1 (+N1 excl.) |
| sigmoid (8/16) | F3 | C3 | C3 | F3 | F3 | F3 | N2 S1 | F3 | N3 |
| tanh (8/16) | F3 | F1 C2 | C3 | F1 K1 N1 | F3 | F3 | N3 | F3 | N2 K1 |

Totals (107 + 1 excluded): F 60, C 13, S 2, K 6, L 8, H 3, N 15 (+1). Formula-like (F/C/S/K) 81 of 107.

- Where a formula could not pass, it is always the same three situations: small ReLU nets on 1-D curves (L/H, 11 runs,
  the net's chord error exceeds 2%), the 2-D bump in every variant (N/K, ripple ≥ tolerance), and smooth-net `dist`
  (N, 5 runs; the cone tip is rounded off). relu_wide nets follow the formula almost everywhere.
- Sizes of the N cells (agent nodes vs ast5 nodes, from `ast5_trace.json`): sigmoid/dist 334, 283 vs 462, 392;
  tanh/dist 189, 106, 145 vs 470, 436, 464; sigmoid/bump 413, 413, 354 vs 476, 452, 466; tanh/bump 501, 468, 461 vs 474,
  468, 468; relu/bump 323, 418, 350 vs 284, 318, 331 (ast5 smaller). Pruning with refit wins on smooth nets
  (−30 to −75% on dist), not on ReLU bumps.
- 1-D, ast5 family route (in memory, blind spec): a 0-term formula passes on all 15 relu_wide, 15 sigmoid and 15 tanh
  nets, and on relu exp/1-2 and square/1-2. On the 13 C cells it passes **without a clamp**, by an argument shift or an
  offset: `sqrt(x0 + 0.01)`, `sqrt(x0 + 0.02) - 0.01`, `log(x0) + 0.02 … 0.05`, `sqrt(x0) + 0.02`. On tanh/square/1-2
  (agents: tanh basis, `x² − x⁶/1000`) it gives `x0 * x0 - 0.02` / `- 0.04`. On small relu nets it passes only with
  hinges, sometimes in the wrong family (relu/sqrt/1-2: `0.93 * log(x0 + 0.4346) + 0.25 * relu(1.25 - x0) + 0.6`;
  relu/log/2: reciprocal + 2 hinges); relu/square/0 has no passing family (agent: `0.05 + 1.6*relu(|x|-0.4) + …`).
- K on relu/product (27, 64, 22 nodes) vs ast5 (30, 56, 27): ast5 passes, but with degenerate shrink families,
  e.g. `1188 * exp(0.0008335 * x0 * x1) - 1188` and `108.5 * log(0.009233 * x0 * x1 + 1) - 18 * relu(u - 1.03) + …`.

## 2. Recurring heuristics (ranked by agent count)

"ast5" = whether the current route already does it.

| # | Heuristic (merged paraphrases) | Agents | From | Citations | ast5 |
|---|---|---|---|---|---|
| 1 | Tabulate the net on a coarse grid / landmark inputs and match a short library (exp, log, sqrt, 1/x, x², norms, product) before reading weights; confirm with an `a·f+b` fit (a≈1, b≈0) | ~95 | all variants, all tasks | sigmoid exp/seed1; relu_wide recip/seed2; tanh product/seed0; relu dist/seed1 | yes: the family route fits the library directly |
| 2 | The largest error of the right formula sits at a box edge/corner or a singular point (net saturates/underfits there); an edge-only error does not mean a wrong formula | ~65 | all variants; 1-D and product most | sigmoid product/seed1; relu_wide log/seed2; tanh recip/seed1; relu square/seed2 | implicit (the checker decides); no edge-specific move |
| 3 | Measure the candidate's **max** error (not RMS) against the 2%-of-range tolerance before investing in it | ~62 | all | tanh bump/seed1; relu_wide recip/seed2; relu exp/seed1 | yes (minimax + tol) |
| 4 | Dead / always-on ReLU units from kink −b/w or interval bounds over the box; fold always-on into one affine term; drop ~1e-5 weights | ~50 | relu, relu_wide (1-D all; 2-D circle, bump, dist) | relu log/seed0; relu_wide sqrt/seed0; relu bump/seed2; relu_wide dist/seed2 | yes exactly (P1 `prune_dead_and_linear`); near-dead only via snap/spline |
| 5 | Prefer the clean formula when it passes; do not fit scale/offset or corrections to the net's own error, do not chase exact | ~35 | sigmoid, tanh 1-D and product; relu_wide | sigmoid exp/seed2; relu square/seed2; tanh log/seed0 | yes (snapping + fewest nodes) |
| 6 | Keep a safety margin: accept snaps only below 60–90% of the tolerance (checker adds boundary/corner points) | ~35 | all; bump/dist most | sigmoid bump/seed2; relu bump/seed2; tanh dist/seed0; relu_wide bump/seed2 | **no**: snaps up to tol on the fit sample, no retry |
| 7 | 1-D ReLU: list kinks and segment slopes; monotone slopes = convex/concave (smooth curve); slope ∝ value = exp, linear growth = square, f' ~ 1/x = log | ~30 | relu, relu_wide 1-D | relu_wide exp/seed1; relu_wide log/seed1; relu exp/seed2 | n/a (the family fit replaces reading) |
| 8 | Read 2-D units geometrically (angle, offset −b/‖w‖, steepness), normalise signs (`σ(z)=1−σ(−z)`, tanh odd, `relu(z)=z+relu(−z)`); rings of units = circle/bump | ~30 | 2-D, all variants | sigmoid circle/seed1; tanh bump/seed2; relu bump/seed0; relu_wide bump/seed0 | partly (normalise, kink lines); not used for formulas |
| 9 | Smooth nets: saturated units are constants; shallow units (‖w‖·range ≲ 1) are near-linear: replace their **sum** by one affine (or quadratic) term fitted **jointly** with the rest | 18 | sigmoid, tanh (dist, circle, exp, square) | sigmoid dist/seed1; tanh dist/seed0; tanh dist/seed2; tanh exp/seed0 | partly: per-unit const/line in `replace_units`, fixed Taylor line, no joint refit |
| 10 | Snap one parameter at a time, coarse grid first, refit the free ones after each; hidden weights first, output weights last and finer | 16 | sigmoid/tanh bump, dist; relu bump | sigmoid bump/seed2; tanh dist/seed1; relu bump/seed0 | formulas yes (`_snap`); nets no (`snap_net` rounds once, no refit) |
| 11 | Unstructured ripple ≥ tolerance means no formula can pass: stop the formula search, compress the net (formula as documentation) | 15 | bump all variants; sigmoid/tanh dist | sigmoid bump/seed0; tanh bump/seed1; relu bump/seed2 | partly: near formula in the trace only |
| 12 | Refit the output layer by **minimax LP** (not least squares) after every pruning/snapping step | 14 | sigmoid, tanh, relu bump/dist | sigmoid bump/seed1; sigmoid dist/seed1; relu bump/seed0; tanh bump/seed2 | formulas only (Lawson); `knot_fit` LP on sigmoid 1-D; no net refit |
| 13 | Prune by greedy backward elimination with an output refit (free affine term); stop when the best removal exceeds the budget | 13 | sigmoid/tanh bump, dist; relu(_wide) bump | sigmoid dist/seed0; tanh dist/seed0; relu_wide bump/seed0 | **no** |
| 14 | Singular edge (sqrt/log at 0, 1/x): patch with a clamp `max(f, c)` / `f(max(x, c))` (or a tangent line), c from the net's edge value, rounded; a shift `sqrt(x + c)` spreads the error and fails | 13 (6 reject the shift) | sigmoid, tanh log/sqrt; relu_wide sqrt | sigmoid log/seed1; sigmoid sqrt/seed0; tanh sqrt/seed2; relu_wide sqrt/seed2 | no clamp, but not needed to pass: the minimax-fitted shift/offset passes all 13 nets (the agents' "a shift fails" is not borne out) |
| 15 | Binary 2-D: bisect the decision along rays from the centre; constant radius = circle; take r from the boundary, not from the unit lines (tangent lines sit inside) | 12 | circle, all variants | tanh circle/seed1; sigmoid circle/seed0; relu circle/seed0 | yes (conic route) |
| 16 | Units on (1,1)/(1,−1) with opposite-sign even ridges = product (polarisation identity) | 11 | product, all variants | sigmoid product/seed1; relu_wide product/seed2; tanh product/seed1 | yes (quadratic family has x0·x1) |
| 17 | Merge kinks closer than ~1% of the domain; kinks within 1–2% of a bound act linear; "try k−1 lines" with a refit | 10 | relu 1-D, relu_wide | relu log/seed1; relu sqrt/seed1; relu log/seed0 | partly (spline MERGE_REL 1%, EDGE_REL 2%); no segment drop with refit |
| 18 | Fit points must include the box border, corners and the worst points of a fine grid (cutting planes); check on a finer grid than the fit | 9 | sigmoid dist, relu bump, tanh bump | sigmoid dist/seed1; sigmoid dist/seed2; relu bump/seed0 | **no** for 2-D (20k uniform + 200 extreme-y points, no border) |
| 19 | Mirror pairs / symmetry: tie the parameters of mirrored units (one try, not forced) | 8 | tanh dist, square; relu square | tanh dist/seed1; tanh square/seed1; relu square/seed0 | no |
| 20 | Product/odd edge sag: model with an odd shrink `p − p³/k`, a clip at the net's extreme, or `relu(|s| − k)` corner terms; even sag `x² − x⁶/1000` | 4 | relu product; tanh square | relu product/seed0; relu product/seed2; tanh square/seed2 | no such terms; exp/log fits imitate them |
| 21 | Norm test: biases ≈ 0 = positively homogeneous; probe rays; corner/edge ratio √2 / 2 / 1 = L2 / L1 / L∞ | 6 | dist, relu(_wide), sigmoid | relu dist/seed2; relu_wide dist/seed1; sigmoid dist/seed2 | L2 yes (sqrt family); L1/L∞ not needed here |
| 22 | Rounded kink/apex: `sqrt(t² + c)` with √c ≈ the net's value at the apex | 2 (+4 accept the apex error) | relu, sigmoid dist | relu dist/seed1; sigmoid dist/seed2 | yes (sqrt family keeps p5 free) |
| 23 | Formula + residual patch (formula as a basis column + refit units; sparse-LP pyramid bumps) | 2 | tanh bump, relu_wide bump | tanh bump/seed0; relu_wide bump/seed2 | no |

Smaller clusters (≤ 9): positive-domain hint for log/sqrt/1/x (9); residual shape, sign changes = noise vs trend =
missing term (7); return a signed score for binary nets (7; conic emits `q < c`, equivalent); keep an exact pruned
fallback (6; ast5 has structural-exact).

## 3. Candidate tool changes

Ranked by support x plausible gain. 1-D smooth and relu_wide nets need nothing new: the family route already passes a
0-term formula on all 45 of them. The gains are in 2-D and in the nets with no passing formula.

1. **Border-aware 2-D fit sample and margin retry** (clusters 6, 18). Add the box border and corners to the 2-D fit
   sample (it is 20k uniform points + 200 extreme-y points today); when a snapped family program fails the checker, add
   its counterexamples, refit and re-snap at 0.8·tol (then 0.6·tol). Evidence: 16 family rows in 7 nets (relu/dist 0, 2;
   relu/product 0, 1; sigmoid/dist 0-2) reached <= 2.0% on the fit sample and then failed the checker; on
   sigmoid/dist/0-1 (agents: no formula) the sqrt family with 4 hinges came within 1.9-2.0%, and explain puts the
   0-term error peak at the box edge (3, -2.99). Plausibly fixes sigmoid/dist/0-1. Complexity: low.
2. **Prune with a minimax LP output refit** (clusters 12, 13, 10, 11). For nets with no passing formula: greedy backward
   elimination of hidden units, each step re-solving the output layer (plus a free affine term) by LP on grid + border +
   counterexample points; then snap hidden weights one at a time (coarse first) with an LP re-solve, output weights last
   and finer. Gains are size, not formula: tanh/dist 0-2 (agents 106-189 vs ast5 436-470 nodes), sigmoid/dist 0-1
   (283-334 vs 392-462), sigmoid/bump (354-413 vs 452-476). Not relu/bump (agents 323-418 vs ast5 284-331).
   Complexity: medium-high (LP size and runtime for 16-64 units; scipy `linprog` HiGHS, with the `highs-ipm` fallback
   tanh/bump/seed0 needed).
3. **Joint fold of near-linear smooth units** (cluster 9). Units whose pre-activation range over the box is small become
   one fitted `c + a·x0 + b·x1` (+ quadratic) term, refit jointly with the remaining output weights, instead of
   `replace_units`' fixed per-unit Taylor line kept only if the whole program passes. sigmoid/dist/seed1 reports the
   affine term took 10 units from 0.039 to 0.013. Same nets as 2; complexity low once 2 exists.
4. **Fewer-segment min/max of lines for small ReLU 1-D nets** (clusters 17, 7). When no family passes at 0 terms: drop
   the weakest kink, refit the remaining lines by minimax LP (the `knot_fit` machinery, today sigmoid-only), emit
   `max/min(lines)`. Fixes: relu log/1-2 and sqrt/1-2, where the family route passes only with hinges and in the
   wrong family (`0.93 * log(x0 + 0.4346) + 0.25 * relu(1.25 - x0) + 0.6` on a sqrt net, reciprocal on log/2), and
   relu/square/0 (no family passes). Complexity: low. Check first whether spline P5 (which keeps every kink) already
   wins these in a full ast5 run.
5. **Odd/even shrink and clip terms on u** (cluster 20). Next to hinges in u, allow `k·u³` or `min(max(u, -c), c)` on
   the 2-D quadratic. Fixes relu/product/0-2 readability: ast5 passes them with `1188 * exp(0.0008335 * x0 * x1) - 1188`
   and `108.5 * log(0.009233 * x0 * x1 + 1) - …`, agents with `p - p*p*p/500` and `min(max(p, -3.8), 3.8)`. Support is
   thin (4 agents). Complexity: low (one family or correction column).
6. **Optional: edge clamp as a readability variant** (cluster 14, 13 agents). `a·f(max(x, c)) + d` /
   `max(a·f(x) + d, c)` with c scanned over nice values at the singular edge. It fixes nothing (shift/offset already
   passes on sigmoid log/0-2, sqrt/0-2, tanh log/1-2, sqrt/0-2, relu_wide sqrt/1-2) and costs about the same nodes;
   worth it only if the project prefers "formula exact in the interior" to "formula shifted everywhere". Complexity: low.

Checked and rejected or already covered: smoothing constant `sqrt(q + c)` (ast5 frees p5 and finds
`sqrt(x0*x0 + x1*x1 + 0.01*x0 + 0.02*x1 + 0.1061) - 0.04` on sigmoid/dist/2); min/max of lines for exact convex/concave
1-D ReLU nets (spline P5); exact dead/always-on folding (P1); folding near-linear sigmoid/tanh units into a polynomial
(covered by 3, not a separate route); product corner clip as a separate route (covered by 5); formula + residual patch
(2 agents; tanh/bump/0's version is larger than the dump).
