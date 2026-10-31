# Irrational functions: how nets model them, and snapping them to formulas

Status: approved in chat ("Approve, write plan"), 2026-10-30. Follows `2026-10-30-ast4-design.md`. Plan:
`docs/plans/2026-10-30-irrational-functions.md`.

## Why

Every task so far has a piecewise-linear target (abs, max, brackets, thresholds), so a ReLU net can be exact and a
program of `if`/`max`/`relu` can match it. The user asked for the next stress test: functions that are not piecewise
linear, which the user calls **irrational functions**: exp, log, sqrt, the circle `x0² + x1² < r²` and similar.

The user named two questions:
1. **How does the net model the function?** Where a ReLU net puts its kinks, where a sigmoid net puts its steps, what
   a circle's boundary looks like inside a ReLU net.
2. **Can the tool snap it to a formula plus terms?** That is, find `exp(x0)` (plus a few correction terms where the net
   is off) instead of a long chain of hinges.

**The spike** (2026-10-30, throwaway, scratchpad only; the build copies it to `experiments/spikes/irrational/`):
- 9 tasks (the task list below) × 3 nets: ReLU, sigmoid and wide ReLU (4× hidden units, 3× steps), seed 0.
- For each net it fitted a fixed set of families `a·f(b·x + c) + d` to the **net** (not the target), snapped the
  parameters one at a time with a refit, and checked the snapped program against the net with the checker.
  Families that came within 25% got greedy hinge correction terms in 1-D.
- **Results** (✓ = the snapped formula passes against the net):

  | task | ReLU | sigmoid | wide ReLU |
  |---|---|---|---|
  | exp | ✗ (4.5% off) | ✓ | ✓ |
  | log | not learned | ✓ | ✓ |
  | sqrt | not learned | ✓ with 1 hinge term | ✓ |
  | 1/x | not learned | ✓ | ✓ |
  | x0·x0 | ✗ (3.5% off) | ✓ | ✓ |
  | circle | ✓ | ✓ | ✓ |
  | dist | ✓ | ✗ (8.3% off) | ✓ |
  | product | ✗ (3% off) | ✓ | ✓ |
  | bump | not learned | ✗ (2.6% off) | ✗ (2.9% off) |

  - The formula passed on 16 of the 23 nets that learned, plus 1 more with one extra term.
  - The best family was the right one on every net where any family passed.
  - `ast4` passes too, but it is long: 50–265 nodes in 1-D and 260–1142 in 2-D, against 3–10 for the formula.
    On 2-D sigmoid nets `ast4` takes about 10 minutes per net.
- **How the nets modelled the curves:**
  - Wide ReLU kinks roughly follow the curvature-optimal spacing (knot density ∝ √|f″|).
  - Small ReLU nets (8 units) stack units on one spot (exp: 3 kinks at x ≈ 0.09, so only 2 real kinks). 4 of the 9
    small ReLU nets did not learn at all.
  - Sigmoid centres bunch where the function bends (exp: 1.46–1.79).
  - The ReLU circle's boundary is a 20-sided polygon (wide: 88 sides), radius 1.99–2.05 for r = 2.
- **The catch:** at the checker's 2% tolerance, 3–12 plain hinges also pass on the 1-D tasks. So the formula has to
  win on size, and the report has to say whether it did.
- **Limits:** one seed; the family list was written knowing the tasks; the 2-D families assumed the centre is at the
  origin (`x0² + x1²`), which a blind tool may not assume; no 2-D correction terms; no tanh.

## Goals

1. **Language** (`nndc/lang.py`):
   - Add the functions `exp`, `log` and `sqrt`, and allow division by any expression, not only by a constant.
   - Outside its domain (log or sqrt of a negative number, division by zero) a program gives NaN or ±inf. The
     checker counts every non-finite output as a disagreement. NumPy warnings are suppressed while evaluating.
   - Each new function call counts as one node, like `relu`.
   - The language description that agents read (the text in `lang.py`) lists the new functions.
   - `lang.py` stays stdlib + NumPy. `compile_back` and the `regions` baseline refuse programs that use the new
     functions, as they refuse sigmoid ones.
2. **Tasks** (`experiments/irrational.py`, `IRRATIONAL_TASKS`). They are kept out of `TASKS`, like the dataset tasks;
   `get_task` resolves them. Each has a true target and an `expected` formula, written now, before any run:

   | task | inputs | output | `expected` | hidden |
   |---|---|---|---|---|
   | `exp` | x0 ∈ [-2, 2] | regression | `exp(x0)` | [8] |
   | `log` | x0 ∈ [0.1, 5] | regression | `log(x0)` | [8] |
   | `sqrt` | x0 ∈ [0, 4] | regression | `sqrt(x0)` | [8] |
   | `square` | x0 ∈ [-2, 2] | regression | `x0*x0` | [8] |
   | `recip` | x0 ∈ [0.5, 4] | regression | `1/x0` | [8] |
   | `circle` | x0, x1 ∈ [-3, 3] | binary | `x0*x0 + x1*x1 < 4` | [16] |
   | `dist` | x0, x1 ∈ [-3, 3] | regression | `sqrt(x0*x0 + x1*x1)` | [16] |
   | `bump` | x0, x1 ∈ [-3, 3] | regression | `exp(-(x0*x0 + x1*x1))` | [16] |
   | `product` | x0, x1 ∈ [-2, 2] | regression | `x0*x1` | [16] |

   - All inputs are real-valued, sampled uniformly. Training: 6000 steps, the default learning rate.
   - Each task also names its **family** for the report (exp, log, sqrt, square, reciprocal; `circle` → conic;
     `dist` → sqrt; `bump` → exp; `product` → square, whose inner quadratic is `x0·x1`). Only the report reads it,
     never the tool (see "Scoring").
   - The circle's spec threshold and inside/outside convention follow the existing binary tasks.
3. **Nets: four variants**, each in its own root so the existing per-root commands work unchanged:

   | variant | root | activation | size |
   |---|---|---|---|
   | ReLU | `runs_irr/relu/` | ReLU | the table's hidden sizes |
   | wide ReLU | `runs_irr/relu_wide/` | ReLU | 4× hidden units, 3× steps |
   | sigmoid | `runs_irr/sigmoid/` | sigmoid | the table's hidden sizes |
   | tanh | `runs_irr/tanh/` | tanh | the table's hidden sizes |

   - `nndc run` gets a `--wide` option (4× hidden, 3× steps); `spec.json` or `train_log.json` records it.
   - **Development seeds: 0–2** (9 tasks × 4 variants × 3 seeds = 108 nets).
   - Every run gets `formula.py` (the `expected` formula as a reference version), as the sigmoid formula runs do.
4. **A new tool version, `ast5`** (`ast5.py`, `ast5_trace.json`, `counterexamples_ast5.json`): `ast4` plus a **formula
   route**. Every other route is as in `ast4`, and the winner is still the checker-passing candidate with the fewest
   nodes, exactness breaking ties. `ast4` stays runnable and gives byte-identical programs.

   The formula route sees only the dump, the blind spec and the weights, like every route. It tries every family on
   every net; it never learns which task a net is.
   - **Families, regression** (u is the inner variable):
     - 1 input: `a·f(u) + d` with `u = b·x0 + c`, for f in {identity, square, exp, log, sqrt, reciprocal}.
     - 2 inputs: the same f, with u a **general quadratic** `p0·x0² + p1·x1² + p2·x0·x1 + p3·x0 + p4·x1 + p5` (the
       blind version of the spike's `x0² + x1²`; it also covers `x0·x1`).
   - **Fit:** least squares against the net's outputs, on a fit sample (1-D: a 2001-point grid; 2-D: 20k uniform
     points plus the 200 points where the net is most extreme), several fixed starting points per family. Families
     whose fit error is over 25% of the output range are dropped.
   - **Snap:** one parameter at a time, the one that snaps at the coarsest level first, with a refit of the free
     parameters after each, using `nice()` at levels 0.1, 0.05, 0.02, 0.01, 0.005, 0.001 (relative to max(1, |p|)).
     A parameter stays as fitted when no level keeps the error within tolerance. The tolerance is the checker's
     (2% of the net's output range).
   - **Correction terms:** if the snapped family does not pass, add up to 4 hinge terms `k·relu(u − t)` or
     `k·relu(t − u)` in the family's inner variable u, greedily (knots t from a 97-point grid over u's range,
     coefficients by least squares), then snap k and t. In 1-D, u is `x0` itself. Each step is checked; the first
     program that passes is the family's candidate.
   - **Binary outputs (2 inputs):** find boundary points by bisection between sample pairs with different decisions;
     fit a conic (`q(x) = 0`, q a general quadratic, by the smallest singular vector); snap its coefficients like the
     regression parameters, with the checker deciding; emit `q(x) > c` or `q(x) < c`, the side taken from the net.
   - The trace records every family tried, its fit and snapped errors, the number of correction terms, and whether it
     passed, so the report can say whether a formula was found even when it was not the smallest program.
   - Argmax outputs and nets with more than 2 inputs: the route does not apply.
   - `nndc ast` runs `ast5` only when asked (`--name ast5`) until the user agrees to make it the default. It runs on
     the `runs_irr/` roots only (see Non-goals).
5. **The comparison:** `ast4` and PySR also run on every irrational net.
   - **PySR**: the step-4 settings (`nndc/baselines/pysr_run.py`) plus the unary operators `exp`, `log`, `sqrt` and
     the binary operator `/`. These settings are fixed here and not tuned. It writes new versions `pysr_irr.py` and
     `pysr_irr_snap.py`, since the settings differ from step 4's `pysr.py`.
6. **Freeze, then fresh held-out nets,** built after the freeze and run once:
   - seed 3 of the 9 development tasks, all 4 variants (36 nets);
   - **4 held-out tasks**, seeds 0–1, all 4 variants (32 nets). Their formulas are inside the family list but with
     constants and centres the development tasks do not have. They are written here, before any tool code, and tool
     development must not use them:

     | task | inputs | output | `expected` |
     |---|---|---|---|
     | `decay` | x0 ∈ [-1, 3] | regression | `3*exp(-0.5*x0)` |
     | `logshift` | x0 ∈ [-1.5, 4] | regression | `log(x0 + 2)` |
     | `ellipse` | x0, x1 ∈ [-3, 3] | binary | `(x0 - 0.5)*(x0 - 0.5)/4 + x1*x1 < 1` |
     | `hypot1` | x0, x1 ∈ [-2, 2] | regression | `sqrt(x0*x0 + x1*x1 + 1)` |

     Hidden sizes as for the development tasks of the same input count. Families: `decay` → exp, `logshift` → log,
     `ellipse` → conic, `hypot1` → sqrt.
   - `runs_irr/<variant>/heldout.json` names the held-out runs (`version: ast5`, the freeze commit).
7. **Report** (`runs_irr/REPORT.md`, built by a new report command over the four roots):
   - **Results table**, one row per net: learned or not, true-vs-net error, the formula passes or not, `ast4` nodes,
     `ast5` nodes and winning route, the best family and its correction-term count, PySR's result.
   - **"How the nets model the functions"** (goal 1), per task and variant:
     - 1-D: real kinks or sigmoid centres in the box (stacked units counted once), dead units, and their positions
       against the curvature-optimal knots for the same count; where the net's error against the true function is
       largest (an edge or the interior).
     - circle and ellipse: the boundary's number of straight pieces (ReLU), the fitted conic and its radius spread.
     - 2-D regression: the net's error against the true function by distance from the centre, and which correction
       terms the route needed.
   - **"Snapping to formulas"** (goal 2): per variant, how many learned nets got a passing family program, with how
     many correction terms, and how often it was the smallest program; whether a hinge-only program of fewer nodes
     also passed (the catch).
   - **Held-out:** the same counts on the 68 held-out nets, against the predictions below.
   - Plots: `experiments/irrational_plots.py` writes one PNG per task and variant (seed 0) into `runs_irr/plots/`:
     net, true function and formula program, kinks marked (1-D); the decision boundary or the error map (2-D). It
     needs matplotlib, in a new `plots` dependency group; the report links the PNGs but does not need them.
8. **`CONCLUSIONS.md`:** an "Irrational functions" section when the project ends.

## Scoring (report only; the tool never sees any of it)

- **Learned:** the training log's existing rule (regression r² ≥ 0.99; binary accuracy as for the other tasks).
- **Formula found** on a net: some passing `ast5` formula-route candidate uses the task's family (the column in the
  task tables), with at most 4 correction terms. The identity family, with or without hinges, never counts as a
  formula: it is a hinge program. A wrong family that passes is counted and listed separately ("other family").
- **The route's pick** among its passing candidates is the one with the fewest nodes, then the lower error. The
  report states whether that pick is the task's family.
- **The results table has two formula columns:** "`formula.py` passes" (the intended formula checked against the
  net, as in the sigmoid runs) and "formula found" (the route).
- **The hinge-only comparison (the catch), report only:** in 1-D, the spike's greedy from-scratch hinge fit
  (`c0 + c1·x0 + Σ k·relu(x0 − t)`, up to 16 hinges, 97-point knot grid, the checker deciding) and its node count.
  It is not an `ast5` candidate, so it cannot change `ast5`'s results. In 2-D the comparison is `ast4`'s program.
  "The formula won on size" means the formula candidate has fewer nodes than both the hinge fit and `ast4`.
- **PySR found the formula:** its snapped program passes and uses the task family's function (`exp`, `log`, `sqrt`,
  `/`, or a product of inputs for square and conic).

## Development and the freeze

- The route is first built exactly as written here and run once on the development nets. Those numbers are reported
  as the **first development run**, against the predictions below.
- After that, the route may be changed using the development nets only. Every change is an amendment in this
  document, the development numbers are re-reported, and the predictions are not rewritten. Then the freeze, then the
  held-out nets. Only the held-out result tests the frozen tool.

## Predictions (written before any tool code)

All counts are over learned nets. "A seed" means any of the development seeds 0–2.
- **Learning:** on each development seed, at least 3 of the 9 small ReLU nets do not learn. Wide ReLU, sigmoid and
  tanh each learn at least 8 of 9 tasks on at least 2 of the 3 seeds.
- **First development run:** formula found on at least 70% of the learned nets. Where it is found, `ast5`'s program
  is at least 5× smaller than `ast4`'s.
- **Bump:** formula found on at least one net (any variant, any seed). It failed at 2.6–2.9% in the spike.
- **Tanh vs sigmoid:** the formula-found count over tanh's 27 development nets is within 3 of sigmoid's.
- **Held-out:** formula found on at least 60% of the learned held-out nets, including at least one `ellipse` net (a
  centre off the origin) and at least one `decay` net.
- **PySR:** finds the formula on fewer development nets than `ast5`, and on at least 5.

## Non-goals

- A general symbolic-regression search over expression trees. PySR is that; this project uses a fixed family list.
- Periodic functions (sin, cos), more than 2 inputs, argmax nets, two hidden layers.
- Running `ast5` on the existing `runs/`, `runs_sigmoid/` and `runs_tanh/` roots, or making it the default. That is a
  later decision, after the held-out result.
- Blind Claude agents on these nets.
- Changing the checker's tolerance for these tasks.

## Anti-goals

- Any change to existing results: `ast4` programs regenerate byte-identically, the ReLU equivalence script shows 0
  differences, and no older version's verdict changes.
- A task name, `expected` formula or target reaching the tool's inputs. The family list is the same for every net.
- Adding a family or tuning the route after seeing a held-out net, or building held-out nets before the freeze.
- Calling a formula "found" when it passes only because the net is so blurry that anything passes. The report puts
  the hinge-only size next to the formula's.

## Constraints

Everything in CLAUDE.md: data commits apart from code commits, the held-out discipline, no destructive commands on
the repo or `runs*/`, and tests that delete use `tmp_path`.

## Left to the builder

The report command's name and layout details, the starting points of each family fit, the bisection details for
boundary points, the plot styling, and how the route's code is split across `nndc/heuristics/`.

## Amendments

- **2026-10-31, the agent-first method is restored** (the user, in chat: "yes to all, maybe even first then develop
  the tool then run the tests"). The non-goal "Blind Claude agents on these nets" is reversed. Development returns to
  the project's original method (`2026-10-27-nn-decompiler-design.md`, step 3), which kink merge, `ast4` and this
  project had dropped:
  1. **Agent pass:** one fresh blind subagent per development net, all 108 (4 variants × 9 tasks × seeds 0–2). Same
     `INSTRUCTIONS.md`, blind folders and audit as the earlier steps: no task name, target or family. Index
     `runs_irr/blind_index.json`, heuristics to `runs_irr/HEURISTICS.md`. The programs are collected after the
     gate-view rerun (`2026-10-31-formula-residual-gate-view-design.md`, Goal 7) finishes, so two jobs never write one
     `results.json`. That rerun still gives the gate-view design its first numbers.
  2. **Distill:** the next route changes (plan Task 9) come from heuristics that recur in `runs_irr/HEURISTICS.md`;
     each change cites the entries it came from.
  3. **Reproduction:** the report adds, per development net, whether `ast5` reproduces the agent's result (both pass;
     `ast5` is not larger; the same formula family where the agent wrote one), as in step 3.
  4. **Held-out:** after the freeze, agents also run on the 68 held-out nets, with heuristics to
     `runs_irr/HEURISTICS_heldout.md`, which development never reads. Reproduction on the held-out nets is the
     headline.
