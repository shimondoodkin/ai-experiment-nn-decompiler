# Distilled Route Changes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the four `ast5` route changes distilled from the agent pass, the reproduction measure (does `ast5` reproduce
each agent's result), and the development rerun that re-reports the numbers before the freeze.

**Architecture:**
- **Reproduction** is task-aware, so it lives in the analysis (`experiments/irrational_analysis.py`) and the report
  (`nndc/irrational_report.py`).
- **The four changes:**
  - change 1 is in the family route (`nndc/heuristics/families.py`);
  - change 4 is in the conic route (`nndc/heuristics/conic.py`);
  - changes 2 and 3 are a new route in `nndc/heuristics/prune.py`. `refactor` runs it only when no family or conic
    program passes.

**Tech Stack:** Python 3.13, NumPy, SciPy (`least_squares`; `linprog` with HiGHS, through `sigmoid.minimax`).

**Spec:** `docs/design/2026-10-30-irrational-functions-design.md`. The binding sections are under "Amendments":
- "2026-10-31, the agent-first method is restored", item 3 (reproduction);
- "2026-10-31, route changes distilled from the agent pass", changes 1–4.

This plan carries out Task 9 of `docs/plans/2026-10-30-irrational-functions.md` up to the freeze; the freeze and the
held-out nets stay in that plan's Tasks 9–10.

## Global Constraints

- **Change 1:**
  - border: 250 points per box edge, corners included (`N_BORDER = 250`);
  - margin retry: tolerance scales 0.8, then 0.6 (`MARGIN_SCALES = (0.8, 0.6)`);
  - at most 2 retries per family (`MARGIN_RETRIES = 2`);
  - a retry happens only when the snapped program was within the tolerance on the fit sample and the checker failed
    it with counterexamples.
- **Change 2:**
  - regression nets with one hidden layer and 1–2 inputs, only when no family or conic program passes;
  - fit points: 1-D a 2001-point grid, 2-D a 61 × 61 grid (the border is on the grid), plus the checker's
    counterexamples;
  - output layer: re-solved by minimax LP with a free affine term after every removal and every snap;
  - stop removing when the best removal would exceed 0.8 · tol (`PRUNE_BUDGET = 0.8`);
  - removal candidates are screened by least squares, and the best 3 are confirmed by LP (`N_SCREEN = 3`);
  - hidden weights snap one at a time at `families.SNAP_LEVELS`, output weights after them at
    `OUT_SNAP_LEVELS = (0.01, 0.005, 0.001)`;
  - LP: HiGHS (`method="highs"`), then `"highs-ipm"`;
  - after a checker failure: at most 2 rounds of adding counterexamples (`PRUNE_RETRIES = 2`);
  - route name `prune-lp`, passes `["F3"]`, appended last to `ROUTES`; trace key `prune_lp`.
- **Change 3:**
  - inside the prune route, before elimination, fold every unit whose pre-activation range over the box,
    `max z − min z`, is at most 1.0 (`FOLD_WIDTH = 1.0`), and every ReLU unit that is dead or always on over the box;
  - the folded units' sum is replaced by LP columns `1, x` plus, when any sigmoid unit folded, the quadratic atoms
    (1-D `x0²`; 2-D `x0², x1², x0·x1`), all re-solved jointly with the remaining output weights.
- **Change 4:**
  - zero each of the conic's `x0·x1`, `x0`, `x1` coefficients whose largest term on the boundary points,
    `|w_i · atom_i(P)|`, is below 2% of the threshold (`CLEAN_FRAC = 0.02`);
  - refit the threshold as the mean of the left side over the boundary points;
  - round it with `decision_list.nicest_in` within ±2%;
  - check the clean program first and keep it if it passes, else the snapped one.
- **Reproduction** (per development net with an agent attempt that is not excluded):
  - the agent's program and `ast5` both pass;
  - `ast5`'s nodes are not more than the agent's;
  - where the agent's program uses the task family's function (`FAMILY_TOKENS`), `ast5`'s does too.

  Excluded runs: `runs_irr/agent_excluded.json` (relu_wide/bump/seed1, whose agent saw run paths).
- **File name:** the route stays `ast5` (pre-freeze development, as with minimax). Before the rerun, the per-run state
  is saved to `runs_irr/before_distill.json`; the first development run's files are in git at `2962329`.
- **Anti-goals:**
  - `ast4` regenerates byte-identically;
  - `experiments/equivalence.py` shows 0 differences;
  - `sigmoid.minimax`'s default behaviour is unchanged;
  - no task name, real name, expected formula or target in the routes' inputs;
  - data commits apart from code commits, no attribution lines, no destructive commands on the repo or `runs*/`;
  - tests that delete use `tmp_path`.

## Review Focus

1. **A prune-route net where no removal fits the budget** still gives the refit, snapped full net (the checker and the
   node count decide), not None. Test in Task 4.
2. **An LP that fails with both methods** makes the prune route return None with a trace note, never raise. Test in
   Task 4.
3. **A conic of the `>` form or with a negative threshold:** the clean variant keeps the comparison direction and the
   sign of the threshold. Test in Task 3.
4. **A margin retry when the checker fails with no counterexamples** (a non-finite program): no retry, no error. Test
   in Task 2.
5. **Fold on a ReLU net:** a unit that changes sign in the box is never folded; an always-on unit folds exactly.
   Test in Task 5.

---

### Task 1: Reproduction measure and the before snapshot

**Files:** Modify `experiments/irrational_analysis.py`, `nndc/irrational_report.py`; Create
`runs_irr/agent_excluded.json` (data); Test `tests/test_irrational_analysis.py`, `tests/test_irrational_report.py`

**Interfaces:**
- Produces: `reproduced(agent_pass: bool, ast5_pass: bool, agent_nodes: int | None, ast5_nodes: int | None,
  agent_formula: bool, ast5_formula: bool) -> bool`.
- Produces: new `irr_analysis.json` keys, all `None` when the run has no `agent_1` or is excluded:
  - `agent_pass`, `agent_nodes`;
  - `agent_formula` and `ast5_formula` (`pysr_found(src, family)` on `agent_1.py` / `ast5.py`);
  - `reproduced`.
- Produces: report section `## Reproduction` (after the gate-view predictions):
  - one row per variant: agents passing, reproduced k/n, `ast5` smaller / equal / larger than the agent;
  - the runs not reproduced, with both node counts and the reason (`ast5 fails`, `larger`, `not the family`);
  - when `before_distill.json` has `reproduced`, a line "before the distilled changes: k/n".

- [ ] **Step 1: Write the failing tests**
  - `test_reproduced_rules`:
    - `reproduced(True, True, 41, 55, False, False)` is False (larger);
    - `reproduced(True, True, 43, 35, True, False)` is False (not the family);
    - `reproduced(True, True, 43, 35, False, False)` is True;
    - `reproduced(True, False, 43, 10, True, True)` is False.
  - `test_analyse_run_reproduction_fields`: a run with `agent_1.py` and `results.json` rows for `agent_1` and `ast5`
    has all five keys; the same run listed in `agent_excluded.json` has `reproduced` None.
  - `test_report_reproduction_section`: the section counts only rows with `reproduced` not None, and lists a larger
    run with both node counts.
- [ ] **Step 2: Run them; expect FAIL** (`reproduced` not defined, no section):
  `uv run pytest tests/test_irrational_analysis.py tests/test_irrational_report.py -q`
- [ ] **Step 3: Implement the measure.** Add `reproduced` and the keys in `analyse_run`; the excluded list is read
  from `<base>/agent_excluded.json` (base = the run's `runs_irr` root, `rd.parent.parent.parent`). Add
  `_reproduction_md(dev, base)` to the report.
- [ ] **Step 4: Run the tests; expect PASS.**
- [ ] **Step 5: Commit (code):** `git commit -m "irrational report: reproduction of the agents' results by ast5"`.
- [ ] **Step 6: Data.**
  - Write `runs_irr/agent_excluded.json`: `{"runs": ["relu_wide/bump/seed1"], "why": "its agent saw run paths in a
    process listing"}`.
  - Run `uv run nndc irr-analyse <run>` on all 108 development nets.
  - Save `runs_irr/before_distill.json`, with a `note` naming the code commit and, per run, `formula_found`,
    `ast5_nodes`, `ast5_pass`, `agent_nodes` and `reproduced`.
  - Run `uv run nndc irr-report`.

  Expected: reproduced about 90/107. (The scratch count gave 90/108 by size alone; the family rule and the exclusion
  move it.)
- [ ] **Step 7: Commit (data):** `git commit -m "data: irrational reproduction before the distilled changes"`.

### Task 2: Border-aware 2-D fit sample and margin retry (change 1)

**Files:** Modify `nndc/heuristics/families.py`, `tests/test_heuristics_families.py` (the `max_err_check` helper and
the `check=lambda s: True` calls)

**Interfaces:**
- Produces: the `check` hook of `family_candidates` returns `tuple[bool, np.ndarray]`: the pass verdict and the
  counterexample inputs, shape (k, n_inputs), possibly empty. The default builds them from
  `check_program(...).counterexamples` (`c["inputs"][name]` for each of `spec.arg_names`).
- Produces: `_fit_sample` 2-D order: `N_FIT_2D` uniform points, then `4 × N_BORDER` border points, then the
  `N_EXTREME` extreme points. `_snap_sample` takes `X[:N_SNAP_2D]` and everything from `X[N_FIT_2D:]` on.
- Produces: an info row gains `retries` (int) and `margin` (the last tolerance scale used, 1.0 when none).

- [ ] **Step 1: Write the failing tests**
  - `test_fit_sample_has_border_and_corners`: `_fit_sample(spec_2d(-3, 3), f)` has
    `N_FIT_2D + 4 * N_BORDER + N_EXTREME` rows and contains the 4 corners.
  - `test_margin_retry_after_checker_failure`: a stub check fails the first program it sees and returns a
    counterexample at a corner, then passes. The row has `retries == 1`, `margin == 0.8` and `passed`.
  - `test_no_retry_without_counterexamples` (Review Focus 4): the check fails with an empty array. `retries == 0` and
    nothing raises.
  - `test_at_most_two_retries`: the check always fails with a counterexample, so the row has `retries == 2`.
- [ ] **Step 2: Run them; expect FAIL** (`N_BORDER` not defined, no `retries`):
  `uv run pytest tests/test_heuristics_families.py -q -k "border or retry or retries"`
- [ ] **Step 3: Implement.**
  - The border: `np.linspace` on each edge.
  - The retry, inside the term loop of `family_candidates`:
    1. Append the counterexamples (and `predict` of them) to both X and Xs; R stays.
    2. Run `_lawson` on the pre-snap model with `LAWSON_ROUNDS`.
    3. `_snap` at `scale · tol`, then check again.
- [ ] **Step 4: Run the whole file; expect PASS** (about 10 minutes):
  `uv run pytest tests/test_heuristics_families.py tests/test_ast_tool.py -q`
- [ ] **Step 5: Commit:** `git commit -m "ast5 families: border points in the 2-D fit sample, margin retry on checker failures"`.

### Task 3: Clean conic snap (change 4)

**Files:** Modify `nndc/heuristics/conic.py`; Test `tests/test_heuristics_conic.py`

**Interfaces:**
- Produces: `_clean(w: np.ndarray, P: np.ndarray) -> np.ndarray`: the clean coefficient vector, or `w` itself when
  nothing changes.
- Produces: the info row gains `clean: {"src": str, "passed": bool} | None`.

- [ ] **Step 1: Write the failing tests**
  - `test_clean_conic_drops_small_terms`:
    - the decision is `x0² + x1² − 0.01·x0 + 0.01·x1 < 3.977` on [−3, 3]²;
    - the returned src is `"def net(x0, x1):\n    return x0 * x0 + x1 * x1 < 4\n"`;
    - `info["clean"]["passed"]` is True.
  - `test_clean_conic_keeps_direction_and_sign` (Review Focus 3): for a `>` conic with a negative threshold, the
    clean src keeps `>` and a negative threshold.
  - `test_clean_conic_falls_back`: the stub check passes only the snapped src, so the snapped src is returned and
    `info["clean"]["passed"]` is False.
- [ ] **Step 2: Run them; expect FAIL:** `uv run pytest tests/test_heuristics_conic.py -q`
- [ ] **Step 3: Implement `_clean`** and call it after the snap in `conic_candidate` (the boundary points `P` are
  already there).
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit:** `git commit -m "ast5 conic: clean variant (small terms dropped, nice threshold) kept when it passes"`.

### Task 4: Prune route with a minimax output refit (change 2)

**Files:** Create `nndc/heuristics/prune.py`, `tests/test_heuristics_prune.py`; Modify `nndc/heuristics/sigmoid.py`
(`minimax` gains `method: str = "highs"`), `nndc/ast_tool.py`

**Interfaces:**
- Consumes: `sigmoid.minimax(A, y, method="highs") -> (coef, max_err)`; `max_err` is inf on failure.
- Produces: `prune_candidate(spec: Spec, mlp: MLP, check: Callable[[str], tuple[bool, np.ndarray]] | None = None)
  -> tuple[str | None, dict]`. The info dict has:
  - `kept` (unit indices) and `folded` (unit indices; empty until Task 5);
  - `err` (share of the range on the fit points) and `retries`;
  - `note` (str or None).
- Produces: the program is emitted with `program_from_expr(expr, spec, derived)`:
  - one derived line per kept unit, `h<j> = sigmoid(...)` or `relu(...)`, `j` its original index;
  - the return is a `join_terms` sum of `h<j>`, the affine atoms and the constant.
- Produces: in `refactor` (version `ast5`, regression), after the near formula is decided, `add("prune-lp", ...,
  ["F3"])` when no family or conic candidate passes; `extra["prune_lp"]` is the info dict.

- [ ] **Step 1: Write the failing tests**
  - `test_prune_drops_redundant_units`: a 1-D sigmoid net of 6 units where 3 have output weight 1e-4. The program
    has at most 3 `sigmoid(` calls, and a 50k-point dense check passes at 2%.
  - `test_prune_keeps_everything_when_needed` (Review Focus 1): a net of 2 strong, distinct units gives a program
    with 2 units, not None.
  - `test_prune_lp_failure_returns_none` (Review Focus 2): with `sigmoid.minimax` patched to return inf, the src is
    None and `info["note"]` mentions the LP.
  - `test_prune_not_for_binary_or_two_layers`: binary and two-hidden-layer nets give `(None, info)`.
  - `test_minimax_default_unchanged`: `minimax(A, y)` equals `minimax(A, y, method="highs")` on a fixed random A.
  - `test_refactor_runs_prune_only_without_formula` (in `tests/test_ast_tool.py`): with the family stubbed to fail,
    the trace has `prune_lp`; with a passing family, it does not.
- [ ] **Step 2: Run them; expect FAIL:** `uv run pytest tests/test_heuristics_prune.py -q`
- [ ] **Step 3: Implement `prune.py`:**
  1. Grid points and the net's hidden activations on them.
  2. Greedy elimination: least-squares screening, the best `N_SCREEN` confirmed by LP, stop at the budget.
  3. Hidden-weight snapping, one parameter at a time, coarsest level first, LP re-solve after each; then
     output-weight snapping with the snapped columns moved to the right-hand side.
  4. Emit, check, and repeat with counterexamples (`PRUNE_RETRIES`).
- [ ] **Step 4: Run; expect PASS:** `uv run pytest tests/test_heuristics_prune.py tests/test_ast_tool.py -q`
- [ ] **Step 5: Timing check:**
  - run `refactor(..., version="ast5")` in memory on tanh/dist/seed1 and relu_wide/bump/seed0 (the largest:
    64 units);
  - record the prune route's wall time in the ledger;
  - if it takes more than 10 minutes, set `N_SCREEN = 1` and record the change as a ruling.
- [ ] **Step 6: Commit:** `git commit -m "ast5: prune-lp route, greedy unit removal with a minimax output refit, when no formula passes"`.

### Task 5: Joint fold of near-linear units (change 3)

**Files:** Modify `nndc/heuristics/prune.py`; Test `tests/test_heuristics_prune.py`

**Interfaces:**
- Produces: `_foldable(mlp: MLP, spec: Spec) -> list[int]`.
- Produces: `info["folded"]`; the LP columns gain the quadratic atoms when a sigmoid unit folded.

- [ ] **Step 1: Write the failing tests**
  - `test_fold_shallow_sigmoid_units`: on [−3, 3]², a net of 4 units with |w| = 0.1 (range 0.6 at most) and 2 steep
    units. `info["folded"]` has the 4 shallow units, and the program has 2 `sigmoid(` calls.
  - `test_fold_relu_exact` (Review Focus 5): a ReLU unit always on over the box is folded, and the program stays
    exact (max error < 1e-6). A unit that changes sign in the box is not folded.
- [ ] **Step 2: Run; expect FAIL** (no fold).
- [ ] **Step 3: Implement `_foldable`** and fold before elimination.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit:** `git commit -m "ast5 prune-lp: joint fold of near-linear units into affine and quadratic terms"`.

### Task 6: Anti-goal checks and the suite

- [ ] **Step 1:** `uv run python experiments/equivalence.py`. Expected: 0 of 248 differ.
- [ ] **Step 2:** Regenerate `ast4` on a copy of 6 runs in the scratchpad (one per variant, plus a sigmoid and a
  tanh run from `runs_sigmoid/` and `runs_tanh/`). Expected: byte-identical `ast4.py`.
- [ ] **Step 3:** Run the full suite in the background into the workspace (about 30 minutes):
  `uv run pytest -q > <workspace>/full_suite_distill.out`. Expected: all pass.

### Task 7: Development rerun and the re-reported numbers

- [ ] **Step 1:** Make sure no blind agent is running, then run the batch, 5 at a time, detached, with neutral
  arguments. For each of the 108 development nets: `nndc ast <run> --name ast5`, `nndc explain <run>`,
  `nndc irr-analyse <run>`.
- [ ] **Step 2:** `evaluate_run` over all runs, then `uv run nndc irr-report`.
- [ ] **Step 3: Check the amendment's expectations and record the result:**
  - sigmoid/dist 0–1 pass a formula (change 1);
  - the prune route is smaller than the first run's `ast5` on tanh/dist 0–2 and sigmoid/dist, and on sigmoid/bump;
  - the 4 circles give `x0 * x0 + x1 * x1 < 4` or close to it;
  - reproduction rises from the Task 1 number.

  Each hit or miss goes into a new amendment, "2026-10-31, development numbers after the distilled changes", in the
  irrational design. The predictions are not rewritten.
- [ ] **Step 4: Commit the data:** `git commit -m "data: irrational development nets after the distilled route changes"`.
  Commit the amendment separately as `docs:`.
