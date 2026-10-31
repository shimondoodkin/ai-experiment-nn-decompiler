# Irrational Functions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** irrational-function tasks (exp, log, sqrt, circle, …) on ReLU, wide ReLU, sigmoid and tanh nets; tool version
`ast5` (= `ast4` plus a formula route) that snaps nets to a formula plus correction terms; a report on how the nets
model the functions and how often the formula is found, with a frozen tool on fresh held-out nets.

**Architecture:** the language gains `exp`/`log`/`sqrt` and division by an expression. The tasks live in
`experiments/irrational.py`; runs go to four roots under `runs_irr/`. The formula route is two heuristics modules
(`families.py` for regression, `conic.py` for binary) wired into `refactor` for `version="ast5"`. Task-aware analysis
(report only) is `experiments/irrational_analysis.py`, read by `nndc/irrational_report.py`.

**Tech Stack:** Python 3.13, NumPy, PyTorch (training), SciPy `least_squares` (route fits; already in the default
`baselines` group), PySR (`--group pysr`), matplotlib (new `plots` group, plots only).

**Spec:** `docs/design/2026-10-30-irrational-functions-design.md`

## Global Constraints

- `nndc/lang.py`, `nndc/spec.py`, `nndc/model.py`, `nndc/check.py` stay stdlib + NumPy (they run in blind folders).
- The tool never sees a task name, `expected` formula, target or task family: `refactor` gets the blind spec, the dump
  and the weights only. Family columns live in `experiments/`, which `nndc/heuristics/` never imports.
- `ast4` programs regenerate byte-identically; `experiments/equivalence.py compare` shows 0 differences on the ReLU
  roots; no older version's verdict changes. The old PySR path (`pysr.py`, `pysr_snap.py`) is unchanged.
- Snap levels `(0.1, 0.05, 0.02, 0.01, 0.005, 0.001)`, relative to `max(1, |p|)`. Fit-drop threshold: 25% of the net's
  output range. Correction terms: at most 4, knots from a 97-point grid over u's range. Tolerance = the checker's
  (`spec.snap_tol_frac` × the net's output range).
- Fit sample: 1-D a 2001-point grid; 2-D 20 000 uniform points (seed 0) plus the 200 points where the net is most
  extreme (by |output − median|) out of those.
- Data commits (`data:` prefix) apart from code commits. No attribution lines in commit messages.
- No destructive commands on the repo or `runs*/`; scratch copies go to the session scratchpad; tests that delete use
  `tmp_path`.
- Held-out nets are built only after the freeze commit, and run once.

## Rulings on the design (made while planning)

- **2-D family names.** In 2-D the inner variable u is already a general quadratic, so "square" would be degree 4.
  The 2-D families are `quadratic` (a·u + d, with a = 1, d = 0 fixed), `sqrt`, `exp`, `log`, `reciprocal`. The 1-D
  families are `identity`, `square`, `exp`, `log`, `sqrt`, `reciprocal`. Task families for the report: `product` →
  `quadratic` (the design's "square, whose inner quadratic is x0·x1"), `circle` and `ellipse` → `conic`. Only 1-D
  `identity` counts as a hinge program. The design gets a one-line amendment.
- **"regions refuses programs".** `regions` takes nets, not programs, and already refuses non-ReLU nets; nothing
  changes there. Only `compile_back` gains a refusal.
- **Analysis is its own step.** The report must not re-run fits and checks, so `nndc irr-analyse` writes
  `irr_analysis.json` per run (recorded as step `irr_analysis` in `run_git.txt`), and `nndc irr-report` only reads.

## Review Focus

1. **A wide build over an existing non-wide folder** (or the reverse) must refuse, as `act` mismatches do; otherwise
   a wide net silently overwrites a small one's results. Test in Task 2.
2. **Families that go non-finite on part of the box** (log or sqrt of a negative u, 1/u through zero) must be dropped
   or fail the check, never crash the route or `refactor`. Test in Task 3.
3. **A constant or near-constant net** (an unlearned ReLU net, output range ≈ 0) must make the route return nothing,
   not divide by a zero range. Test in Task 3.
4. **A binary net with no boundary in the box** (all one class) must make the conic route return None. Test in Task 4.
5. **Held-out task names in the default build set:** `nndc run irrational` must build only the 9 development tasks;
   the held-out tasks need `irrational-heldout`. Test in Task 2.

---

### Task 1: Language: exp, log, sqrt and division by an expression

**Files:**
- Modify: `nndc/lang.py` (`FUNCTIONS`, `allowlist_summary`, `_check_expr`'s Div rule, the call evaluator near the
  `sigmoid`/`tanh` branch)
- Modify: `nndc/compile_back.py:344` (the sigmoid/tanh refusal)
- Test: `tests/test_lang.py`, `tests/test_check.py`, `tests/test_compile_back.py`

**Interfaces:**
- Produces: programs may call `exp(x)`, `log(x)`, `sqrt(x)` (exactly one argument each) and divide by any
  expression. Each call is one node in `program_size`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lang.py
def test_exp_log_sqrt_evaluate():
    p = parse_program("def net(x0):\n    return exp(x0) + log(x0) + sqrt(x0)\n", ["x0"])
    X = np.array([[1.0], [4.0]])
    assert np.allclose(evaluate(p, X), np.exp(X[:, 0]) + np.log(X[:, 0]) + np.sqrt(X[:, 0]))

def test_division_by_expression():
    p = parse_program("def net(x0, x1):\n    return x0 / (x1 + 1)\n", ["x0", "x1"])
    assert np.allclose(evaluate(p, np.array([[3.0, 2.0]])), [1.0])

def test_new_functions_take_one_argument():
    with pytest.raises(ProgramError, match="exactly one argument"):
        parse_program("def net(x0):\n    return exp(x0, x0)\n", ["x0"])

def test_out_of_domain_gives_nonfinite_without_warning():
    p = parse_program("def net(x0):\n    return log(x0) + sqrt(x0) + 1 / x0\n", ["x0"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = evaluate(p, np.array([[-1.0], [0.0]]))
    assert not np.isfinite(out).any()

def test_new_function_counts_one_node():
    assert program_size(parse_program("def net(x0):\n    return exp(x0)\n"))["nodes"] == \
           program_size(parse_program("def net(x0):\n    return relu(x0)\n"))["nodes"]

def test_allowlist_mentions_new_functions():
    s = allowlist_summary()
    assert "exp(x)" in s and "log(x)" in s and "sqrt(x)" in s and "division by a nonzero number constant" not in s

# tests/test_check.py
def test_nonfinite_program_output_fails_regression():
    # a 1-input regression net (any helper net in tests/helpers.py); program log(x0 - 100) is NaN everywhere
    r = check_program("def net(x0):\n    return log(x0 - 100)\n", mlp, spec)
    assert not r.snap_pass

# tests/test_compile_back.py
def test_compile_back_refuses_exp():
    with pytest.raises(CompileError, match="cannot be compiled into a ReLU net"):
        compile_back("def net(x0):\n    return exp(x0)\n", spec)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_lang.py tests/test_check.py tests/test_compile_back.py -q`
Expected: the new tests FAIL (`function exp is not allowed`, `division is allowed only by a nonzero number constant`);
`test_nonfinite_program_output_fails_regression` may already pass (the checker treats non-finite as error) — keep it
as a pin.

- [ ] **Step 3: Implement**

- `FUNCTIONS` gains `exp`, `log`, `sqrt`; the one-argument rule covers them; evaluate with `np.exp`/`np.log`/`np.sqrt`
  inside the existing `np.errstate(all="ignore")`.
- The Div rule goes; `ast.Div` stays in `_BINOPS` (`np.divide`).
- `allowlist_summary`: "+ - * /", and the functions list adds `exp(x), log(x), sqrt(x)`.
- `compile_back`: refuse `exp`/`log`/`sqrt` with the sigmoid/tanh message (`"compile_back: sigmoid/tanh/exp/log/sqrt
  cannot be compiled into a ReLU net"`) and division by a non-constant with `"division by a non-constant"`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. If an existing test asserts the old allowlist text or the old division error, update it to the
new wording (the language change is deliberate) and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add nndc/lang.py nndc/compile_back.py tests/test_lang.py tests/test_check.py tests/test_compile_back.py
git commit -m "lang: exp, log, sqrt and division by an expression; compile_back refuses them"
```

### Task 2: Tasks, wide nets and building

**Files:**
- Create: `experiments/irrational.py`
- Modify: `experiments/tasks.py:103` (`get_task`), `nndc/run.py:106-159` (`build_run`), `nndc/cli.py` (`cmd_run`,
  the `run` parser)
- Create: `experiments/spikes/irrational/` (copy of the scratchpad `spike.py`, `summary.py` and a README with the
  spike table from the design; the `out/` folder stays out)
- Test: `tests/test_tasks.py`, `tests/test_run.py`

**Interfaces:**
- Produces:
  - `IRRATIONAL_TASKS: dict[str, Task]` (the 9 development tasks, `seeds=(0, 1, 2)`, `steps=6000`, hidden `[8]` for
    1 input and `[16]` for 2) and `IRRATIONAL_HELDOUT_TASKS: dict[str, Task]` (`decay`, `logshift`, `ellipse`,
    `hypot1`, `seeds=(0, 1)`), with the specs, targets and `expected` strings of the design's tables.
  - `FAMILY: dict[str, str]`, task name → family (`exp`, `log`, `sqrt`, `square`, `reciprocal`, `quadratic`,
    `conic`) for all 13 tasks.
  - `widened(task: Task) -> Task`: 4× every hidden size, 3× steps.
  - `build_run(task, seed, root, retrain=False, bug=False, act="relu", wide=False) -> Path`.
  - `nndc run irrational|irrational-heldout|<task> [--seed K] --act A [--wide] --root R`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tasks.py
def test_irrational_tasks_resolve_and_stay_out_of_TASKS():
    from experiments.irrational import IRRATIONAL_TASKS, IRRATIONAL_HELDOUT_TASKS, FAMILY
    assert set(IRRATIONAL_TASKS) == {"exp", "log", "sqrt", "square", "recip", "circle", "dist", "bump", "product"}
    assert set(IRRATIONAL_HELDOUT_TASKS) == {"decay", "logshift", "ellipse", "hypot1"}
    assert not (set(IRRATIONAL_TASKS) | set(IRRATIONAL_HELDOUT_TASKS)) & set(TASKS)
    assert set(FAMILY) == set(IRRATIONAL_TASKS) | set(IRRATIONAL_HELDOUT_TASKS)
    assert get_task("ellipse").spec.output == "binary"

def test_irrational_expected_matches_target():
    # expected_program evaluated with nndc.lang equals task.target on 1000 uniform points (binary: as decisions)
    for t in [*IRRATIONAL_TASKS.values(), *IRRATIONAL_HELDOUT_TASKS.values()]:
        ...  # max abs diff <= 1e-9, or identical decisions

def test_widened():
    t = widened(get_task("exp"))
    assert t.hidden == [32] and t.steps == 18000 and get_task("exp").hidden == [8]

# tests/test_run.py  (use tmp_path roots; patch train to a fast stub as the existing build tests do)
def test_irrational_run_gets_formula_py_for_relu(tmp_path): ...   # build_run("exp", 0, tmp_path) writes formula.py
def test_step1_relu_run_still_has_no_formula_py(tmp_path): ...     # build_run("absdiff", 0, tmp_path): no formula.py
def test_wide_build_refuses_small_folder(tmp_path):
    build_run("exp", 0, tmp_path)
    with pytest.raises(ValueError, match="wide"):
        build_run("exp", 0, tmp_path, wide=True)

def test_run_irrational_set_excludes_heldout(): ...   # the name list cmd_run builds for "irrational"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_tasks.py tests/test_run.py -q`
Expected: FAIL on the import of `experiments.irrational`.

- [ ] **Step 3: Implement**

- `get_task` resolves `TASKS`, then `DATASET_TASKS`, then the two irrational dicts (lazy import, as for datasets).
- `build_run(wide=True)` trains `widened(t)`; the train log's existing `hidden`/`steps` fields record it. Refuse when
  the folder's `train_log.json` `hidden` differs from the requested task's (message names `wide`).
- `formula.py` is written when `(non-ReLU and task in TASKS)` (unchanged) **or** the task is irrational, for every
  activation.
- `cmd_run`: the names `irrational` and `irrational-heldout` expand to the dicts' keys; `--wide` is passed through.
- The circle and ellipse specs use the same threshold and 1/0 convention as the existing binary tasks.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Smoke build** (scratch root, not the repo)

Run: `uv run nndc run exp --seed 0 --act sigmoid --root <scratchpad>/irr_smoke`
Expected: a `PASS exact` dump line; the folder has `formula.py`.

- [ ] **Step 6: Commit**

```bash
git add experiments/irrational.py experiments/tasks.py nndc/run.py nndc/cli.py experiments/spikes/irrational tests/test_tasks.py tests/test_run.py
git commit -m "irrational tasks (9 development, 4 held-out), wide nets, formula.py for every irrational run"
```

### Task 3: The family route (regression)

**Files:**
- Create: `nndc/heuristics/families.py`
- Test: `tests/test_heuristics_families.py`

**Interfaces:**
- Consumes: `nndc.heuristics.snap.nice`, `nndc.check.check_program`, `nndc.heuristics.cleanup.join_terms` (or the
  module's own formatter; programs must drop `1*`, `+ 0` and zero terms).
- Produces:
  - `SNAP_LEVELS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.001)`, `MAX_TERMS = 4`, `KNOTS = 97`, `DROP_FRAC = 0.25`.
  - `family_candidates(spec: Spec, mlp: MLP, predict=None, check=None) -> tuple[list[tuple[str, str]], list[dict]]`
    (`predict` and `check` are test hooks, see Step 1): the passing programs as
    `(family, src)`, and one info dict per family tried:
    `{"family", "fit_err", "snapped_err", "terms", "passed", "nodes", "src", "params"}` (errors as fractions of the
    net's output range; `src`/`nodes` of the last program tried; `passed` by `check_program(src, mlp, spec)`).
    Returns `([], [])` when the spec is not regression, has more than 2 inputs, or the net's output range is 0.

- [ ] **Step 1: Write the failing tests**

Build tiny "nets" for the tests with a helper that trains nothing: an `MLP` whose output equals a known function is
not available, so test against the formula itself by patching the route's `predict` hook. `family_candidates` takes
an optional `predict: Callable[[np.ndarray], np.ndarray] | None = None` (default: the net's forward pass) and
`check: Callable[[str], bool] | None = None` (default: `check_program(...).snap_pass`), used only by tests.

```python
def test_exp_net_snaps_to_exp():
    spec = blind_spec_1d(-2, 2)                     # helper: blind regression spec, one input
    progs, info = family_candidates(spec, None, predict=lambda X: np.exp(X[:, 0]), check=max_err_check(np.exp, 0.02))
    assert ("exp", "def net(x0):\n    return exp(x0)\n") in progs

def test_shifted_log_snaps_with_offset():
    progs, _ = family_candidates(blind_spec_1d(-1.5, 4), None, predict=lambda X: np.log(X[:, 0] + 2), check=...)
    assert any(f == "log" and "log(x0 + 2)" in s for f, s in progs)

def test_correction_term_added_when_flattened():
    f = lambda x: np.sqrt(x) - 0.08 * np.maximum(0, x - 3)       # sqrt with an edge flattening
    progs, info = family_candidates(blind_spec_1d(0, 4), None, predict=lambda X: f(X[:, 0]), check=...)
    row = next(r for r in info if r["family"] == "sqrt")
    assert row["passed"] and 1 <= row["terms"] <= MAX_TERMS and "relu(" in row["src"]

def test_2d_radial_exp_uses_general_quadratic():
    spec = blind_spec_2d(-3, 3)
    g = lambda X: np.exp(-((X[:, 0] - 0.5) ** 2 + X[:, 1] ** 2))
    progs, _ = family_candidates(spec, None, predict=g, check=...)
    assert any(fam == "exp" for fam, _ in progs)

def test_nonfinite_family_is_dropped_not_raised():
    # a net that is negative on part of the box: log/sqrt/reciprocal fits go NaN there; no exception
    progs, info = family_candidates(blind_spec_1d(-2, 2), None, predict=lambda X: X[:, 0] ** 3, check=...)
    assert all(np.isfinite(r["fit_err"]) or not r["passed"] for r in info)

def test_constant_net_returns_nothing():
    assert family_candidates(blind_spec_1d(0, 1), None, predict=lambda X: np.full(len(X), 0.3), check=...) == ([], [])

def test_not_applicable_to_binary_or_three_inputs(): ...   # ([], []) for a binary spec and a 3-input spec
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_heuristics_families.py -q`
Expected: FAIL on the import.

- [ ] **Step 3: Implement `family_candidates`**

Per family (1-D: `identity`, `square`, `exp`, `log`, `sqrt`, `reciprocal` with u = b·x0 + c; 2-D: `quadratic`,
`sqrt`, `exp`, `log`, `reciprocal` with u = p0·x0² + p1·x1² + p2·x0·x1 + p3·x0 + p4·x1 + p5; the model is a·f(u) + d,
and `quadratic` fixes a = 1, d = 0):
1. Fit by `scipy.optimize.least_squares(method="lm")` on the fit sample, non-finite residuals replaced by a large
   constant, several starting points (the starts are the builder's; the 2-D starts may come from the linear
   least-squares `quadratic` fit). Keep the start with the lowest max error. Drop the family if that error is over
   `DROP_FRAC` of the range.
2. Snap all parameters with the spike's sequential snap (`spike.py: snap_params`): each round, the free parameter
   that snaps at the coarsest level, refit of the other free ones, accept a level only if the max error on the fit
   sample stays ≤ the tolerance. A parameter that never snaps stays as fitted (rounded to 4 significant digits).
3. Emit and check. If it fails, add hinge terms greedily in u (`k·relu(u − t)` or `k·relu(t − u)`, t over `KNOTS`
   points across u's range on the fit sample, k by least squares together with a and d), snap again over all
   parameters, emit, check; up to `MAX_TERMS` terms, stopping at the first pass.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_heuristics_families.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/heuristics/families.py tests/test_heuristics_families.py
git commit -m "heuristics: family route (fit a·f(u)+d to the net, snap, add hinge terms in u)"
```

### Task 4: The conic route (binary, 2 inputs)

**Files:**
- Create: `nndc/heuristics/conic.py`
- Test: `tests/test_heuristics_conic.py`

**Interfaces:**
- Produces: `conic_candidate(spec: Spec, mlp: MLP, decide=None, check=None) -> tuple[str | None, dict]`. `decide`
  (tests only) maps inputs to 0/1 decisions, default the net's decision; `check` as in Task 3. The info dict:
  `{"family": "conic", "n_boundary", "coefs", "snapped", "passed", "nodes", "src"}`. Returns `(None, info)` when the
  spec is not binary with 2 inputs, or fewer than 20 boundary points are found.

- [ ] **Step 1: Write the failing tests**

```python
def test_circle_snaps():
    src, info = conic_candidate(blind_binary_2d(-3, 3), None,
                                decide=lambda X: (X[:, 0]**2 + X[:, 1]**2 < 4).astype(int), check=decision_check(...))
    assert info["passed"] and "x0*x0 + x1*x1 < 4" in src.replace("(", "").replace(")", "")

def test_offcentre_ellipse_snaps():
    d = lambda X: ((X[:, 0] - 0.5)**2 / 4 + X[:, 1]**2 < 1).astype(int)
    src, info = conic_candidate(blind_binary_2d(-3, 3), None, decide=d, check=...)
    assert info["passed"]

def test_no_boundary_returns_none():
    src, info = conic_candidate(blind_binary_2d(-3, 3), None, decide=lambda X: np.ones(len(X), int), check=...)
    assert src is None

def test_side_is_taken_from_the_net(): ...   # outside-is-1 version of the circle gives `> 4`
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_heuristics_conic.py -q`
Expected: FAIL on the import.

- [ ] **Step 3: Implement**

- Boundary points: 20 000 uniform points (seed 0); pair each with a random partner; for pairs whose decisions
  differ, 30 bisection steps on the segment. Cap at 2000 points.
- Fit: the smallest right-singular vector of `[x0², x1², x0·x1, x0, x1, 1]` over the boundary points; scale so the
  quadratic coefficient of largest |value| is 1.
- Snap: coefficients one at a time (coarsest level first, as Task 3), refit of the free ones by linear least squares
  with the snapped ones fixed; a snap is kept only if the checker passes (binary `snap_pass`).
- Emit `q(x) < c` or `q(x) > c` (constant moved to the right), the side chosen by the net's majority decision where
  `q(x) < c`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_heuristics_conic.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/heuristics/conic.py tests/test_heuristics_conic.py
git commit -m "heuristics: conic route for 2-input binary nets"
```

### Task 5: Tool version `ast5`

**Files:**
- Modify: `nndc/ast_tool.py` (`ROUTES`, `TOOL_VERSIONS`, `refactor`), `nndc/run.py:166` (`OWN_CEX`),
  `nndc/cli.py:271` (`--name` choices), `nndc/report.py:138` (`_tool`'s order)
- Test: `tests/test_ast_tool.py`, `tests/test_run.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `family_candidates` (Task 3), `conic_candidate` (Task 4).
- Produces: `ROUTES` ends with `"family", "conic"`; `TOOL_VERSIONS = ("ast", "ast4", "ast5")`; `OWN_CEX = ("ast4",
  "ast5")`; `refactor(..., version="ast5")` keeps every `ast4` route (kink merge, tanh rewrite) and adds each passing
  family program and the conic program as candidates; `trace["families"]` is the list of family info dicts (plus the
  conic dict for binary nets). `nndc ast --name ast5`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ast5_adds_family_candidates(): ...    # refactor on a sigmoid exp-like test net with version="ast5":
                                               # trace has "families"; a "family" candidate exists
def test_ast4_ignores_family_route(): ...      # same net, version="ast4": no "family" candidate, no "families" key
def test_run_ast_ast5_writes_own_files(tmp_path): ...   # ast5.py, ast5_trace.json; ast4.py untouched;
                                                         # results.json has "ast5"; record_git line "ast5"
def test_report_tool_prefers_ast5(): ...        # _tool({"ast4": ..., "ast5": ...}) picks ast5
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_ast_tool.py tests/test_run.py tests/test_report.py -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

In `refactor`: `kinks = version in ("ast4", "ast5")`; `formula = version == "ast5"`. The family and conic routes run
on the routes' net (`mlp` after the tanh rewrite) inside `add()`, so an error skips only that route; every candidate is
still checked against `check_mlp`. Ranking is unchanged (fewest nodes, then exactness, then `ROUTES` order).

- [ ] **Step 4: Equivalence and the full suite**

Run:
- `uv run pytest -q` → all pass.
- On scratch copies of 4 runs (one each of `runs/absdiff/seed0`, `runs_sigmoid/neareq/seed1`,
  `runs_sigmoid/iris/seed0`, `runs_tanh/absdiff/seed0`), `run_ast(rd, "ast4")`: `ast4.py` byte-identical to the repo's.
- `uv run python -m experiments.equivalence copy --copy <scratch>/eq`, then `evaluate` and `compare` → 0 differences.

- [ ] **Step 5: Commit**

```bash
git add nndc/ast_tool.py nndc/run.py nndc/cli.py nndc/report.py tests/test_ast_tool.py tests/test_run.py tests/test_report.py
git commit -m "ast5: ast4 plus the family and conic routes (nndc ast --name ast5)"
```

### Task 6: PySR with exp, log, sqrt and division

**Files:**
- Modify: `nndc/baselines/pysr_run.py`, `nndc/baselines/common.py:18` (`BASELINE_VERSIONS`), `nndc/cli.py`
  (`_baseline_fns`, the `baseline` choices)
- Test: `tests/test_baselines_pysr.py`

**Interfaces:**
- Produces: `PYSR_IRR_SETTINGS` = `PYSR_SETTINGS` with `binary_operators` + `"/"` and `unary_operators` +
  `"exp", "log", "sqrt"`, nothing else changed. `run_pysr(rd, name="pysr", settings=PYSR_SETTINGS, irr=False)`;
  `run_pysr_irr(rd)` = `run_pysr(rd, "pysr_irr", PYSR_IRR_SETTINGS, irr=True)`, writing `pysr_irr.py`,
  `pysr_irr_snap.py`, `pysr_irr_info.json`. `BASELINE_VERSIONS` += `"pysr_irr", "pysr_irr_snap"`.
  `nndc baseline pysr_irr`. `_src(e, irr=False)`: with `irr=True` it emits `exp(...)`, `log(...)`, `sqrt(...)`
  (sympy `Pow(b, 1/2)`) and `a / b` for negative integer powers of non-constants.

- [ ] **Step 1: Write the failing tests** (not `slow`; sympy only)

```python
def test_src_irr_converts_exp_log_sqrt_division():
    x0, x1 = sympy.symbols("x0 x1")
    src = sympy_to_program(sympy.exp(x0) + sympy.log(x1) + sympy.sqrt(x0) + 1 / (x0 + 1), spec2, irr=True)
    # evaluates equal to the sympy expression on a few positive points
def test_src_default_still_refuses_division_by_nonconstant():
    with pytest.raises(UnsupportedExpr):
        sympy_to_program(1 / (x0 + 1), spec2)
def test_irr_settings_only_add_operators():
    assert {k: v for k, v in PYSR_IRR_SETTINGS.items() if "operators" not in k} == \
           {k: v for k, v in PYSR_SETTINGS.items() if "operators" not in k}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_baselines_pysr.py -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement** as in Interfaces. The info file for `pysr_irr` is written with `write_info(rd, "pysr_irr",
  info)` and `record_git(rd, "pysr_irr")`.

- [ ] **Step 4: Run the suite, then one real fit**

Run: `uv run pytest -q`, then `uv run --group pysr nndc baseline pysr_irr <scratch smoke run from Task 2>`.
Expected: all pass; the smoke run gets a `pysr_irr` line (any verdict).

- [ ] **Step 5: Commit**

```bash
git add nndc/baselines/pysr_run.py nndc/baselines/common.py nndc/cli.py tests/test_baselines_pysr.py
git commit -m "baselines: pysr_irr (step-4 settings plus exp, log, sqrt and /)"
```

### Task 7: Analysis, report and plots

**Files:**
- Create: `experiments/irrational_analysis.py`, `nndc/irrational_report.py`, `experiments/irrational_plots.py`
- Modify: `nndc/cli.py` (`irr-analyse`, `irr-report`), `pyproject.toml` (`plots = ["matplotlib>=3.9"]`)
- Test: `tests/test_irrational_analysis.py`, `tests/test_irrational_report.py`

**Interfaces:**
- Consumes: `FAMILY` (Task 2), `ast5_trace.json["families"]` (Task 5), `results.json`, `train_log.json`.
- Produces:
  - `analyse_run(rd: Path) -> dict`, written to `rd / "irr_analysis.json"`, with `record_git(rd, "irr_analysis")`.
    Keys: `learned`, `true_vs_net` (max |net − target| / net range on the checker's input set; regression only),
    `family`, `formula_found` (bool), `found_terms`, `other_families` (passing non-task families), `route_pick`
    (the route's smallest passing family), `ast4_nodes`, `ast5_nodes`, `ast5_route`, `formula_nodes`,
    `hinge_only` (`{"hinges", "nodes", "passed"}`, 1-D only), `won_on_size` (formula nodes < both the hinge fit's
    and `ast4`'s; in 2-D `ast4`'s only), `pysr_found` (`pysr_irr_snap` passes and its source uses the family's
    function: `exp`, `log`, `sqrt`, `/` for `reciprocal`, a product of two inputs for `square`, `quadratic` and
    `conic`), plus `model`:
    - 1-D: `kinks` (ReLU kinks −b/w inside the box, stacked within 1% of the range counted once) or `centres`
      (sigmoid/tanh −b/w inside the box), `dead_units`, `optimal_knots` (same count, from the target's
      √|f″| density, numeric second differences), `knot_gap` (mean |kink − optimal| / range, both sorted),
      `max_err_at` (`"edge"` within 5% of either end, else `"interior"`);
    - circle/ellipse: `pieces` (distinct activation patterns of 720 ray-bisected boundary points from the target's
      centre; ReLU only), `conic` (the route's snapped coefficients), `radius_min_max`;
    - 2-D regression: `err_by_radius` (max |net − target| in 10 bins of distance from the target's centre).
  - `hinge_fit(rd) -> dict`: the spike's from-scratch greedy fit (`c0 + c1·x0 + Σ k·relu(x0 − t)`, up to 16 hinges,
    97 knots, each step checked); report only.
  - `build_irrational_report(base: Path, heldout: bool) -> str` and `nndc irr-report --base runs_irr --out
    runs_irr/REPORT.md`; sections per the design ("Results", "How the nets model the functions", "Snapping to
    formulas", "Held-out"), with the prediction lines and a met / not met column. Held-out runs are those named in
    each root's `heldout.json`.
  - `nndc irr-analyse RUNS… | --all --root R`.
  - `experiments/irrational_plots.py --base runs_irr` writes `runs_irr/plots/<task>_<variant>.png` (seed 0).

- [ ] **Step 1: Write the failing tests**

```python
def test_formula_found_needs_task_family(tmp_path): ...
    # fake run folder (task "exp") whose ast5_trace has a passing "sqrt" row only: formula_found False,
    # other_families == ["sqrt"]; with a passing "exp" row with 2 terms: formula_found True, found_terms 2
def test_identity_never_counts(): ...    # passing "identity" row on task "exp": formula_found False
def test_kinks_merge_stacked_units(): ...# ReLU 1-D weights with 3 kinks at 0.090/0.093/0.096 and one at 1.2 → 2 kinks
def test_optimal_knots_for_square_are_even(): ...  # 4 knots on [-2, 2] for x² → about [-1.5, -0.5, 0.5, 1.5] (±0.05)
def test_won_on_size(): ...               # formula 7 nodes, hinge-only 25, ast4 58 → True; hinge-only 6 → False
def test_report_counts_and_predictions(tmp_path): ...  # two fake roots with irr_analysis.json files: the
                                                       # "Snapping to formulas" counts and one met/not-met line
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_irrational_analysis.py tests/test_irrational_report.py -q`
Expected: FAIL on the imports.

- [ ] **Step 3: Implement** as in Interfaces.

- [ ] **Step 4: Run the suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/irrational_analysis.py experiments/irrational_plots.py nndc/irrational_report.py nndc/cli.py pyproject.toml uv.lock tests/test_irrational_analysis.py tests/test_irrational_report.py
git commit -m "irrational analysis (report only), irr-report and plots"
```

### Task 8: First development run

- [ ] **Step 1: Build the 108 nets** (background, one process per variant):

```bash
uv run nndc run irrational --act relu --root runs_irr/relu
uv run nndc run irrational --act relu --wide --root runs_irr/relu_wide
uv run nndc run irrational --act sigmoid --root runs_irr/sigmoid
uv run nndc run irrational --act tanh --root runs_irr/tanh
```

Expected: 27 `PASS exact` dump lines per root.

- [ ] **Step 2: `ast4` and `ast5` on every development run** (`nndc ast --all --split all --root R --name ast4`, then
  `--name ast5`; split into several background processes by task folder, at most 8 at once; 2-D sigmoid and tanh
  nets take about 10 minutes each per version).
- [ ] **Step 3: PySR** (`uv run --group pysr nndc baseline pysr_irr --all --root R`, background; about 2 minutes a
  net).
- [ ] **Step 4: Analysis, report, plots:** `nndc irr-analyse --all --root R` for each root;
  `nndc irr-report --base runs_irr --out runs_irr/REPORT.md`; `uv run --group plots python -m
  experiments.irrational_plots --base runs_irr`. Re-run `evaluate_run` over every run before committing.
- [ ] **Step 5: Record** the first-development-run numbers against each prediction in the design's new "Amendments"
  section (plus the 2-D family-name ruling). Data commit:

```bash
git add runs_irr docs/design/2026-10-30-irrational-functions-design.md
git commit -m "data: irrational development nets (108), ast4, ast5, pysr_irr, analysis; first development run"
```

### Task 9: Development changes (optional), then the freeze

> Amended 2026-10-31 (design, Amendments): a blind agent pass on all 108 development nets comes first; the
> changes here are distilled from `runs_irr/HEURISTICS.md`, and the report gains a reproduction measure. Task 10 adds
> agents on the 68 held-out nets (`runs_irr/HEURISTICS_heldout.md`).

- [ ] **Step 1:** If the first run shows a fixable route problem, change the route using the development nets only.
  Each change is a design amendment with its reason; re-run `ast5` and the analysis on the development runs, re-report.
  Code commits and data commits stay apart. Predictions are not rewritten.
- [ ] **Step 2: Freeze:** a commit with the final tool code (or an empty `git commit --allow-empty -m "freeze: ast5"`
  when nothing changed after Task 8); its hash goes into the design's Amendments and into each root's `heldout.json`.

### Task 10: Held-out nets

- [ ] **Step 1: Build** (after the freeze): seed 3 of the 9 development tasks and seeds 0–1 of the 4 held-out tasks,
  all 4 variants (`nndc run <task> --seed 3 …` per task; `nndc run irrational-heldout …`). Expected: 68 `PASS exact`
  dump lines.
- [ ] **Step 2:** `ast5` (frozen), `ast4`, `pysr_irr`, `irr-analyse` on the 68; each root's `heldout.json`:
  `{"version": "ast5", "freeze": "<hash>", "runs": [...]}`.
- [ ] **Step 3:** Report and plots again; the held-out section shows the met / not-met lines. Re-run `evaluate_run` over
  every run. Data commit:

```bash
git add runs_irr
git commit -m "data: irrational held-out nets (68), frozen ast5, ast4, pysr_irr, analysis"
```

### Task 11: Write-up

- [ ] **Step 1:** `CONCLUSIONS.md` "Irrational functions" section: the two questions, what held and what failed.
- [ ] **Step 2:** `CLAUDE.md` (the irrational runs, `ast5`, `irr-analyse`/`irr-report`, held-out sets) and `README.md`
  (a short highlights section linking `runs_irr/REPORT.md`).
- [ ] **Step 3:** Full suite green; one final whole-branch review; commit (`docs: irrational functions …`).
