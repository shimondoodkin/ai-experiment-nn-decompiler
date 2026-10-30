# Near Formulas, Residuals and the Gate View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** minimax fitting and near formulas in `ast5`'s family route, a blind residual description, the gate view of
one-layer nets, `explain.md` per run, and the irrational development step rerun with them.

**Architecture:** `nndc/heuristics/families.py` gains a Lawson minimax refit and near-formula candidates; `refactor`
records `trace["near"]` when no formula passes. Two new blind modules, `nndc/gate_view.py` (units and groups) and
`nndc/explain.py` (residual description and `explain.md`), behind `nndc explain`. The task-aware analysis and the
irrational report read the new fields.

**Tech Stack:** Python 3.13, NumPy, SciPy `least_squares` (already used by the route).

**Spec:** `docs/design/2026-10-31-formula-residual-gate-view-design.md` (extends
`docs/design/2026-10-30-irrational-functions-design.md`; that project's plan is `docs/plans/2026-10-30-irrational-functions.md`,
Tasks 1–7 done, Task 8 stopped after 18 of 108 nets).

## Global Constraints

- Lawson minimax: 40 rounds for fits and hinge refits; 10 rounds for refits inside snapping, on the snapping subsample.
  Fallback if a 2-D net's family route exceeds 10 minutes: least-squares refits while snapping plus one 40-round
  minimax refit of the free parameters before each program is emitted (recorded as a design amendment).
- Near-formula snapping tolerance: `max(tol, 1.25 × the family's minimax error)`. Near-formula choice: among candidates
  within 3% of the output range, fewest correction terms, then smaller error, then fewer nodes; if none within 3%,
  smallest error. 1-D `identity` is never a near formula; 2-D `quadratic` is. Recorded only when no family or conic
  program passes. Binary: the failing conic, error = `disagreements_outside / n_inputs`.
- Gate view zones (sigmoid z): off < −4, low knee −4..−1.5, middle −1.5..1.5, high knee 1.5..4, on > 4; ReLU: off,
  active. Sample: 100 000 uniform points, seed 0. Tolerance: regression 2% of the net's output range; binary 2% of the
  raw score's range (output − threshold). Stand-ins within a quarter of the tolerance, simplest first: constant, line,
  step, clamp, rounded hinge `k·relu(z − t) + c`.
- Groups, a partition, in order: negligible (magnitude < tol/4), lines (stand-in constant or line), rings (same output
  sign, positive offsets, chained by offset with each within 15% of the previous and the chain's max ≤ 1.35 × its min,
  magnitudes within 2× of the chain median, ≥ 3 units, no angular gap ≥ 180°), same line (angles within 10°, same or
  opposite, and offsets within 5% of the box width), the rest.
- Residual: max error and edge/inside (within 5% of a face); 2-D: 10 rings of distance from the formula's centre (vertex
  of its inner quadratic, else the box centre) and 12 angle sectors; 1-D: 10 equal slices.
- `nndc/gate_view.py` and `nndc/explain.py` are blind (dump, weights, blind spec, `ast5_trace.json` only) and NumPy only.
- `ast4` unchanged; ReLU equivalence 0 differences; no accepted program of an older version changes.
- Data commits apart from code commits; no attribution lines; no destructive commands on the repo or `runs*/`.

## Review Focus

1. **A Lawson round whose errors are all zero** (an exact fit) must stop, not divide by zero. Test in Task 1.
2. **A family whose minimax fit goes non-finite** on part of the box must keep the least-squares fit. Test in Task 1.
3. **A ring check on a net with fewer than 3 positive-offset units** must give no ring, not an error. Test in Task 3.
4. **`explain` on a ReLU or two-layer net:** ReLU gets the two-zone table; two layers get the formula part only, with a
   line saying the gate view needs one hidden layer. Test in Task 4.
5. **The near formula's centre in 2-D when its inner quadratic has no vertex** (a degenerate quadratic, e.g. `x0·x1`
   has a saddle; a pure linear u) falls back to the box centre. Test in Task 4.

---

### Task 1: Minimax fitting in the family route

**Files:** Modify `nndc/heuristics/families.py`; Test `tests/test_heuristics_families.py`

**Interfaces:**
- Produces: `LAWSON_ROUNDS = 40`, `LAWSON_SNAP_ROUNDS = 10`;
  `_lawson(m: Model, X, y, R: float, rounds: int, nfev: int) -> np.ndarray` (parameter values; returns the input values
  unchanged when the result is not better in max error or not finite). Used after `_fit_family`'s best start, in
  `_add_hinge`'s refit, and inside `_snap` (10 rounds, subsample). Every info row gains `"minimax_err"` (share of range).

- [ ] **Step 1: Write the failing tests**

```python
def test_lawson_lowers_max_error():
    # sqrt(x + 0.1) plus a bump near x = 3: least squares spreads the error, minimax lowers the worst point
    f = lambda X: np.sqrt(X[:, 0] + 0.1) + 0.06 * np.exp(-((X[:, 0] - 3) / 0.3) ** 2)
    _, info = run(f, spec_1d(0, 4))
    row = next(r for r in info if r["family"] == "sqrt")
    assert row["minimax_err"] < row["fit_err"]          # fit_err stays the least-squares error

def test_lawson_on_exact_fit_stops():
    _, info = run(lambda X: np.exp(X[:, 0]), spec_1d(-2, 2))
    assert next(r for r in info if r["family"] == "exp")["minimax_err"] <= 1e-9

def test_lawson_keeps_ls_fit_when_minimax_goes_nonfinite(monkeypatch): ...
    # patch _lawson's weighted refit to return NaN values: the row's minimax_err == fit_err and the family still runs
```

- [ ] **Step 2:** `uv run pytest tests/test_heuristics_families.py -q` → the 3 new tests FAIL (`KeyError: 'minimax_err'`).
- [ ] **Step 3:** Implement `_lawson` (weights ← weights × |error|, renormalised to mean 1, stop when all errors are 0
  or the max error stops falling for 3 rounds; weighted residuals in `_refit`) and wire it in as in Interfaces.
- [ ] **Step 4:** the family tests pass; time `family_candidates` on the pinned net
  (`runs_irr/sigmoid/dist/seed0`, a scratch copy) and on `runs_irr/sigmoid/bump/seed0`. Expected: under 10 minutes
  each; otherwise apply the fallback in Global Constraints and add the amendment.
- [ ] **Step 5: Commit** `heuristics: minimax (Lawson) refits in the family route`.

### Task 2: Near formulas

**Files:** Modify `nndc/heuristics/families.py`, `nndc/ast_tool.py`; Test `tests/test_heuristics_families.py`,
`tests/test_ast_tool.py`

**Interfaces:**
- Consumes: `_lawson` (Task 1).
- Produces:
  - each info row gains `"near": [{"terms", "src", "err", "nodes"}]`, one per term count tried, filled only for families
    that did not pass (relaxed snap at `max(tol, 1.25 × minimax_err)`);
  - `near_formula(rows: list[dict], dim: int) -> dict | None` in `families.py`, the choice rule of Global Constraints,
    returning `{"family", "src", "err", "terms", "nodes"}`;
  - `refactor(..., version="ast5")`: `trace["near"]` = `near_formula(...)` when no `family` or `conic` candidate passes;
    for a failing conic, `{"family": "conic", "src", "err": disagreements_outside / n_inputs, "terms": 0, "nodes"}`;
    otherwise `trace["near"] = None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_near_formula_prefers_fewer_terms_within_3pct():
    rows = [{"family": "reciprocal", "near": [{"terms": 4, "src": "r4", "err": 0.021, "nodes": 30}]},
            {"family": "sqrt", "near": [{"terms": 0, "src": "s0", "err": 0.025, "nodes": 9},
                                       {"terms": 4, "src": "s4", "err": 0.022, "nodes": 31}]}]
    assert near_formula(rows, 2)["src"] == "s0"

def test_near_formula_falls_back_to_smallest_error():
    rows = [{"family": "exp", "near": [{"terms": 0, "src": "e0", "err": 0.06, "nodes": 6}]},
            {"family": "log", "near": [{"terms": 2, "src": "l2", "err": 0.04, "nodes": 15}]}]
    assert near_formula(rows, 1)["src"] == "l2"

def test_identity_is_never_near_in_1d_but_quadratic_is_in_2d():
    assert near_formula([{"family": "identity", "near": [{"terms": 0, "src": "i", "err": 0.01, "nodes": 3}]}], 1) is None
    assert near_formula([{"family": "quadratic", "near": [{"terms": 0, "src": "q", "err": 0.025, "nodes": 5}]}], 2)["src"] == "q"

def test_near_rows_filled_when_family_fails():
    f = lambda X: np.sqrt(X[:, 0]) + 0.05 * np.sin(6 * X[:, 0])     # sqrt plus ripple: no family passes at 2%
    progs, info = run(f, spec_1d(0, 4))
    assert not any(fam == "sqrt" for fam, _ in progs)
    near = next(r for r in info if r["family"] == "sqrt")["near"]
    assert near and all(0 < c["err"] for c in near)

# tests/test_ast_tool.py
def test_ast5_trace_near_none_when_formula_passes(): ...   # the exp test net of tests/test_run.py: trace["near"] is None
```

- [ ] **Step 2:** run them → FAIL (`ImportError: near_formula`).
- [ ] **Step 3:** implement as in Interfaces (the relaxed snap reuses `_snap` with the relaxed tolerance; run it only
  after the family has failed every term count).
- [ ] **Step 4:** `uv run pytest tests/test_heuristics_families.py tests/test_ast_tool.py -q` → PASS.
- [ ] **Step 5: Commit** `ast5: near formulas when no formula passes (trace only, not a version)`.

### Task 3: The gate view

**Files:** Create `nndc/gate_view.py`; Test `tests/test_gate_view.py`

**Interfaces:**
- Produces:
  - `unit_table(spec: Spec, mlp: MLP) -> list[dict]`, one-hidden-layer nets (tanh via `as_sigmoid`), keys: `unit`,
    `angle` (degrees, 2-D; 1-D: 0 or 180), `offset`, `width`, `zones` (dict zone → share), `magnitude`, `sign`,
    `stand_in` (name or None), `stand_in_err`;
  - `tolerance(spec, mlp) -> float` as in Global Constraints;
  - `groups(units: list[dict], tol: float, box_width: float) -> list[dict]`, each `{"kind": "negligible" | "lines" |
    "ring" | "same line" | "rest", "units": [...], "summary": {...}}` (ring summary: offset min/max, magnitude median,
    count; negligible: summed magnitude);
  - raises `ValueError` for nets with more or fewer than one hidden layer.

- [ ] **Step 1: Write the failing tests** (synthetic nets built from weights, no runs)

```python
def ring_net(n=6, r=1.0, k=4.0, mag=1.0, extra=()):
    """n sigmoid units with gates at distance r around the origin, plus extra (w0, w1, b, out) units."""
    ...

def test_ring_found():
    g = groups(*_view(ring_net()))
    assert [len(x["units"]) for x in g if x["kind"] == "ring"] == [6]

def test_two_rings_split_by_offset():
    g = groups(*_view(ring_net(extra=ring_units(n=5, r=2.5))))
    assert sorted(len(x["units"]) for x in g if x["kind"] == "ring") == [5, 6]

def test_negative_offset_units_never_join_a_ring(): ...   # 4 units with offset -0.7 at spread angles: no ring
def test_fewer_than_three_units_no_ring(): ...             # 2 units at r = 1: kind "rest"
def test_negligible_and_lines_first(): ...                 # a tiny unit -> negligible; a unit with |z| < 1 on the box -> lines
def test_same_line_pair(): ...                             # two units, same gate line, opposite directions -> "same line"
def test_zones_of_a_steep_unit(): ...                      # gate through the middle, |w| = 20: off ≈ 0.5, on ≈ 0.5
def test_relu_units_two_zones(): ...                       # zones keys == {"off", "active"}
def test_two_layer_net_refused(): ...                      # ValueError
```

- [ ] **Step 2:** run → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3:** implement as in Interfaces and Global Constraints.
- [ ] **Step 4:** `uv run pytest tests/test_gate_view.py -q` → PASS. Then the pinned case on the real net (not a test;
  `runs_irr` is not committed yet): inner ring of 5 (offset 0.7–0.9), outer ring of 4 or 5 (2.4–3.6), ≥ 2 negligible,
  units 7 and 14 in no ring. Record the result in the ledger.
- [ ] **Step 5: Commit** `gate view: units by gate, zone, magnitude and sign; groups (negligible, lines, rings, same line)`.

### Task 4: Residual description and `nndc explain`

**Files:** Create `nndc/explain.py`; Modify `nndc/cli.py`; Test `tests/test_explain.py`

**Interfaces:**
- Consumes: `unit_table`, `tolerance`, `groups` (Task 3); `trace["near"]` (Task 2).
- Produces:
  - `describe_residual(spec: Spec, predict, src: str) -> dict`: `max_err` (share of the net's range), `where`
    ("edge" | "inside"), and `by_slice` (1-D, 10 values) or `by_ring` (10) and `by_sector` (12) with `centre` (2-D);
  - `formula_centre(src: str, spec: Spec) -> tuple[float, float]`: the vertex of the inner quadratic when the
    quadratic part is definite, else the box centre;
  - `explain_run(rd: Path) -> str`, writing `rd / "explain.md"` and `record_git(rd, "explain")`;
  - `nndc explain RUNS… | --all --root R`.

- [ ] **Step 1: Write the failing tests**

```python
def test_residual_finds_edge_error(): ...          # predict = x + 0.1·relu(x - 3.8) on [0, 4], src = "return x0": where == "edge"
def test_residual_by_sector_shows_ripple(): ...    # predict = r + 0.05·cos(5θ): by_sector varies, by_ring flat-ish
def test_formula_centre_vertex_and_fallback():
    assert formula_centre("def net(x0, x1):\n    return sqrt((x0 - 0.5) * (x0 - 0.5) + x1 * x1)\n", spec2) == pytest.approx((0.5, 0.0), abs=1e-6)
    assert formula_centre("def net(x0, x1):\n    return x0 * x1\n", spec2) == (0.0, 0.0)     # saddle: box centre
def test_explain_run_writes_markdown(tmp_path): ...  # tmp run (build_run of a short sigmoid "exp" net, ast5 run):
                                                     # explain.md has "## Formula" and "## Units"; run_git.txt has "explain:"
def test_explain_two_layer_net_formula_only(tmp_path): ...   # a median3-style 2-layer tmp net: "needs one hidden layer"
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement; `explain.md` sections: `## Formula` (passing program, or near formula with its error and
  residual lines), `## Units` (the table), `## Groups` (one line per group in words). No task name, target or family.
- [ ] **Step 4:** tests pass; `uv run nndc explain runs_irr/sigmoid/dist/seed0` on a scratch copy prints sensible
  groups (the pinned case again, now in `explain.md`).
- [ ] **Step 5: Commit** `nndc explain: near formula, residual and gate view per run (explain.md)`.

### Task 5: Analysis and report

**Files:** Modify `experiments/irrational_analysis.py`, `nndc/irrational_report.py`; Test
`tests/test_irrational_analysis.py`, `tests/test_irrational_report.py`

**Interfaces:**
- Consumes: `trace["near"]` (Task 2).
- Produces: analysis keys `near_family`, `near_err`, `near_terms`, `near_family_is_task_family`, `explain` (bool:
  `explain.md` exists). Report: results table column "near formula" (`family err%` for nets without a found formula),
  a link to `explain.md`, the snapping table's column "near formula within 3%" (regression), and the new design's
  three prediction lines (sigmoid dist found ≥ 2 of 3; ≥ 90% of learned regression nets found-or-near-of-family
  within 3%; on the 18 earlier nets, found count not lower and never larger where both pass — the last one reads a
  saved list, see Task 6).

- [ ] **Step 1: Write the failing tests** (fake analysis rows as in the existing report tests): the near column shows
  `sqrt 2.5%`; the 90% prediction line reads met/not met from fake rows; the link `[explain](sigmoid/dist/seed0/explain.md)`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** both test files pass, then `uv run pytest -q` (full
  suite, background, ~25 min) → all pass.
- [ ] **Step 5: Commit** `irrational report: near formulas, explain links, the gate-view design's predictions`.

### Task 6: Development rerun

- [ ] **Step 1:** Before rerunning, save the 18 earlier results for the comparison prediction:
  `runs_irr/before_minimax.json` = per run `{formula_found, ast5_nodes, ast5_pass}` from the current `irr_analysis.json`
  files (written by a one-off script; the file is committed with the data).
- [ ] **Step 2:** Equivalence: `ast4` regenerates with the same content on 3 scratch copies; ReLU `experiments.equivalence`
  compare → 0 differences.
- [ ] **Step 3:** All 108 runs, 6 at a time in the background: `ast4` (only where missing), `ast5`, `pysr_irr` (only
  where missing), `irr-analyse`, `explain`. Then `nndc irr-report`, plots, and `evaluate_run` over every run.
- [ ] **Step 4:** Record the first development numbers of both designs' predictions as amendments (irrational design:
  the first development run, now with this design's route; this design: its three predictions). Data commit:
  `data: irrational development nets (108) with ast5 (minimax, near formulas), explain.md, analysis`.
- [ ] **Step 5:** Continue with Tasks 9–11 of `docs/plans/2026-10-30-irrational-functions.md` (development changes,
  freeze, held-out, write-up).
