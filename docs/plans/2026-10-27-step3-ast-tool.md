# Step 3: AST heuristics tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic, LLM-free tool (`nndc ast`) that refactors a decompiled ReLU net into a clean program using the heuristics Claude Code wrote down, and an honest measurement of how often it reproduces Claude's refactorings. The headline is measured on held-out seed-2 models.

**Architecture:**
- `nndc/symnet.py` parses a `dump.py` (via `ast`) into a symbolic net: units = `relu(affine)`, plus output affines.
- One module per heuristic under `nndc/heuristics/` either transforms the symbolic net or proposes a candidate expression.
- `nndc/ast_tool.py` builds candidates along four routes (structural, boundary, truth table, fit), checks each with the existing checker, and keeps the best by (exact, fewest nodes).
- `ast.py` becomes one more program version in each run folder, evaluated and reported like `dump.py` and `agent_n.py`.

**Tech Stack:** Python 3.13, NumPy, existing `nndc` package (lang, check, run, report), pytest.

**Spec:** `docs/design/2026-10-27-step3-ast-tool-design.md` (parent: `docs/design/2026-10-27-nn-decompiler-design.md`).

## Global Constraints

- **No LLM** anywhere in the tool.
- **Never consult the target or task name.** The tool receives only:
  - the dump source
  - the **blind** spec (`Spec.from_json(spec.blind())`)
  - the `MLP` (the same weights Claude saw)
  - the checker

  Selection is by checker verdict and node count only.
- **Held-out discipline:**
  - No test, fixture, or development command may read anything under `runs/*/seed2/` before Task 8.
  - `nndc ast --all` takes `--split distill|heldout|all`. The default is `distill`, which means every run except `seed2`.
  - The held-out run happens once, in Task 8, after the tool's code is committed. Its commit hash is recorded in `REPORT.md`.
- **Thresholds fixed here** (change only with a ledgered ruling before Task 8):
  - Near-merge: direction cosine ≥ 0.98 and threshold difference ≤ 0.02 × (input range along that direction).
  - Snap: relative tolerance 0.05 for coefficients; 0.01 × input range for thresholds.
  - Small-term drop: combined `Σ|wᵢ|·max|xᵢ|` ≤ 0.005 × the unit's output range.
  - Fit acceptance: R² > 0.999.
  - Truth-table cap: 4096 inputs.
  - Reproduction: passes check, same `recovered` status as Claude's best valid attempt, and nodes ≤ max(1.5 × Claude's nodes, Claude's nodes + 5).
- **Nice-value sequence for snapping**, in order of preference:
  1. `0`
  2. integers
  3. multiples of `1/2`
  4. multiples of `1/4`, `1/5`, `1/8`, `1/10`, `1/20`, `1/50`, `1/100`

  The first one within tolerance wins.
- Program output must parse under `nndc.lang` (version 1 allowlist). No new language constructs.
- Tests use handcrafted nets (`experiments.handcrafted`) or synthetic nets built in the test, never `runs/`.
- Commit messages: conventional style, no attribution lines.

## Review Focus

1. **A dump whose shape the parser doesn't expect** (an `agent_n.py` passed by mistake, or a hand-edited dump) → `from_dump` raises `ValueError` naming the line. `nndc ast` reports it for that run and continues with the others. Pinned in Task 1.
2. **A net where every unit is dead** (constant output) → the structural route emits `return <constant>`, which passes the check. Pinned in Task 2.
3. **Two-layer nets (median3)** → P4/P5/P6 apply only to the last hidden layer when it reads inputs directly. Otherwise the structural route emits the pruned and merged net as code, and the fit route may win. Nothing crashes. Pinned in Task 6.
4. **Argmax outputs** → the structural route keeps `argmax(o0, …)`, and boundary/truth-table routes are skipped. Pinned in Task 6.
5. **Re-running `nndc ast`** overwrites `ast.py`/`ast_trace.json` but never touches `agent_*` files, and `evaluate_run` keeps the `agent_*` versions. Pinned in Task 7.

---

### Task 1: Symbolic net (parse, evaluate, bounds, emit)

**Files:**
- Create: `nndc/symnet.py`, `tests/test_symnet.py`, `tests/helpers.py` (shared test helpers: `hand(name) -> (MLP, blind Spec)`, `random_mlp(sizes, seed)`, `normalize_ws(s)`; later test files import them)

**Interfaces:**
- Consumes: `nndc.lang.parse_program`, `nndc.lang.evaluate`, `nndc.spec.Spec`, `nndc.dump.dump_program`.
- Produces:
  ```python
  @dataclass
  class Affine:
      coef: dict[str, float]         # variable name -> coefficient (no zeros stored)
      const: float = 0.0
      def eval(self, env: dict[str, np.ndarray]) -> np.ndarray
      def interval(self, bounds: dict[str, tuple[float, float]]) -> tuple[float, float]
      def scaled(self, k: float) -> "Affine"
      def plus(self, other: "Affine") -> "Affine"
      def to_src(self) -> str        # e.g. "x0 - x1 + 0.5"; "0" when empty

  @dataclass
  class Unit:
      name: str                      # "h1_0"
      layer: int
      pre: Affine                    # over inputs or previous-layer unit names
      relu: bool = True              # False once proven always-on (P1)

  @dataclass
  class SymNet:
      inputs: list[str]
      units: list[Unit]              # topological order
      outputs: list[Affine]          # 1 for regression/binary, K for argmax
      kind: str                      # spec.output
      threshold: float = 0.0
      def copy(self) -> "SymNet"
      def unit(self, name: str) -> Unit

  def from_dump(src: str, spec: Spec) -> SymNet
  def evaluate(net: SymNet, X: np.ndarray) -> np.ndarray        # (N, n_inputs) -> (N, n_outputs) float64
  def unit_bounds(net: SymNet, spec: Spec) -> dict[str, tuple[float, float]]   # pre-activation interval per unit
  def to_program(net: SymNet) -> str                            # valid program-language source
  def substitute(net: SymNet, name: str, value: Affine) -> SymNet  # replace a unit's output by an affine everywhere
  ```

- [ ] **Step 1: Write failing tests** in `tests/test_symnet.py`:
  ```python
  def hand(name):
      t = get_task(name)
      return t.handcrafted(), Spec.from_json(t.spec.blind())

  @pytest.mark.parametrize("name", ["absdiff", "deadredundant", "xor", "triangle", "window"])
  def test_roundtrip_dump_symnet_program(name):
      mlp, spec = hand(name)
      net = from_dump(dump_program(mlp, spec), spec)
      X = build_input_set(mlp, spec)[0][:5000]
      np.testing.assert_allclose(evaluate(net, X)[:, 0], mlp.forward(X.astype(np.float32))[:, 0], atol=1e-5)
      assert check_program(to_program(net), mlp, spec).exact_pass

  def test_unit_bounds_deadredundant():
      mlp, spec = hand("deadredundant")
      b = unit_bounds(from_dump(dump_program(mlp, spec), spec), spec)
      assert b["h1_0"] == pytest.approx((0, 2)) and b["h1_1"] == pytest.approx((-3, -1))

  def test_from_dump_rejects_non_dump_shape():
      spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
      with pytest.raises(ValueError, match="line"):
          from_dump("def net(x0):\n    if x0 > 1:\n        return 1\n    return 0\n", spec)

  def test_substitute_inlines_affine():
      mlp, spec = hand("deadredundant")
      net = from_dump(dump_program(mlp, spec), spec)
      net2 = substitute(net, "h1_1", Affine({}, 0.0))
      assert "h1_1" not in to_program(net2)
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_symnet.py -q`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `nndc/symnet.py`.
  - `from_dump` accepts only: assignments `name = relu(<affine>)` or `name = <affine>`, and `return <affine>` or `return argmax(o0, ...)`. An affine is a sum of `c * var`, `var`, `-c * var` and constants. Anything else raises `ValueError(f"line {n}: not a dump-shaped statement")`.
  - `unit_bounds` propagates intervals layer by layer, using post-relu intervals for the next layer.
  - `to_program` writes one assignment per remaining unit (`relu(...)` only if `unit.relu`) and the return. Comments are added only where the spec header says the output kind.
- [ ] **Step 4: Run** `uv run pytest tests/test_symnet.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: symbolic net parsed from dumps (eval, bounds, emit)`.

---

### Task 2: P1 bounds and P2 normalize/merge

**Files:**
- Create: `nndc/heuristics/__init__.py`, `nndc/heuristics/bounds.py`, `nndc/heuristics/merge.py`, `tests/test_heuristics_structure.py`

**Interfaces:**
- Consumes: `SymNet`, `Affine`, `unit_bounds`, `substitute` (Task 1).
- Produces:
  ```python
  # bounds.py   (HEURISTICS: deadredundant/hand, tax/seed0-1, window/seed0-1, absdiff/seed1, neareq/seed0-1, sum8/hand, xor/seed0, ramp/hand)
  def prune_dead_and_linear(net: SymNet, spec: Spec) -> tuple[SymNet, list[str]]   # (net, trace lines)
  # merge.py    (HEURISTICS: absdiff/seed0-1, neareq/seed0-1, deadredundant/hand, median3/seed1, xor/seed1, window/seed0)
  def normalize(net: SymNet) -> SymNet           # unit pre-affine scaled to L1 norm 1 over its vars; consumers rescaled (exact)
  def merge_units(net: SymNet, spec: Spec, near: bool) -> tuple[SymNet, list[str]]
      # near=False: merge only identical normalized units; near=True: cos >= 0.98 and |dt| <= 0.02*range
  ```
- [ ] **Step 1: Write failing tests:**
  ```python
  def test_dead_and_linear_units_removed():
      mlp, spec = hand("deadredundant")
      net, trace = prune_dead_and_linear(from_dump(dump_program(mlp, spec), spec), spec)
      names = [u.name for u in net.units if u.relu]
      assert "h1_1" not in [u.name for u in net.units] and "h1_0" not in names   # dead removed, always-on linearized
      assert check_program(to_program(net), mlp, spec).exact_pass and trace

  def test_all_dead_gives_constant():
      mlp = MLP([np.array([[1.0]], np.float32), np.array([[3.0]], np.float32)],
                [np.array([-5.0], np.float32), np.array([0.25], np.float32)])
      spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
      net, _ = prune_dead_and_linear(from_dump(dump_program(mlp, spec), spec), spec)
      assert to_program(net).strip().endswith("return 0.25")

  def test_duplicates_merge_exactly():
      mlp, spec = hand("deadredundant")
      net, _ = merge_units(normalize(from_dump(dump_program(mlp, spec), spec)), spec, near=False)
      assert sum(u.relu for u in net.units) == 3          # h1_2 and h1_3 became one
      assert check_program(to_program(net), mlp, spec).exact_pass

  def test_near_merge_is_approximate_but_passes_snap():
      mlp, spec = hand("absdiff_perturbed")
      net, _ = merge_units(normalize(from_dump(dump_program(mlp, spec), spec)), spec, near=True)
      assert check_program(to_program(net), mlp, spec).snap_pass
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_heuristics_structure.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules. Each trace line names the unit and the reason, e.g. `"h1_1: dead (pre-activation in [-3, -1])"`.
- [ ] **Step 4: Run** `uv run pytest tests/test_heuristics_structure.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: heuristics P1 (dead/linear units) and P2 (normalize, merge)`.

---

### Task 3: P3 snapping

**Files:**
- Create: `nndc/heuristics/snap.py`, `tests/test_heuristics_snap.py`

**Interfaces:**
- Produces:
  ```python
  # (HEURISTICS: absdiff_perturbed/hand, sparse8/hand, sum8/hand, tax/seed0, xor/seed0-1, absdiff/seed0-1)
  NICE_DENOMS = (1, 2, 4, 5, 8, 10, 20, 50, 100)
  def nice(v: float, tol: float) -> float            # first of 0, ints, then k/denom in NICE_DENOMS order, within tol; else v
  def snap_net(net: SymNet, spec: Spec) -> tuple[SymNet, list[str]]
      # coefficients: tol = 0.05*|v| (and 0 if the small-term rule allows dropping);
      # thresholds t = -const/scale: tol = 0.01 * input range along the unit's direction;
      # small terms dropped only if their combined effect <= 0.005 * unit output range
  ```
- [ ] **Step 1: Write failing tests:**
  ```python
  @pytest.mark.parametrize("v, tol, out", [(0.98, 0.05, 1.0), (-1.03, 0.06, -1.0), (0.124, 0.006, 0.125),
                                            (0.0013, 0.01, 0.0), (0.37, 0.001, 0.37), (0.198, 0.004, 0.2)])
  def test_nice(v, tol, out):
      assert nice(v, tol) == pytest.approx(out)

  def test_snap_perturbed_absdiff_to_unit_weights():
      mlp, spec = hand("absdiff_perturbed")
      net, trace = snap_net(normalize(from_dump(dump_program(mlp, spec), spec)), spec)
      src = to_program(net)
      assert check_program(src, mlp, spec).snap_pass and grade_task(src, get_task("absdiff_perturbed"))["recovered"]

  def test_small_weights_dropped_only_when_combined_effect_small():
      mlp, spec = hand("sparse8")   # 6 weights of 0.0005 on [0,1]: combined 0.003 -> dropped
      src = to_program(snap_net(from_dump(dump_program(mlp, spec), spec), spec)[0])
      assert "x0" not in src.split("def net")[1].split("\n", 1)[1]
      mlp, spec = hand("sum8")      # 8 weights of 0.02 on [0,255]: combined 40.8 -> kept
      src = to_program(snap_net(from_dump(dump_program(mlp, spec), spec), spec)[0])
      assert all(f"x{k}" in src.split("def net")[1].split("\n", 1)[1] for k in range(8))
  ```
  (Using `grade_task` in a *test* is allowed; only the tool itself may not consult the target.)
- [ ] **Step 2: Run** `uv run pytest tests/test_heuristics_snap.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `nndc/heuristics/snap.py`.
- [ ] **Step 4: Run** `uv run pytest tests/test_heuristics_snap.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: heuristic P3 (snapping with combined small-term rule)`.

---

### Task 4: P4 motifs, P5 1-D spline, P9 cleanup/emission

**Files:**
- Create: `nndc/heuristics/motifs.py`, `nndc/heuristics/spline.py`, `nndc/heuristics/cleanup.py`, `tests/test_heuristics_motifs.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class ReluSum:                 # one output of a single-hidden-layer net over input-level affines
      terms: list[tuple[float, Affine]]   # c * relu(z)
      linear: Affine
  def as_relu_sum(net: SymNet, out: int = 0) -> ReluSum | None   # None if the net isn't 1 hidden layer over inputs
  # motifs.py (HEURISTICS: max/hand, deadredundant/hand, absdiff/hand, ramp/hand, triangle/hand, window/hand, neareq/hand, sum8/hand, xor/hand)
  def motif_expr(rs: ReluSum) -> str      # exact rewrite into abs/max/min/clamp/hat/sum where identities hold
  # spline.py (HEURISTICS: tax/seed0-1, triangle/hand, ramp/hand, window/seed0, absdiff/seed0)
  def spline_expr(rs: ReluSum, spec: Spec) -> str | None   # only if all z share one direction; kinks merged/snapped; convex->max, concave->min
  # cleanup.py (HEURISTICS: sparse8/hand, sum8/hand)
  def program_from_expr(expr: str, spec: Spec, derived: dict[str, str] | None = None) -> str
      # "def net(...)": named derived variables (d0 = x0 - x1) then return expr; equal coefficients -> c * sum(...)
  ```
- [ ] **Step 1: Write failing tests**. Each case asserts that the expression is exactly equivalent and is the expected readable form:
  ```python
  @pytest.mark.parametrize("name, expected", [
      ("absdiff", "abs(x0 - x1)"), ("max", "max(x0, x1)"), ("ramp", "min(max(x0, 0), 1)"),
      ("triangle", "max(0, 1 - abs(x0 - 3))"), ("window", "0.5 - abs(x0)"), ("neareq", "0.2 - abs(x0 - x1)"),
      ("sum8", "max(0, 0.02 * sum(x0, x1, x2, x3, x4, x5, x6, x7) - 20)"),
  ])
  def test_motifs_on_handcrafted(name, expected):
      mlp, spec = hand(name)
      rs = as_relu_sum(from_dump(dump_program(mlp, spec), spec))
      src = program_from_expr(motif_expr(rs), spec)
      assert check_program(src, mlp, spec).exact_pass
      assert normalize_ws(src.splitlines()[-1]) == f"return {expected}"   # normalize_ws: test helper, collapses spaces

  def test_spline_convex_to_max_of_lines():
      # 1-D convex net: 0.1*x + 0.1*relu(x-20) + 0.2*relu(x-80) on [0, 200]
      mlp = MLP([np.array([[1], [1], [1]], np.float32), np.array([[0.1, 0.1, 0.2]], np.float32)],
                [np.array([0, -20, -80], np.float32), np.zeros(1, np.float32)])
      spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 200)])
      expr = spline_expr(as_relu_sum(from_dump(dump_program(mlp, spec), spec)), spec)
      assert expr.startswith("max(") and check_program(program_from_expr(expr, spec), mlp, spec).snap_pass

  def test_two_layer_net_has_no_relu_sum():
      mlp = random_mlp([3, 4, 4, 1], seed=0)   # helper as in tests/test_dump.py
      spec = Spec(task="", output="regression", inputs=[InputSpec(f"x{i}", f"x{i}", "float", -1, 1) for i in range(3)])
      assert as_relu_sum(from_dump(dump_program(mlp, spec), spec)) is None
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_heuristics_motifs.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules.
  - **Motif matching:** match on `ReluSum` terms by comparing their affines.
    - Opposite pairs: `z2 == -z1`.
    - Shifted copies: same coefficients, different constants.
    - The hat pattern: `+a, -2a, +a` on equally spaced constants.
    - Linear passthrough: `relu(z) - relu(-z) = z`. Fold it into `linear` first.
    - Then `linear + relu(...)` → `max`/`min` when `linear` equals the relu's argument offset.
  - **Output form:** emit the simplest equivalent of the expected readable forms, with `0.5 - abs(x0)` rather than `-abs(x0) + 0.5`.
- [ ] **Step 4: Run** `uv run pytest tests/test_heuristics_motifs.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: heuristics P4 (motifs), P5 (1-D spline), P9 (emission)`.

---

### Task 5: P6 boundary, P7 truth table, P8 candidate fit

**Files:**
- Create: `nndc/heuristics/boundary.py`, `nndc/heuristics/truthtable.py`, `nndc/heuristics/fit.py`, `tests/test_heuristics_rules.py`

**Interfaces:**
- Produces:
  ```python
  # boundary.py (HEURISTICS: neareq/seed0-1, window/seed0-1, window/hand, neareq/hand, xor/seed1)
  def boundary_expr(net: SymNet, spec: Spec) -> tuple[str, dict[str, str]] | None
      # binary only; all units on one direction d (after normalize/merge); returns (expr, derived vars)
      # e.g. ("abs(d0) < 0.2", {"d0": "x0 - x1"}); crossings snapped with nice(·, 0.01*range of d)
  # truthtable.py (HEURISTICS: xor/seed0-1, xor/hand)
  TT_CAP = 4096
  def truth_table_expr(decide: Callable[[np.ndarray], np.ndarray], spec: Spec) -> str | None
      # binary + discrete domain <= TT_CAP; library: xi, not xi, all 2-input functions per pair,
      # all/any/parity of all inputs, sum(x) >= k; returns the first match by expression size
  # fit.py (HEURISTICS: median3/seed0-1)
  def fit_expr(predict: Callable[[np.ndarray], np.ndarray], spec: Spec, seed: int = 0) -> str | None
      # regression, <= 4 inputs; features: xi, sum, min/max over all and pairs, median3 (3 inputs),
      # abs(xi - xj), relu(xi - xj); least squares y ~ a*f + b; accept R^2 > 0.999; a, b snapped with nice()
  ```
- [ ] **Step 1: Write failing tests:**
  ```python
  def test_truth_table_finds_xor():
      mlp, spec = hand("xor")
      decide = lambda X: (mlp.forward(X.astype(np.float32))[:, 0] > 0).astype(float)
      assert truth_table_expr(decide, spec) in ("x0 != x1", "x1 != x0")

  def test_truth_table_skips_large_or_continuous():
      _, spec = hand("absdiff")
      assert truth_table_expr(lambda X: X[:, 0] > 0, spec) is None

  def test_boundary_snaps_window_from_trained_like_net():
      # perturbed window: 0.51 - relu(1.02 x) - relu(-0.98 x)
      mlp = MLP([np.array([[1.02], [-0.98]], np.float32), np.array([[-1, -1]], np.float32)],
                [np.zeros(2, np.float32), np.array([0.51], np.float32)])
      spec = Spec(task="", output="binary", inputs=[InputSpec("x0", "x0", "float", -2, 2)])
      expr, derived = boundary_expr(from_dump(dump_program(mlp, spec), spec), spec)
      src = program_from_expr(expr, spec, derived)
      assert "abs(" in src and "0.5" in src and check_program(src, mlp, spec).snap_pass

  def test_fit_finds_median():
      spec = Spec(task="", output="regression", inputs=[InputSpec(f"x{i}", f"x{i}", "float", -3, 3) for i in range(3)])
      expr = fit_expr(lambda X: np.median(X, axis=1) * 1.003 + 0.001, spec)
      X = np.random.default_rng(1).uniform(-3, 3, (1000, 3))
      prog = parse_program(program_from_expr(expr, spec), spec.arg_names)
      np.testing.assert_allclose(evaluate(prog, X), np.median(X, axis=1), atol=1e-9)

  def test_fit_rejects_poor_fits():
      spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", -3, 3), InputSpec("x1", "x1", "float", -3, 3)])
      assert fit_expr(lambda X: np.sin(3 * X[:, 0]) * X[:, 1], spec) is None
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_heuristics_rules.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules.
  - `evaluate`/`parse_program` here come from `nndc.lang`.
  - `median3` is emitted as `max(min(a, b), min(max(a, b), c))`.
  - Boolean functions are emitted with `!=`, `==`, `and`, `or`, `not`, and comparisons only.
- [ ] **Step 4: Run** `uv run pytest tests/test_heuristics_rules.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: heuristics P6 (boundary), P7 (truth table), P8 (candidate fit)`.

---

### Task 6: The tool, with search, selection and trace

**Files:**
- Create: `nndc/ast_tool.py`, `tests/test_ast_tool.py`

**Interfaces:**
- Consumes: Tasks 1–5, `check_program`, `program_size`, `parse_program`, `MLP`.
- Produces:
  ```python
  @dataclass
  class Candidate:
      route: str                 # "structural-exact" | "structural-snap" | "spline" | "boundary" | "truth-table" | "fit"
      src: str
      passes: list[str]          # e.g. ["P1", "P2", "P4", "P9"]
      check: dict                # CheckResult.to_json()
      nodes: int | None

  def refactor(dump_src: str, spec: Spec, mlp: MLP) -> tuple[str, dict]
      # spec must be blind (task == ""), else ValueError; returns (best program source, trace dict)
      # trace = {"candidates": [asdict(Candidate) ...], "winner": route, "trace_lines": [...]}
  ```
- **Routes, in order:**
  - **structural-exact:** P1 → P2(exact) → P4 if `as_relu_sum` else the pruned net → P9.
  - **structural-snap:** P1 → P2(near) → P3 → P4/P9.
  - **spline:** P5 on the snapped ReluSum.
  - **boundary:** P6 (binary only).
  - **truth-table:** P7 (binary, small discrete).
  - **fit:** P8 (regression, ≤ 4 inputs).
  - Argmax nets run only the two structural routes.
- **Selection:** among candidates with `snap_pass`, take `exact_pass` first, then fewest nodes, then route order. If none passes, use structural-exact. It always passes, because it is exact.

- [ ] **Step 1: Write failing tests:**
  ```python
  @pytest.mark.parametrize("name", ["absdiff", "absdiff_perturbed", "deadredundant", "max", "ramp", "triangle",
                                    "window", "neareq", "xor", "sparse8", "sum8"])
  def test_refactor_handcrafted_recovers(name):
      t = get_task(name); mlp = t.handcrafted(); spec = Spec.from_json(t.spec.blind())
      src, trace = refactor(dump_program(mlp, spec), spec, mlp)
      assert check_program(src, mlp, spec).snap_pass and grade_task(src, t)["recovered"], (name, src)
      assert trace["winner"] and trace["candidates"]

  def test_refactor_requires_blind_spec():
      t = get_task("absdiff"); mlp = t.handcrafted()
      with pytest.raises(ValueError, match="blind"):
          refactor(dump_program(mlp, t.spec), t.spec, mlp)

  def test_two_layer_and_argmax_nets_do_not_crash():
      for sizes, kind in [([3, 6, 6, 1], "regression"), ([2, 4, 3], "argmax")]:
          mlp = random_mlp(sizes, seed=3)
          spec = Spec(task="", output=kind, n_outputs=sizes[-1],
                      inputs=[InputSpec(f"x{i}", f"x{i}", "float", -1, 1) for i in range(sizes[0])])
          src, trace = refactor(dump_program(mlp, spec), spec, mlp)
          assert check_program(src, mlp, spec).snap_pass
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_ast_tool.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `nndc/ast_tool.py`.
- [ ] **Step 4: Run** `uv run pytest tests/test_ast_tool.py -q`. Expected: PASS. If a handcrafted case fails, fix the heuristic, never the expectation; these are the distillation set.
- [ ] **Step 5: Commit** `feat: ast tool with candidate routes, checker-based selection and trace`.

---

### Task 7: Integration (`nndc ast`, run versions, report reproduction table)

**Files:**
- Modify: `nndc/run.py` (`evaluate_run` includes `ast.py`), `nndc/cli.py` (`ast` subcommand), `nndc/report.py` (reproduction section)
- Test: `tests/test_ast_integration.py`

**Interfaces:**
- Produces:
  ```python
  # nndc/run.py: versions = dump, agent_n..., and "ast" if ast.py exists
  # nndc/ast_tool.py
  def run_ast(rd: Path) -> dict            # reads dump.py + blinded spec + weights; writes ast.py, ast_trace.json; evaluate_run
  # nndc/report.py
  def reproduction(root: Path = RUNS) -> list[dict]
      # per run with both ast and a valid agent attempt: {"run", "split": "heldout"|"distill",
      #  "ast_check", "agent_check", "ast_recovered", "agent_recovered", "ast_nodes", "agent_nodes",
      #  "reproduces": bool, "route": str}
  ```
  - CLI: `nndc ast <run>... | --all [--split distill|heldout|all]`. The default split is `distill`, meaning all runs whose folder is not `seed2`.
  - `REPORT.md` gains:
    - a "Reproduction (step 3)" section with held-out and in-sample counts shown separately
    - a per-route win count
    - the list of runs where Claude recovered the formula and the tool did not
  - If `runs/ast_freeze.txt` exists, the reproduction section starts with `Tool frozen at <hash> before the held-out run.`
  - The "valid agent attempt" is Claude's best attempt with an empty `outside_paths`. If a run has no valid attempt, its reproduction row is marked "no valid Claude attempt" and excluded from the counts.

- [ ] **Step 1: Write failing tests** (using `tmp_path` runs built with `build_run(..., root=tmp_path)`, never `runs/`):
  ```python
  def test_run_ast_adds_version_and_keeps_agents(tmp_path):
      rd = build_run("absdiff", None, root=tmp_path)
      (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
      res = run_ast(rd)
      assert (rd / "ast.py").exists() and (rd / "ast_trace.json").exists()
      assert {"dump", "agent_1", "ast"} <= set(res["versions"]) and res["versions"]["ast"]["task"]["recovered"]
      run_ast(rd)                                   # rerun: overwrites ast, keeps agent_1
      assert (rd / "agent_1.py").exists()

  def test_reproduction_rows_and_split(tmp_path):
      for seed_dir in ["hand", "seed2"]:
          rd = tmp_path / "absdiff" / seed_dir
          rd.mkdir(parents=True)
          (rd / "results.json").write_text(json.dumps({"run": f"absdiff/{seed_dir}", "versions": {
              "dump": {"size": {"nodes": 50}, "check": {"snap_pass": True, "exact_pass": True}, "task": {"recovered": False}, "meta": None},
              "agent_1": {"size": {"nodes": 10}, "check": {"snap_pass": True, "exact_pass": True}, "task": {"recovered": True},
                          "meta": {"outside_paths": []}},
              "ast": {"size": {"nodes": 14}, "check": {"snap_pass": True, "exact_pass": True}, "task": {"recovered": True}, "meta": None}}}))
          (rd / "spec.json").write_text("{}")
      rows = {r["run"]: r for r in reproduction(tmp_path)}
      assert rows["absdiff/seed2"]["split"] == "heldout" and rows["absdiff/hand"]["split"] == "distill"
      assert rows["absdiff/hand"]["reproduces"]          # 14 <= max(15, 15)

  def test_cli_ast_default_split_skips_seed2(tmp_path, monkeypatch):
      monkeypatch.chdir(tmp_path)
      build_run("absdiff", None, root=tmp_path / "runs")
      (tmp_path / "runs" / "absdiff" / "seed2").mkdir(parents=True)   # must not be touched
      assert main(["ast", "--all"]) == 0
      assert (tmp_path / "runs/absdiff/hand/ast.py").exists()
      assert not (tmp_path / "runs/absdiff/seed2/ast.py").exists()
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_ast_integration.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the changes.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: nndc ast command, ast version in runs, reproduction report`.

---

### Task 8: In-sample run, freeze, held-out run, report

**Files:**
- Generated: `runs/**/ast.py`, `runs/**/ast_trace.json`, `runs/**/results.json`, `REPORT.md`
- Modify: `README.md` (one paragraph on step 3 and its headline numbers)

- [ ] **Step 1: In-sample run.** `uv run nndc ast --all` (distill split, 23 runs). Read the reproduction section of `REPORT.md` (`uv run nndc report`).
  - Improving heuristics on in-sample failures is allowed here. Each improvement needs a test in Tasks 2–6 style, made with a handcrafted or synthetic net, and a ledgered ruling.
  - Do not open any `seed2` file.
- [ ] **Step 2: Freeze.**
  - Run `uv run pytest -q`. Expected: all pass.
  - Commit the code: `feat: step-3 tool frozen for held-out run`.
  - Write `git rev-parse --short HEAD` to `runs/ast_freeze.txt` (FREEZE).
- [ ] **Step 3: Held-out run.** `uv run nndc ast --all --split heldout` (6 runs), then `uv run nndc report`. The tool code must not change after this step. If a crash occurs:
  - record it as a held-out failure
  - fix it only in a new commit
  - report both numbers: before and after the fix
- [ ] **Step 4: Write up.**
  - Regenerate `REPORT.md` with `uv run nndc report`. It shows the freeze line from `runs/ast_freeze.txt` (Task 7).
  - Add a README paragraph with the held-out headline: X of 6 reproduced, plus the list of what only Claude managed.
- [ ] **Step 5: Commit** `data: step-3 ast tool results (in-sample and held-out)`.
