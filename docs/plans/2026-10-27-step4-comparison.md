# Step 4: comparison with existing methods, and decompile-fix-reinsert — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the repo's two methods (the blind Claude agent and the AST tool) against runnable
representatives of existing approaches (B1–B5) on 41 nets. Then run the user's
decompile → fix → reinsert experiment (R) against weight-space repair.

**Architecture:**
- **Baselines** live in `nndc/baselines/` (tree, regions, PySR). Each writes one more program version
  (`tree.py`, `regions.py`, `pysr.py`, `pysr_snap.py`) into a run folder, and `evaluate_run` checks and
  grades it like `ast.py`.
- **B5** reuses the agent prepare/collect machinery with a new folder kind `io`: samples instead of the
  dump, an obfuscated oracle, and a checker without net-derived boundary points.
- **Experiment R** lives in `nndc/repair/`:
  - repair/held-out data;
  - metrics (efficacy, generalization, drawdown);
  - weight-space arms;
  - the program arms, which come back from blind folders of kind `fix` / `fixio`.
- `nndc/compile_back.py` turns a fixed program into a ReLU MLP.
- **Planted-bug nets** are trained into `runs_bugs/` by `experiments/defects.py`.
- **Report:** `nndc/compare.py` adds the "Comparison (step 4)" section to `REPORT.md`.

**Tech Stack:**
- Python 3.13, NumPy, PyTorch (CPU), pytest
- new group `baselines`: scipy (HiGHS LP), scikit-learn, sympy
- new optional group `pysr`: PySR, Julia via juliacall

**Spec:** `docs/design/2026-10-27-step4-comparison-design.md` (literature: `reports/Neural net decompilation prior work.md`, `reports/Counterexample guided program refinement.md`, `reports/Decompile fix and reinsert network blocks.md`).

## Global Constraints

- **Populations:**
  - `runs/` (29 formula nets);
  - `runs_datasets/` (12 Iris nets: iris and iris_petal, seeds 0–5);
  - `runs_bugs/` (5 planted-bug nets, created in Task 9).
- **Hyperparameters are fixed by the design and copied here.** No baseline is tuned per task or after
  seeing results.
  - samples: 50,000 per model, seed 0
  - tree sweep: `max_leaf_nodes` ∈ {2, 4, …, 512}
  - PySR:
    - binary operators `+ - * max min greater`, plus `cond` if available
    - unary operators `abs relu`
    - `maxsize=25`, `niterations=40`, `timeout_in_seconds=120`, `deterministic=True`,
      `parallelism="serial"`, `random_state=0`
    - 5,000 samples
  - PySR+snap tolerances: 5% relative for multiplicative coefficients; 1% of the relevant output or
    input range for additive constants and thresholds
  - regions budget: 4,096 regions or 60 s
  - B5: 2,000 samples
  - defect: a 10% sub-box with +20% of output range, or a flipped label
  - repair data: repair 2,000, held-out 20,000, out-of-range 20,000
  - fine-tune: Adam, lr 1e-3, 300 steps, full batch; anchors 2,000
  - special-case flag: an added test with support < 5 repair points
- **Selection parity:** tree and PySR+snap keep the **smallest program that passes the checker**, using
  the AST tool's rule: fewest nodes first, exactness breaks ties. Raw PySR uses `get_best()`.
- **Blindness:**
  - No baseline or agent folder ever contains the task name, real input names or `expected`.
  - Non-agent baselines see only the blind spec and samples of the net.
  - Fix folders contain true labels by design, but no task name.
- **Out of bounds:**
  - The AST tool's code (`nndc/ast_tool.py`, `nndc/heuristics/`) is not modified in step 4.
  - Existing `agent_n` and `ast` results are reused, never re-run.
- **One attempt per agent run.** Each run is one fresh blind general-purpose subagent, collected with
  `nndc agent collect --transcript <subagent jsonl>`.
- **Run errors:** a baseline that fails on one run records the error in `<name>_info.json`. It never
  stops a batch (as `nndc ast` does).

## Review Focus

1. **A baseline program the language rejects** must be recorded as invalid with its reason, and the
   batch goes on. Examples: a PySR expression with a float power, a tree leaf with NaN. Test in Task 5.
2. **Programs with assignments in only some branches, or early returns in nested ifs**, must keep their
   meaning when compiled back into a net. Test in Task 8.
3. **All-discrete tiny domains (xor: 4 points)** have no separate held-out set. Generalization must be
   reported as `None`, not as 0 or 1. Test in Task 10.
4. **An infeasible or oversized LP repair** returns `None` and is recorded as "infeasible". It is never
   a crash or a silently unrepaired net. Test in Task 11.
5. **The io folder must not leak the net.** It has no `dump.py` or `weights.npz`, no task name,
   `oracle.bin` is not loadable by `np.load` as is, and `check.py` uses no boundary points. Test in
   Task 7.

---

### Task 1: Dependencies and the PySR feasibility spike

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/notes/2026-10-27-pysr-spike.md`

**Interfaces:**
- Produces:
  - `uv run` includes scipy, scikit-learn and sympy (`baselines` is in `default-groups`);
  - `uv run --group pysr` adds PySR.

- [ ] **Step 1: Add the dependency groups**
  - `baselines = ["scipy>=1.14", "scikit-learn>=1.6", "sympy>=1.13"]`
  - `pysr = ["pysr>=1.5"]`
  - `[tool.uv] default-groups = ["dev", "baselines"]`

- [ ] **Step 2: Install and run the existing tests**

Run: `uv sync && uv run pytest -q`
Expected: all 154+ tests pass.

- [ ] **Step 3: PySR spike**

Run: `uv sync --group pysr`, then a warm-up `uv run --group pysr python -c "import pysr"`. The first
call downloads Julia; it is not timed. Then fit `absdiff/hand` with the Global Constraints settings on
5,000 samples from `nndc.baselines.common.net_samples`, or an inline equivalent if Task 2 is not done
yet.

- [ ] **Step 4: Record the outcome**

Write `docs/notes/2026-10-27-pysr-spike.md`:
- installed version;
- whether `cond` exists;
- warm-up time and fit time;
- the best equation.

**Decision rule (from the design):** if installation fails, or no equation comes back within 120 s,
B1/B2 are dropped. Note this in the file, and skip Task 5 and the PySR part of Task 14.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock docs/notes/2026-10-27-pysr-spike.md
git commit -m "chore: baseline dependency groups; PySR feasibility spike"
```

### Task 2: Shared samples, smallest-passing selection, new program versions

**Files:**
- Create: `nndc/baselines/__init__.py`, `nndc/baselines/common.py`
- Modify: `nndc/run.py` (`evaluate_run` version list)
- Test: `tests/test_baselines_common.py`

**Interfaces:**
- Produces:
  ```python
  # nndc/baselines/common.py
  N_SAMPLES = 50_000
  BASELINE_VERSIONS = ["tree", "regions", "pysr", "pysr_snap"]
  def net_samples(spec: Spec, mlp: MLP, n: int = N_SAMPLES, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]
      # X (n, d) uniform over the box (discrete inputs: integers; the full grid when domain_size <= n),
      # raw: (n,) value | (n,) score out-threshold | (n, k) logits; dec: (n,) decision (value / 0-1 / class)
  def smallest_passing(cands: list[str], mlp: MLP, spec: Spec) -> tuple[int | None, list[dict]]
      # index of the passing candidate with key (nodes, not exact_pass); per-candidate
      # {"snap_pass", "exact_pass", "nodes", "error"}; None when none pass
  def write_info(rd: Path, name: str, info: dict) -> None   # <name>_info.json
  ```
- `evaluate_run` also evaluates `BASELINE_VERSIONS` files and `io_agent_<n>.py` (regex
  `^io_agent_(\d+)\.py$`) when present. `fix_*` / `fixio_*` files are **not** versions; they are
  evaluated by `nndc.repair`.

- [ ] **Step 1: Write the failing tests**

```python
def test_net_samples_shapes_and_grid():
    mlp, spec = hand("xor")          # bool domain of 4 points -> full grid
    X, raw, dec = net_samples(spec, mlp, n=50)
    assert X.shape == (4, 2) and set(dec) <= {0.0, 1.0}
    mlp, spec = hand("absdiff")
    X, raw, dec = net_samples(spec, mlp, n=1000)
    assert X.shape == (1000, 2) and np.allclose(dec, np.abs(X[:, 0] - X[:, 1]), atol=1e-5)

def test_smallest_passing_prefers_fewest_nodes():
    mlp, spec = hand("absdiff")
    big = "def net(x0, x1):\n    return relu(x0 - x1) + relu(x1 - x0)\n"
    small = "def net(x0, x1):\n    return abs(x0 - x1)\n"
    wrong = "def net(x0, x1):\n    return x0\n"
    i, rows = smallest_passing([big, wrong, small], mlp, spec)
    assert i == 2 and rows[1]["snap_pass"] is False

def test_evaluate_run_picks_up_baseline_versions(tmp_path):
    rd = build_run("absdiff", None, tmp_path)
    (rd / "tree.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    (rd / "io_agent_1.py").write_text("def net(x0, x1):\n    return x0\n")
    res = evaluate_run(rd)
    assert res["versions"]["tree"]["check"]["exact_pass"] and "io_agent_1" in res["versions"]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_baselines_common.py -v`
Expected: FAIL (`ModuleNotFoundError: nndc.baselines`).

- [ ] **Step 3: Implement `nndc/baselines/common.py` and extend `evaluate_run`**

`net_samples` uses `experiments.tasks.uniform_sample` and `nndc.check.net_output`. For argmax, `raw` is
`mlp.forward` itself.

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/baselines nndc/run.py tests/test_baselines_common.py
git commit -m "feat: shared baseline samples, smallest-passing selection, baseline program versions"
```

### Task 3: B3 decision-tree distillation and `nndc baseline`

**Files:**
- Create: `nndc/baselines/tree.py`
- Modify: `nndc/cli.py` (new `baseline` command)
- Test: `tests/test_baselines_tree.py`

**Interfaces:**
- Consumes: `net_samples`, `smallest_passing`, `write_info` (Task 2).
- Produces:
  ```python
  LEAF_SWEEP = [2, 4, 8, 16, 32, 64, 128, 256, 512]
  def tree_to_program(tree, spec: Spec) -> str
      # fitted sklearn tree -> nested `if x_i <= t:` / `else:` source.
      # Leaves: value (regression), 1/0 (binary; assert 0 <= spec.threshold < 1), class index (argmax).
  def run_tree(rd: Path) -> dict    # writes tree.py and tree_info.json
      # {"leaves", "passed", "sweep": [per-size rows], "seconds"}
  ```
  CLI: `nndc baseline {tree,regions,pysr,all} [runs...] [--all] [--root R]`. Per-run errors are caught
  and printed (`ValueError`, `KeyError`, `OSError`, `ProgramError`), as in `cmd_ast`.
- Fitting:
  - `DecisionTreeRegressor` on the value (regression);
  - `DecisionTreeClassifier` on the net's decision (binary/argmax);
  - `random_state=0`.
- If no tree in the sweep passes, write the 512-leaf tree and set `passed: false`.

- [ ] **Step 1: Write the failing tests**

```python
def test_tree_program_matches_sklearn_predict():
    mlp, spec = hand("absdiff")
    X, raw, dec = net_samples(spec, mlp, n=2000)
    t = DecisionTreeRegressor(max_leaf_nodes=16, random_state=0).fit(X, dec)
    prog = parse_program(tree_to_program(t, spec), spec.arg_names)
    assert np.allclose(evaluate(prog, X), t.predict(X))

def test_window_tree_passes_with_few_leaves(tmp_path):
    rd = build_run("window", None, tmp_path)
    info = run_tree(rd)
    assert info["passed"] and info["leaves"] <= 4
    assert evaluate_run(rd)["versions"]["tree"]["check"]["snap_pass"]

def test_cli_baseline_survives_a_bad_run(tmp_path, capsys):
    rd = build_run("window", None, tmp_path)
    (tmp_path / "broken" / "seed0").mkdir(parents=True)
    (tmp_path / "broken" / "seed0" / "spec.json").write_text("{}")
    assert main(["baseline", "tree", "--all", "--root", str(tmp_path)]) == 0
    assert "ERROR" in capsys.readouterr().out
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_baselines_tree.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement `tree.py` and the CLI command**

Walk `tree_.children_left/right/threshold/feature/value`. For regression leaves use the float64 value;
for classification leaves use the argmax class. Print a constant `<= t` with `repr(float(t))`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_baselines_tree.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/baselines/tree.py nndc/cli.py tests/test_baselines_tree.py
git commit -m "feat: B3 decision-tree distillation baseline and nndc baseline command"
```

### Task 4: B4 exact activation-region decomposition

**Files:**
- Create: `nndc/baselines/regions.py`
- Test: `tests/test_baselines_regions.py`

**Interfaces:**
- Produces:
  ```python
  MAX_REGIONS = 4096; MAX_SECONDS = 60.0; MIN_SLACK = 1e-9
  def regions_program(mlp: MLP, spec: Spec, max_regions: int = MAX_REGIONS,
                      max_seconds: float = MAX_SECONDS) -> tuple[str | None, dict]
      # (source or None when over budget, {"regions", "lp_calls", "budget_exceeded", "seconds"})
  def run_regions(rd: Path) -> dict   # writes regions.py (if any) and regions_info.json
  ```
  The CLI name `regions` is wired into `nndc baseline`.

**Algorithm** (the signature does not fix it):
- Depth-first over units in topological order (layer by layer, index order). Each unit's pre-activation
  is an affine function of the inputs inside the current region.
- Split each unit into `a·x + c > 0` (active: the unit is that affine) and `<= 0` (the unit is 0).
- A child is kept if the LP "maximise s subject to the region's half-spaces with slack s, the box,
  s <= 1" (`scipy.optimize.linprog`, `method="highs"`) gives s > `MIN_SLACK`. This means a non-empty
  interior.
- A unit that is always positive or always non-positive in the region adds no branch.
- Leaves return the output affine:
  - regression: the value;
  - binary: the raw output, as in `dump.py`;
  - argmax: `argmax(a0, a1, ...)`.
- Emitted as nested `if <affine> > 0:` / `else:` with float64 shortest-repr constants.
- Integer inputs use the real relaxation (accepted, noted in info).

- [ ] **Step 1: Write the failing tests**

```python
def test_absdiff_regions_exact():
    mlp, spec = hand("absdiff")
    src, info = regions_program(mlp, spec)
    assert info["regions"] >= 2 and check_program(src, mlp, spec).exact_pass

def test_iris_like_argmax_regions_exact():
    mlp = random_mlp([2, 6, 3], seed=4)
    spec = Spec(task="", output="argmax", n_outputs=3,
                inputs=[InputSpec("x0", "x0", "float", -1, 1), InputSpec("x1", "x1", "float", -1, 1)])
    src, _ = regions_program(mlp, spec)
    assert check_program(src, mlp, spec).exact_pass

def test_budget_exceeded_gives_no_program():
    mlp = random_mlp([3, 16, 16, 1], seed=0)
    spec = Spec(task="", output="regression", inputs=[InputSpec(f"x{i}", f"x{i}", "float", -3, 3) for i in range(3)])
    src, info = regions_program(mlp, spec, max_regions=4)
    assert src is None and info["budget_exceeded"]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_baselines_regions.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `regions.py` and add `regions` to the CLI**

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_baselines_regions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/baselines/regions.py nndc/cli.py tests/test_baselines_regions.py
git commit -m "feat: B4 exact activation-region decomposition baseline"
```

### Task 5: B1/B2 PySR, raw and snapped (skip if the spike failed)

**Files:**
- Create: `nndc/baselines/pysr_run.py`
- Test: `tests/test_baselines_pysr.py`. The printer tests use sympy only. One fit test is marked
  `@pytest.mark.slow` and skipped unless `pysr` imports.

**Interfaces:**
- Consumes: `net_samples`, `smallest_passing` (Task 2), `nndc.heuristics.snap.nice` (read-only use).
- Produces:
  ```python
  N_PYSR = 5000
  PYSR_SETTINGS: dict   # exactly the Global Constraints values; "cond" appended when available
  class UnsupportedExpr(ValueError)
  def sympy_to_program(expr, spec: Spec) -> str
  def snap_expr(expr, out_range: float, in_range: float)   # -> sympy expr
  def run_pysr(rd: Path) -> dict   # writes pysr.py, pysr_snap.py (if any pass), pysr_info.json
  ```
- **Printer mapping:**
  - `Add`, `Mul` → `+`, `*`
  - `Abs` → `abs`
  - `Max`/`Min` → `max`/`min`
  - `Piecewise((a, c), (b, True))` → `(a if c else b)`
  - relationals → comparisons
  - `Pow` with a positive integer exponent → repeated `*`
  - `Rational`/`Float`/`Integer` → a float literal
  - `Symbol x_k` → `xk`
  - anything else → `UnsupportedExpr`, recorded in info
- **Snapping, by where each Float sits:**
  - a factor of a `Mul`: `nice(v, 0.05*|v|)`;
  - a term of an `Add`: `nice(v, 0.01*out_range)`;
  - an argument of a relational or `greater`: `nice(v, 0.01*in_range)`, where `in_range` is the
    largest input span.
- **Binary nets:** fit the score `out − threshold`; the program returns the score.
- **B2:** apply `snap_expr` to every Pareto equation, then `smallest_passing`.

- [ ] **Step 1: Write the failing tests**

```python
def two_float_spec():
    return Spec(task="", output="regression",
                inputs=[InputSpec("x0", "x0", "float", -3, 3), InputSpec("x1", "x1", "float", -3, 3)])

def test_printer_handles_piecewise_and_max():
    x0, x1 = sympy.symbols("x0 x1")
    e = sympy.Max(x0, x1) + sympy.Piecewise((1.0, x0 > 0.5), (0.0, True)) + 2 * sympy.Abs(x1) ** 2
    src = sympy_to_program(e, two_float_spec())
    prog = parse_program(src, ["x0", "x1"])
    X = np.array([[1.0, -2.0], [0.0, 3.0]])
    assert np.allclose(evaluate(prog, X), [1 + 1 + 8, 3 + 0 + 18])

def test_printer_rejects_float_power():
    x0 = sympy.Symbol("x0")
    with pytest.raises(UnsupportedExpr):
        sympy_to_program(x0 ** 0.5, two_float_spec())

def test_snap_expr_by_position():
    x0 = sympy.Symbol("x0")
    e = snap_expr(0.987 * x0 + 0.0031, out_range=6.0, in_range=6.0)
    assert e == x0   # 0.987 -> 1 (5% rel), 0.0031 -> 0 (1% of 6)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_baselines_pysr.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `pysr_run.py` and add `pysr` to the CLI**

The import of `pysr` is inside `run_pysr`. `nndc baseline pysr` needs `uv run --group pysr`.

- [ ] **Step 4: Run the tests (and the slow fit test once)**

Run: `uv run pytest tests/test_baselines_pysr.py -v` then `uv run --group pysr pytest -m slow tests/test_baselines_pysr.py -v`
Expected: PASS. The slow test fits `absdiff/hand` and gets a valid `pysr.py`.

- [ ] **Step 5: Commit**

```bash
git add nndc/baselines/pysr_run.py nndc/cli.py tests/test_baselines_pysr.py
git commit -m "feat: B1/B2 PySR baselines with sympy printer and positional snapping"
```

### Task 6: Checker option without boundary points

**Files:**
- Modify: `nndc/check.py`
- Test: `tests/test_check.py`

**Interfaces:**
- Produces:
  - `build_input_set(mlp, spec, seed=0, extra=None, boundary: bool = True)`
  - `check_program(src, mlp, spec, seed=0, extra_inputs=None, boundary: bool = True)`
  - With `boundary=False`, the input set is uniform + grid + corners only. The default is unchanged.

- [ ] **Step 1: Write the failing test**

```python
def test_no_boundary_points_option():
    mlp, spec = hand("window")
    full, _ = build_input_set(mlp, spec)
    plain, _ = build_input_set(mlp, spec, boundary=False)
    assert plain.shape[0] < full.shape[0]
    r = check_program("def net(x0):\n    return abs(x0) < 0.5\n", mlp, spec, boundary=False)
    assert r.snap_pass
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/test_check.py::test_no_boundary_points_option -v`
Expected: FAIL (`TypeError: unexpected keyword 'boundary'`).

- [ ] **Step 3: Add the parameter**

`_boundary_points` is skipped when `boundary` is false.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS. Existing results are unchanged because the default is `True`.

- [ ] **Step 5: Commit**

```bash
git add nndc/check.py tests/test_check.py
git commit -m "feat: checker option without net-derived boundary points"
```

### Task 7: B5 input/output-only agent folders, and agent folder kinds

**Files:**
- Create: `nndc/agent/oracle.py`, `nndc/agent/check_io_template.py`, `nndc/agent/instruction_io.md`
- Modify: `nndc/agent/prepare.py`, `nndc/agent/collect.py`, `nndc/cli.py`, `nndc/report.py` (`_best_attempt` ignores non-`agent_` kinds; already true by prefix — add a test)
- Test: `tests/test_agent_io.py`

**Interfaces:**
- Consumes: `check_program(..., boundary=False)` (Task 6).
- Produces:
  ```python
  # nndc/agent/oracle.py   (stdlib + numpy; copied into _checker/)
  def encode(data: bytes, key: str) -> bytes    # XOR with the SHA-256 counter-mode keystream of key
  def decode(data: bytes, key: str) -> bytes
  # nndc/agent/prepare.py
  KINDS = {"agent": "agent", "io": "io_agent", "fix": "fix", "fixio": "fixio"}   # kind -> file prefix in the run folder
  def prepare(rd: Path, blind_root: Path = BLIND_ROOT, index: Path = INDEX, kind: str = "agent",
              program: Path | None = None) -> Path
  # nndc/agent/collect.py
  def collect(blind, index=INDEX, heuristics=..., projects=PROJECTS, transcripts=None) -> tuple[Path, int | None]
      # reads kind from the index entry (default "agent"); writes <prefix>_<n>.py/_notes.md/_checks.log/_meta.json;
      # kinds agent/io then evaluate_run; kinds fix/fixio then nndc.repair.evaluate_repair (Task 12);
      # only kind "agent" appends to HEURISTICS.md; rollback removes <prefix>_<n>* on failure
  def audit_transcripts(paths, blind, forbidden: tuple[str, ...] = ()) -> dict   # adds "forbidden_paths"
  ```
- **The io folder holds exactly:**
  - `spec.json` (blind);
  - `samples.csv`: 2,000 rows, seed 0, uniform over the box. Header `x0,…,out` (regression, or binary
    score + `class`), or `x0,…,out0,…,outk,class` (argmax);
  - `INSTRUCTIONS.md`;
  - `check.py` and `_checker/` (`spec.py`, `model.py`, `lang.py`, `check.py`, `oracle.py`);
  - `oracle.bin`: `encode(weights.npz bytes, key=folder name)`.

  `check.py` decodes it in memory and calls `check_program(..., boundary=False)`. For kind `io`, the
  audit uses `forbidden=("oracle.bin", "_checker")`.
- **`instruction_io.md`:** the same intent as `instruction.md`, rewritten. "This is a black-box function
  given by samples in `samples.csv`. Write a short readable program that reproduces it…". The same
  practical details, and "Use only `samples.csv`, `spec.json` and `check.py`; do not open `oracle.bin`
  or `_checker/`."
- CLI: `nndc agent prepare --io <runs…>` (sets kind `io`).

- [ ] **Step 1: Write the failing tests**

```python
def test_io_folder_contents_do_not_leak(tmp_path):
    rd = build_run("window", None, tmp_path / "runs")
    b = prepare(rd, blind_root=tmp_path / "blind", index=tmp_path / "idx.json", kind="io")
    names = {p.name for p in b.iterdir()}
    assert {"samples.csv", "oracle.bin", "check.py", "spec.json", "INSTRUCTIONS.md"} <= names
    assert not names & {"dump.py", "weights.npz"}
    assert "window" not in (b / "spec.json").read_text()
    with pytest.raises(Exception):
        np.load(b / "oracle.bin")
    assert len((b / "samples.csv").read_text().splitlines()) == 2001

def test_io_check_runs_without_boundary_points(tmp_path):
    rd = build_run("window", None, tmp_path / "runs")
    b = prepare(rd, blind_root=tmp_path / "blind", index=tmp_path / "idx.json", kind="io")
    (b / "agent.py").write_text("def net(x0):\n    return abs(x0) < 0.5\n")
    out = subprocess.run([sys.executable, "-I", "check.py", "agent.py"], cwd=b, capture_output=True, text=True)
    assert out.stdout.startswith("PASS")

def test_collect_io_kind_names_and_forbidden_audit(tmp_path):
    # mock transcript mentioning oracle.bin -> meta.forbidden_paths non-empty; file io_agent_1.py; HEURISTICS untouched
    ...
    assert (rd / "io_agent_1.py").exists() and meta["forbidden_paths"] and not heur.exists()
```

The third test writes a one-line JSONL transcript whose tool input `file_path` is `<blind>/oracle.bin`,
and passes `transcripts=[that file]` and `heuristics=tmp_path / "H.md"`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_agent_io.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the oracle, the template, the instruction and the kind-aware prepare/collect**

Existing `agent` folders and collects must behave byte-for-byte as before. `tests/test_agent.py` must
pass unchanged.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/agent nndc/cli.py nndc/report.py tests/test_agent_io.py
git commit -m "feat: B5 input/output-only agent folders with obfuscated oracle; kind-aware prepare/collect"
```

### Task 8: Compile a program back into a ReLU MLP

**Files:**
- Create: `nndc/compile_back.py`
- Test: `tests/test_compile_back.py`

**Interfaces:**
- Produces:
  ```python
  class CompileError(ValueError)
  RAMP_REL = 1e-6
  def compile_program(src: str, spec: Spec) -> tuple[MLP, dict]
      # info {"exact": bool, "ramps": int, "hidden": [layer widths]}; MLP float32 weights
  ```

**Algorithm** (the tests do not fix it):

1. **Flatten statements to one expression.** Walk the body with an environment of name → expression.
   - `if c: A else: B`, where both branches return, becomes `IfExp(c, A, B)`.
   - A branch that falls through continues with the following statements.
   - A variable assigned in only some branches becomes `IfExp(c, new, old)`.
2. **Value compile** (expression → a DAG of affine and ReLU nodes):
   - numbers, names, `+ −`, multiplication or division by a constant, `sum`: affine
   - `relu(e)`: ReLU
   - `max(a, b)`: `b + relu(a − b)`, folded left over more arguments
   - `min(a, b)`: `a − relu(a − b)`
   - `abs(e)`: `relu(e) + relu(−e)`
   - the product of two non-constant expressions: `CompileError`
   - a comparison or boolean in value context: `step(margin(c))`
   - `IfExp(c, a, b)`: `b + gate(step(margin(c)), a − b)`
   - `step(m) = relu(m/ε) − relu(m/ε − 1)` with `ε = RAMP_REL × width(interval(m))`. Each step counts
     as one ramp.
   - `gate(s, d) = relu(d − M(1 − s)) − relu(−d − M(1 − s))` with `M = max|interval(d)|`. Intervals
     come from interval arithmetic over the spec box.
3. **Margin compile** (boolean context; positive means true):
   - `a < b`, `a <= b`: `b − a`
   - `a > b`, `a >= b`: `a − b`
   - on an all-integer pair, `x == c` becomes `0.5 − abs(x − c)` and `!=` becomes `abs(x − c) − 0.5`;
     otherwise `==` / `!=` raise `CompileError`
   - `and`: `min`
   - `or`: `max`
   - `not`: negation
   - `True` / `False`: `±1`
   - `IfExp(c, A, B)`: `max(min(m(c), m(A)), min(−m(c), m(B)))`
4. **Outputs:**
   - **regression:** the value.
   - **binary:** if the returned expression is boolean-shaped (comparisons/and/or/not, or `IfExp`
     leaves that are 0/1 constants or booleans), the output is `threshold + margin`. Otherwise it is
     the value.
   - **argmax:**
     - `argmax(e0, …, ek)` outputs `e0 … ek` as values;
     - an `IfExp` tree with integer-constant leaves gives `score_k = margin` of the `or` over the paths
       that reach leaf k. A class with no leaf gets the constant `−M_big`, where `M_big = 1 + max` over
       the other scores' interval bounds.
5. **Layering:**
   - A ReLU node's depth is 1 + the largest depth of the ReLU nodes its argument uses.
   - Hidden layer L holds the depth-L ReLU nodes, plus pass-through pairs `relu(v)`, `relu(−v)` for
     every affine value a deeper layer still needs.
   - The output layer is the final affine.
   - `info["exact"] = (ramps == 0)`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("task, body", [
    ("absdiff", "return abs(x0 - x1)"),
    ("median3", "return max(min(x0, x1), min(max(x0, x1), x2))"),
    ("tax", "return 0.1 * min(x0, 20) + 0.2 * min(max(x0 - 20, 0), 60) + 0.4 * max(x0 - 80, 0)"),
    ("window", "return abs(x0) < 0.5"),
    ("xor", "return x0 != x1"),
])
def test_compiled_net_matches_program_exactly(task, body):
    spec = Spec.from_json(get_task(task).spec.blind())
    src = f"def net({', '.join(spec.arg_names)}):\n    {body}\n"
    mlp, info = compile_program(src, spec)
    assert info["exact"] and check_program(src, mlp, spec).exact_pass

def test_decision_list_argmax_exact():
    spec = Spec.from_json(get_task("iris_petal").spec.blind())
    src = ("def net(x0, x1):\n    if x0 + x1 < 3.25:\n        return 0\n"
           "    if x0 + 1.125 * x1 < 6.8:\n        return 1\n    return 2\n")
    mlp, info = compile_program(src, spec)
    assert info["exact"] and check_program(src, mlp, spec).exact_pass

def test_branch_assignment_and_fallthrough():
    spec = Spec.from_json(get_task("absdiff").spec.blind())
    src = ("def net(x0, x1):\n    d = x0 - x1\n    if d < 0:\n        d = -d\n"
           "    if x0 > 2.5 or x1 > 2.5:\n        return d < 1\n    return d < 0.5\n")
    spec_b = Spec(task="", output="binary", inputs=spec.inputs)
    mlp, _ = compile_program(src, spec_b)
    assert check_program(src, mlp, spec_b).exact_pass

def test_regression_jump_uses_one_ramp():
    spec = Spec.from_json(get_task("tax").spec.blind())
    mlp, info = compile_program("def net(x0):\n    return 10 if x0 > 100 else 0\n", spec)
    assert info["ramps"] == 1 and not info["exact"]
    assert np.allclose(mlp.forward(np.array([[99.0], [101.0]], np.float32))[:, 0], [0, 10], atol=1e-3)

def test_product_of_variables_rejected():
    spec = Spec.from_json(get_task("absdiff").spec.blind())
    with pytest.raises(CompileError):
        compile_program("def net(x0, x1):\n    return x0 * x1\n", spec)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_compile_back.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `compile_back.py`**

It reuses `nndc.lang.parse_program` for validation and walks `prog`'s AST.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_compile_back.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/compile_back.py tests/test_compile_back.py
git commit -m "feat: compile a fixed program back into a ReLU MLP (exact for piecewise-linear and decisions)"
```

### Task 9: Planted-bug nets

**Files:**
- Create: `experiments/defects.py`
- Modify: `experiments/train.py` (`train(task, seed, target=None)`), `nndc/run.py` (`build_run(..., bug: bool = False)`), `nndc/cli.py` (`nndc run --bug`)
- Test: `tests/test_defects.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class Defect:
      kind: str                 # "offset" (regression) | "flip" (binary)
      box_lo: list[float]; box_hi: list[float]   # equal for a single discrete point
      offset: float             # 0.0 for flips
      def inside(self, X: np.ndarray) -> np.ndarray
      def to_json(self) -> dict
  def make_defect(task: Task, seed: int) -> Defect
  def defective_target(task: Task, d: Defect) -> Callable[[np.ndarray], np.ndarray]
  ```
- **The defect sub-box:**
  - side `0.1 ** (1/d)` of each input span;
  - lower corner uniform in `[lo, hi − side]` with `np.random.default_rng(seed)`;
  - regression offset = `0.2 × (max − min)` of the true target on 20,000 uniform samples (seed 0);
  - all-discrete domains (xor): one domain point chosen with `rng.integers`, label flipped.
- **`build_run(task, seed, root, retrain, bug=True)`:**
  - trains on `defective_target` and keeps `spec.task` = the original task, so grading measures the
    truth;
  - writes `defect.json`;
  - adds `"defect"` to `train_log.json`, plus `"bug_learned"`: the share of 20,000 training-distribution
    points inside the defect where the net agrees with the defective target rather than the truth.
- CLI: `nndc run TASK --seed S --bug --root runs_bugs` (defect seed = S).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("name, d", [("tax", 1), ("absdiff", 2), ("median3", 3)])
def test_defect_box_is_ten_percent(name, d):
    t = get_task(name); df = make_defect(t, 0)
    span = np.array([i.hi - i.lo for i in t.spec.inputs])
    vol = np.prod(np.array(df.box_hi) - np.array(df.box_lo)) / np.prod(span)
    assert abs(vol - 0.1) < 1e-9

def test_defective_target_differs_only_inside():
    t = get_task("absdiff"); df = make_defect(t, 0)
    X = uniform_sample(t.spec, np.random.default_rng(1), 5000)
    diff = defective_target(t, df)(X) != t.target(X)
    assert (diff == df.inside(X)).all()

def test_xor_flips_one_point():
    t = get_task("xor"); df = make_defect(t, 0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    assert (defective_target(t, df)(X) != t.target(X)).sum() == 1

def test_build_run_bug_keeps_true_task(tmp_path):
    rd = build_run("xor", 0, tmp_path, bug=True)
    assert Spec.load(rd / "spec.json").task == "xor" and (rd / "defect.json").exists()
    assert json.loads((rd / "train_log.json").read_text())["bug_learned"] > 0.5
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_defects.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `defects.py`, the `target` parameter and `bug` in `build_run` and the CLI**

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/defects.py experiments/train.py nndc/run.py nndc/cli.py tests/test_defects.py
git commit -m "feat: planted-bug nets (10% sub-box offset or flipped label) for the fix experiment"
```

### Task 10: Repair data, metrics, the special-case flag, and the R-tree arm

**Files:**
- Create: `nndc/repair/__init__.py`, `nndc/repair/data.py`, `nndc/repair/metrics.py`
- Modify: `experiments/grade.py` (rename `_out_of_range` → `out_of_range_inputs`, keep behaviour)
- Test: `tests/test_repair_metrics.py`

**Interfaces:**
- Produces:
  ```python
  # data.py
  REPAIR_N, HELDOUT_N, OOR_N = 2000, 20_000, 20_000
  REPAIR_SEED, HELDOUT_SEED, OOR_SEED, ANCHOR_SEED = 100, 101, 102, 103
  SMALL_DOMAIN = 16
  @dataclass
  class RepairData:
      X_repair: np.ndarray; y_repair: np.ndarray
      X_heldout: np.ndarray; y_heldout: np.ndarray
      X_oor: np.ndarray | None; y_oor: np.ndarray | None
      flowers: bool             # P3: the held-out set is the 50 held-out flowers
      whole_domain: bool        # domain_size <= SMALL_DOMAIN: repair = held-out = the domain
  def repair_data(rd: Path) -> RepairData
  # metrics.py
  def right(spec: Spec, pred: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray   # |pred-y|<=tol (regression) / equal
  def repair_metrics(spec: Spec, orig: np.ndarray, new: np.ndarray, y: np.ndarray, tol: float,
                     whole_domain: bool) -> dict
      # {"efficacy", "generalization", "drawdown", "accuracy" | "max_abs_err", "n_failing", "n_right"}
      # efficacy is on the arrays given; generalization is None when whole_domain
  def added_tests(orig_src: str, new_src: str, X: np.ndarray, arg_names: list[str]) -> list[dict]
      # comparisons in new_src (ast.unparse-normalised) not in orig_src;
      # {"test", "support": min(#true, #false) over X, "flagged": support < 5}
  def decide(spec: Spec, raw: np.ndarray) -> np.ndarray   # program/net raw -> decision (argmax over logits for nets)
  ```
- **Data sources:**
  - **Formula runs:** points from `task.sampler` or `uniform_sample`, labelled by `task.target`. The
    out-of-range set comes from `out_of_range_inputs`.
  - **Iris runs** (`get_task(spec.task)` in `DATASET_TASKS`): `table(name)`, split with
    `train_test_split(test_size=50, stratify=y, random_state=0)` into repair (100) and held-out (50).
    `flowers=True`, no out-of-range set.
- **`tol`:** `spec.snap_tol_frac × (max − min of y_heldout)` for regression; 0 for decisions.

- [ ] **Step 1: Write the failing tests**

```python
def test_metrics_counts():
    spec = Spec(task="", output="binary", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
    y    = np.array([1, 1, 0, 0, 1.])
    orig = np.array([1, 0, 0, 1, 1.])   # wrong on 1 and 3
    new  = np.array([1, 1, 1, 0, 1.])   # fixes 1, 3; breaks 2
    m = repair_metrics(spec, orig, new, y, 0.0, whole_domain=False)
    assert m["generalization"] == 1.0 and m["drawdown"] == pytest.approx(1 / 3)

def test_whole_domain_generalization_is_none(tmp_path):
    rd = build_run("xor", None, tmp_path)
    d = repair_data(rd)
    assert d.whole_domain and len(d.X_heldout) == 4
    m = repair_metrics(Spec.load(rd / "spec.json"), d.y_heldout, d.y_heldout, d.y_heldout, 0.0, d.whole_domain)
    assert m["generalization"] is None

def test_iris_split_is_stratified_and_disjoint():
    d = repair_data(Path("runs_datasets/iris_petal/seed0"))
    assert len(d.X_repair) == 100 and len(d.X_heldout) == 50 and d.flowers
    assert np.bincount(d.y_heldout.astype(int)).tolist() in ([17, 17, 16], [17, 16, 17], [16, 17, 17])

def test_added_test_support_flag():
    X = np.linspace(0, 1, 101)[:, None]
    orig = "def net(x0):\n    return x0 < 0.5\n"
    new = "def net(x0):\n    if x0 > 0.985:\n        return 1\n    return x0 < 0.5\n"
    rows = added_tests(orig, new, X, ["x0"])
    assert rows == [{"test": "x0 > 0.985", "support": 1, "flagged": True}]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_repair_metrics.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `data.py`, `metrics.py` and the rename in `grade.py`**

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/repair experiments/grade.py tests/test_repair_metrics.py
git commit -m "feat: repair/held-out data, efficacy/generalization/drawdown metrics, special-case flag"
```

### Task 11: Weight-space arms, the R-tree arm, and `nndc repair`

**Files:**
- Create: `nndc/repair/weights.py`, `nndc/repair/run.py`
- Modify: `nndc/cli.py` (`nndc repair baselines|evaluate <runs…> | --all --root R`)
- Test: `tests/test_repair_weights.py`

**Interfaces:**
- Consumes: `repair_data`, `repair_metrics`, `decide`, `added_tests` (Task 10); `compile_program`
  (Task 8); `experiments.train.train` (with `target`, Task 9).
- Produces:
  ```python
  # weights.py
  FT_STEPS, FT_LR, N_ANCHOR = 300, 1e-3, 2000
  def finetune(mlp: MLP, spec: Spec, X: np.ndarray, y: np.ndarray, seed: int = 0) -> MLP
      # full batch Adam; loss MSE / BCEWithLogits(out - threshold) / CrossEntropy
  def anchors(rd: Path, mlp: MLP, data: RepairData) -> tuple[np.ndarray, np.ndarray]
      # P1/P2: ANCHOR_SEED training-distribution points where the original net is right, labelled by the net;
      # P3: uniform box points labelled by the net
  def lp_repair(mlp: MLP, spec: Spec, X: np.ndarray, y: np.ndarray, tol: float) -> MLP | None
      # min t s.t. |dW|,|db| <= t on the last layer; regression |f + dW h + db - y| <= tol;
      # binary s*(f + dW h + db - thr) >= g with s = +-1, g = 1e-3 * p95|score|;
      # argmax (f_y - f_k) + (dW_y - dW_k) h + (db_y - db_k) >= g for k != y; HiGHS; infeasible -> None
  # run.py
  ARMS_W = ["w_ft", "w_ft_anchor", "w_lp", "w_retrain"]
  def run_repair_baselines(rd: Path) -> dict   # writes repair/<arm>.npz, repair/r_tree.py
  def evaluate_repair(rd: Path) -> dict
      # every arm present: w_* nets, r_tree.py, fix_<n>.py, fixio_<n>.py (program arms: also compile_back
      # -> repair/<name>_net.npz, reinsertion check = check_program(program, compiled net, spec));
      # writes repair/results.json:
      # {arm: {"repair": metrics on the repair set, "heldout": metrics, "oor": metrics | None,
      #        "nodes", "added_tests", "reinsertion": {"exact_pass", "snap_pass", "ramps"} | None,
      #        "error"}}
  ```
- **`w_retrain`:** only when `defect.json` exists. `train(task, seed)` on the true target.
- **`r_tree`:** for each leaf count in `LEAF_SWEEP`, 5-fold CV accuracy (or −MSE) on the repair set,
  `random_state=0`. Keep the best; ties go to fewer leaves. Refit on the whole repair set; the program
  comes from `tree_to_program`.
- **Iris repeated splits (design: one flower is 2 points of held-out accuracy):** for `flowers` runs,
  `evaluate_repair` also reruns the non-agent arms (`r_tree`, `w_ft`, `w_ft_anchor`, `w_lp`) on
  `RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0)` over the 150 flowers. It stores
  `"cv": {"generalization": [mean, sd], "drawdown": [mean, sd]}` per arm. Agent arms keep the single
  split.

- [ ] **Step 1: Write the failing tests**

```python
def test_lp_repair_fixes_points_and_reports_infeasible():
    mlp, spec = hand("window")
    X = np.array([[0.6], [0.0], [-1.5]]); y = np.array([1, 1, 0.])   # make 0.6 positive
    fixed = lp_repair(mlp, spec, X, y, 0.0)
    assert fixed is not None and (decide(spec, fixed.forward(X.astype(np.float32))) == y).all()
    X2 = np.array([[0.1], [0.1]]); y2 = np.array([1, 0.])            # contradictory labels
    assert lp_repair(mlp, spec, X2, y2, 0.0) is None

def test_finetune_moves_toward_labels():
    mlp, spec = hand("absdiff")
    X = np.random.default_rng(0).uniform(-3, 3, (200, 2)); y = np.abs(X[:, 0] - X[:, 1]) + 1
    new = finetune(mlp, spec, X, y)
    assert np.abs(new.forward(X.astype(np.float32))[:, 0] - y).mean() < 0.5

def test_evaluate_repair_end_to_end(tmp_path):
    rd = build_run("xor", 0, tmp_path, bug=True)
    run_repair_baselines(rd)
    (rd / "fix_1.py").write_text("def net(x0, x1):\n    return x0 != x1\n")
    res = evaluate_repair(rd)
    assert res["fix_1"]["heldout"]["accuracy"] == 1.0 and res["fix_1"]["reinsertion"]["exact_pass"]
    assert set(ARMS_W) <= set(res) and "r_tree" in res
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_repair_weights.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `weights.py`, `run.py` and the CLI**

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/repair nndc/cli.py tests/test_repair_weights.py
git commit -m "feat: weight-space repair arms (fine-tune, anchored, last-layer LP, retrain) and repair evaluation"
```

### Task 12: Fix-agent folders (`fix`, `fixio`)

**Files:**
- Create: `nndc/agent/instruction_fix.md`, `nndc/agent/instruction_fixio.md`, `nndc/agent/score_template.py`
- Modify: `nndc/agent/prepare.py` (kinds `fix`, `fixio`), `nndc/agent/collect.py` (fix kinds → `evaluate_repair`), `nndc/cli.py` (`nndc agent prepare --fix|--fixio`)
- Test: `tests/test_agent_fix.py`

**Interfaces:**
- Consumes: `prepare(..., kind, program)`, `collect` kinds (Task 7); `repair_data`, `evaluate_repair`
  (Tasks 10–11).
- Produces:
  - **The fix folder holds:**
    - `spec.json` (blind);
    - `program.py`: the faithful program, i.e. the `program` argument; omitted for `fixio`;
    - `repair.csv`: header `x0,…,net,label,failing`, the `RepairData.X_repair` rows. `net` is the
      original net's decision; `failing` is 1 where `net` is not right;
    - `score.py` plus `_checker/` (`spec.py`, `lang.py`);
    - `INSTRUCTIONS.md`.
  - **`score.py agent.py` prints exactly:**
    - `repair set: N points; your program is right on R`
    - `of the F points the network gets wrong, now right: A`
    - `of the K points the network gets right, now wrong: B`
  - It appends one line to `agent_checks.log` (same format as `check.py`).
  - **`instruction_fix.md`** (the user's idea in plain words): "`program.py` is a program decompiled
    from a neural network; it reproduces the network, including its mistakes. `repair.csv` lists
    inputs with the correct label and marks where the network is wrong. Fix the algorithm so it is
    right on these cases, keep it unchanged where the network is right, and keep it short and readable.
    Do not add special cases for single points." Then the same practical details as `instruction.md`,
    with `score.py` instead of `check.py`. `instruction_fixio.md` is the same without `program.py`
    ("write a program that is right on these cases").
  - **Program choice for `--fix`:** the run's best valid `agent_n` (`_best_attempt(valid_only=True)`),
    else `ast.py`.

- [ ] **Step 1: Write the failing tests**

```python
def test_fix_folder_and_score(tmp_path):
    rd = build_run("xor", 0, tmp_path / "runs", bug=True)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return x0 != x1 or (x0 == 1 and x1 == 1)\n")
    b = prepare(rd, blind_root=tmp_path / "b", index=tmp_path / "i.json", kind="fix", program=rd / "agent_1.py")
    assert (b / "program.py").exists() and "xor" not in (b / "spec.json").read_text()
    (b / "agent.py").write_text("def net(x0, x1):\n    return x0 != x1\n")
    out = subprocess.run([sys.executable, "-I", "score.py", "agent.py"], cwd=b, capture_output=True, text=True).stdout
    assert "now wrong: 0" in out

def test_fixio_folder_has_no_program(tmp_path):
    rd = build_run("xor", 0, tmp_path / "runs", bug=True)
    b = prepare(rd, blind_root=tmp_path / "b", index=tmp_path / "i.json", kind="fixio")
    assert not (b / "program.py").exists() and (b / "repair.csv").exists()

def test_collect_fix_runs_repair_evaluation(tmp_path):
    # prepare fix folder, write agent.py, collect with an empty transcript list -> fix_1.py and repair/results.json["fix_1"]
    ...
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_agent_fix.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the templates, instructions, prepare kinds and CLI flags**

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/agent nndc/cli.py tests/test_agent_fix.py
git commit -m "feat: blind fix-agent folders (with and without the decompiled program)"
```

### Task 13: Comparison section in REPORT.md, and README positioning

**Files:**
- Create: `nndc/compare.py`
- Modify: `nndc/report.py` (`build_report` appends `comparison_md()`), `README.md` ("Prior work" section)
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `results.json` versions (Tasks 2–7), `repair/results.json` (Task 11), `*_checks.log`,
  `*_meta.json`.
- Produces:
  ```python
  METHODS = [("Claude (dump)", "agent"), ("Claude (input/output only)", "io_agent"), ("AST tool", "ast"),
             ("PySR", "pysr"), ("PySR+snap", "pysr_snap"), ("tree", "tree"), ("regions", "regions")]
  def method_table(root: Path, split: str) -> str        # split "heldout" (seed2) | "insample" | "all"
  def recovery_matrix(root: Path) -> str
  def iris_table(root: Path) -> str                      # verdict, extra-seed disagreements, nodes, flower accuracy
  def loop_counts(roots: list[Path]) -> str              # per agent kind: runs, checks, FAILs before first PASS
  def fix_table(roots: list[Path]) -> str                # arm x medians; P1+P2 vs P3; flagged; reinsertion
  def what_this_shows(root: Path) -> list[str]           # the three computed lines (B5 vs dump, B2 vs B1, B4 size)
  def comparison_md(formula_root: Path = RUNS, iris_root: Path = Path("runs_datasets"),
                    bug_root: Path = Path("runs_bugs")) -> str
  ```
- **Rules:**
  - Claude (dump) uses the best valid `agent_n`; `io_agent` uses the best `io_agent_n` without
    `forbidden_paths`.
  - Audit-flagged runs are excluded from the method rows and counted in a footnote.
  - If more than 1/3 of io runs are flagged, the io row reads "invalid (audit)".
  - A missing version counts as "no program".
  - Extra-seed disagreements use `check_program(..., seed=s)` for s in 2, 3, 4, 5, computed in
    `iris_table`.
  - `fix_table` rows: `r_agent` (`fix_*`), `r_io` (`fixio_*`), `r_tree`, `w_ft`, `w_ft_anchor`, `w_lp`,
    `w_retrain`. Columns: efficacy, generalization, drawdown, task in/out, nodes, flagged added tests,
    reinsertion PASS.
  - Per-net rows show the faithful program next to the fixed one.
- **README:**
  - Replace the "Prior work" section with the positioning sentence from
    `reports/Neural net decompilation prior work.md`.
  - Add one line on the fix-and-reinsert gap, citing `reports/Decompile fix and reinsert network blocks.md`.
  - Add links to all three reports.

- [ ] **Step 1: Write the failing tests**

```python
def test_method_table_counts_and_no_program(tmp_path):
    rd = build_run("absdiff", None, tmp_path)
    (rd / "tree.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n"); evaluate_run(rd)
    md = method_table(tmp_path, "all")
    assert "| tree | 1/1 | 1/1 |" in md
    assert "| PySR | 0/1 | 0/1 |" in md   # missing version: counted under "no program" (last column = 1)

def test_io_row_invalid_when_too_many_flagged(tmp_path):
    # two runs, io_agent_1 meta with forbidden_paths on one of them -> 1/2 > 1/3 -> "invalid (audit)"
    ...

def test_report_includes_comparison(tmp_path):
    rd = build_run("absdiff", None, tmp_path); evaluate_run(rd)
    assert "## Comparison (step 4)" in build_report(tmp_path)
```

Row format of `method_table`:
`| <method> | <passes check n/N> | <exact n/N> | <recovered in n/N> | <recovered out n/N> | <median nodes> | <median s> | <no program n> |`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_compare.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `compare.py`, hook it into `build_report`, update the README**

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/compare.py nndc/report.py README.md tests/test_compare.py
git commit -m "feat: step-4 comparison section in REPORT.md; README prior-work positioning"
```

### Task 14: Data — B1–B4 on all 41 nets

**Files:** `runs/**`, `runs_datasets/**` (new `tree.py`, `regions.py`, `pysr*.py`, `*_info.json`, `results.json`)

- [ ] **Step 1: Run the non-PySR baselines**

Run: `uv run nndc baseline tree --all && uv run nndc baseline regions --all && uv run nndc baseline tree --all --root runs_datasets && uv run nndc baseline regions --all --root runs_datasets`
Expected: one line per run; errors are printed and do not stop the batch.

- [ ] **Step 2: Run PySR** (skip if the spike dropped it). Run it in the background, because it takes
  about 41 × 2 min.

Run: `uv run --group pysr nndc baseline pysr --all` and the same with `--root runs_datasets`.

- [ ] **Step 3: Check that existing versions did not change**

Run: `git diff --stat -- 'runs/**/agent_*' 'runs/**/ast.py' 'runs_datasets/**/agent_*' 'runs_datasets/**/ast.py'`
Expected: empty.

- [ ] **Step 4: Commit**

```bash
git add runs runs_datasets
git commit -m "data: B1-B4 baselines on the 29 formula and 12 Iris nets"
```

### Task 15: Data — B5 input/output-only agent runs (41)

- [ ] **Step 1: Prepare**

Run: `uv run nndc agent prepare --io --all` and `uv run nndc agent prepare --io --all --root runs_datasets`
Expected: 41 blind folders under `%TEMP%/nndc-blind`.

- [ ] **Step 2: Dispatch one fresh general-purpose subagent per folder, in batches of up to 8**

Prompt: "Work only inside `<folder>`. Read INSTRUCTIONS.md and follow it. Keep any scratch files inside
this folder." No other context.

- [ ] **Step 3: Collect each folder**

Run: `uv run nndc agent collect <folder> --transcript <...>/subagents/agent-<id>.jsonl`
Expected: `io_agent_1` verdict printed. List `forbidden_paths` hits for review.

- [ ] **Step 4: Commit**

```bash
git add runs runs_datasets
git commit -m "data: B5 Claude from input/output only on 41 nets (one attempt each)"
```

### Task 16: Data — experiment R

- [ ] **Step 1: Train and decompile the planted-bug nets (P1)**

Run, for each of absdiff, median3, tax, window and xor: `uv run nndc run <task> --seed 0 --bug --root runs_bugs`.
Then run `uv run nndc ast --all --split all --root runs_bugs`.
Expected: 5 runs; `bug_learned` printed in `train_log.json`. A net with `bug_learned` < 0.5 stays in and
is noted in the report.

- [ ] **Step 2: Decompile P1 with one blind agent each**

Run `nndc agent prepare` on the 5 runs, dispatch 5 subagents, and collect them (as in step 2 of the
project).

- [ ] **Step 3: Weight-space arms and r_tree on all R nets**

Run: `uv run nndc repair baselines --all --root runs_bugs`, then the same for `runs/tax/seed0 runs/tax/seed1 runs/xor/seed2` and `--all --root runs_datasets`.

- [ ] **Step 4: Fix-agent runs**

Prepare folders:
- `--fix` for the 5 P1, 3 P2 and 12 P3 nets;
- `--fixio` for the 5 P1 and 3 P2 nets.

That is 28 folders. Dispatch one subagent each and collect.

- [ ] **Step 5: Evaluate and commit**

Run: `uv run nndc repair evaluate --all --root runs_bugs` (and the P2/P3 runs).

```bash
git add runs_bugs runs runs_datasets
git commit -m "data: experiment R (planted-bug, natural-bug and Iris nets): fix agents and weight-space repair"
```

### Task 17: Report and final review

- [ ] **Step 1: Regenerate the report**

Run: `uv run nndc report` (and `--root runs_datasets --out runs_datasets/REPORT.md` if its report is
regenerated by the same tool).
Expected: `REPORT.md` has "## Comparison (step 4)" with the method tables, the recovery matrix, the
Iris table, the loop counts and the fix table.

- [ ] **Step 2: Read the tables against the raw files**

Spot-check 3 runs per table by hand against `results.json` and `repair/results.json`.

- [ ] **Step 3: Full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add REPORT.md runs_datasets/REPORT.md
git commit -m "docs: step-4 comparison and fix-and-reinsert results in REPORT.md"
```

- [ ] **Step 5: One fresh review of the whole step-4 branch** (code and reported claims) before merging.
