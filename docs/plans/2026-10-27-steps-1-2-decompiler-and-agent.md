# Decompiler + Claude Code refactoring (steps 1–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn small trained/handcrafted ReLU MLPs into readable Python-subset programs, check any program against the original net, and let the user run Claude Code on blind copies and collect the results, up to the spike gate.

**Architecture:** One program format, a restricted Python subset parsed with `ast` and run by our own vectorized interpreter, is used for every version of a model (`dump.py`, `agent_<n>.py`). A single checker compares any program with the net's NumPy reference. Each model lives in `runs/<task>/<hand|seed<k>>/`. Claude Code runs are user-driven: `nndc agent prepare` builds a blind folder outside the repo, the user runs Claude Code in it, and `nndc agent collect` copies the results back, audits the transcript, and re-checks independently.

**Tech Stack:** Python 3.13, NumPy, PyTorch (CPU, training only), pytest, uv, argparse.

**Spec:** `docs/design/2026-10-27-nn-decompiler-design.md` (this plan implements §4, §5 step 1, §5 step 2, the step-1/2 parts of §6–§7).

**Out of scope here (later plans):** AST heuristics tool (step 3), literature search and baselines (step 4), C/pruning/regions/benchmarks (step 5), bigger models and RL (step 6), the "with HEURISTICS.md" condition.

## Global Constraints

- Python ≥ 3.13; `torch>=2.11,<2.12` from the PyTorch CPU index; `numpy>=2.3`; dev: `pytest>=8`. Managed with `uv`; `uv.lock` committed.
- `nndc/spec.py`, `nndc/model.py`, `nndc/lang.py`, `nndc/check.py` import **only the stdlib, NumPy, and each other** (relative imports). They get copied into blind folders. `torch` may be imported only inside function bodies in `model.py`, and anywhere in `experiments/`.
- Program files are **never** `eval`'d, `exec`'d, or imported. Only `ast.parse` + the allowlist interpreter.
- Anything a refactorer can see (blind folder contents, `dump.py`) contains **no task name, no real input names, no target, no training data**. Programs always use neutral argument names `x0, x1, …` and the function name `net`.
- Net reference is float32 (`MLP.forward`). Program evaluation is float64.
- Checker defaults (from the design, fixed in advance): `atol=1e-5`, `rtol=1e-4`, `margin_frac=0.05` (`m = 0.05 × p95(|out − threshold|)`, for argmax the top-2 gap), `snap_tol_frac=0.02` (`snap_tol = 0.02 × (max − min of net output over the input set)`), exhaustive cap `10_000_000` inputs, binary threshold `0.0` with class 1 iff `out > threshold` (strict).
- Seeds for trained runs: 0, 1, 2. CPU only.
- Paths via `pathlib`, files written UTF-8 with `\n`. Must work on Windows and Linux.
- Run folder layout: `runs/<task>/hand/` (handcrafted) or `runs/<task>/seed<k>/` (trained), containing `spec.json`, `weights.npz`, `net.pt` (trained only), `train_log.json`, `dump.py`, `agent_<n>.py`, `agent_<n>_notes.md`, `agent_<n>_checks.log`, `agent_<n>_meta.json`, `counterexamples.json`, `results.json`.
- Commit messages: conventional style, no attribution lines.

## Review Focus

1. **Claude writes code outside the allowlist** (imports, `math.floor`, list comprehensions, helper functions, `**`, loops). Expect a `ProgramError` naming the construct and line, shown by `check.py`, not a traceback. Pinned in Task 3.
2. **A program path without `return`, or a wrong signature** (different arg names/count, function not called `net`). Expect a clear error, and the check fails rather than crashing. Pinned in Tasks 3 and 5.
3. **NaN/inf outputs or division by a zero constant.** Division by constant 0 is rejected at parse time, and NaN outputs count as disagreements. Pinned in Tasks 3 and 5.
4. **Re-running `nndc run` or `collect` on an existing run** must not retrain silently, overwrite collected attempts, or duplicate an attempt. Pinned in Tasks 7 and 8.
5. **No transcript found / temp path with spaces / Windows paths.** `collect` still works, records `transcript_found: false`, and the report flags it. Pinned in Task 8.

---

### Task 1: Project scaffold and model spec

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `nndc/__init__.py`, `nndc/spec.py`, `experiments/__init__.py`, `tests/__init__.py`, `tests/test_spec.py`

**Interfaces:**
- Produces (`nndc/spec.py`):
  ```python
  InputType = Literal["float", "int", "byte", "bool"]
  OutputKind = Literal["regression", "binary", "argmax"]

  @dataclass
  class InputSpec:
      name: str          # neutral program name, "x0", "x1", ...
      real_name: str     # shown only in REPORT/README
      type: InputType
      lo: float
      hi: float          # bool => 0..1, byte => 0..255

  @dataclass
  class Spec:
      task: str
      output: OutputKind
      inputs: list[InputSpec]
      n_outputs: int = 1
      threshold: float = 0.0
      atol: float = 1e-5
      rtol: float = 1e-4
      margin_frac: float = 0.05
      snap_tol_frac: float = 0.02
      override_reason: str | None = None
      def to_json(self) -> dict
      @staticmethod
      def from_json(d: dict) -> "Spec"
      def blind(self) -> dict          # to_json() minus task, real_name, override_reason
      def save(self, path: Path) -> None
      @staticmethod
      def load(path: Path) -> "Spec"   # accepts full or blind json (task -> "" if missing)
      def is_discrete(self) -> bool    # all inputs int/byte/bool
      def domain_size(self) -> int     # product of (hi-lo+1) for discrete inputs; raises if not discrete
  ```

- [ ] **Step 1: Scaffold.** `git init`. Write `.gitignore` (`.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`). Write `pyproject.toml`: project `nndc`, `requires-python = ">=3.13"`, deps `numpy>=2.3`, `torch>=2.11,<2.12`; dev group `pytest>=8`; script `nndc = "nndc.cli:main"`; packages `nndc`, `experiments`; `[[tool.uv.index]] name="pytorch-cpu" url="https://download.pytorch.org/whl/cpu" explicit=true` and `[tool.uv.sources] torch = { index = "pytorch-cpu" }`. Run `uv sync`. Expected: environment created, `uv run python -c "import torch, numpy"` succeeds.

- [ ] **Step 2: Write failing tests** in `tests/test_spec.py`:
  ```python
  def make():
      return Spec(task="absdiff", output="regression",
                  inputs=[InputSpec("x0", "a", "float", -3, 3), InputSpec("x1", "b", "float", -3, 3)])

  def test_roundtrip(tmp_path):
      s = make(); s.save(tmp_path / "spec.json")
      assert Spec.load(tmp_path / "spec.json") == s

  def test_blind_hides_task_and_real_names():
      b = make().blind()
      text = json.dumps(b)
      assert "absdiff" not in text and '"a"' not in text and "real_name" not in text
      assert [i["name"] for i in b["inputs"]] == ["x0", "x1"]

  def test_blind_loads():
      s = Spec.from_json(make().blind())
      assert s.task == "" and s.inputs[0].name == "x0"

  def test_domain_size():
      s = Spec(task="t", output="binary", inputs=[InputSpec("x0", "p", "bool", 0, 1), InputSpec("x1", "q", "byte", 0, 255)])
      assert s.is_discrete() and s.domain_size() == 2 * 256
      assert not make().is_discrete()
  ```
- [ ] **Step 3: Run** `uv run pytest tests/test_spec.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 4: Implement** `nndc/spec.py` as in Interfaces. `Spec.load` on blind JSON fills `real_name` with `name`.
- [ ] **Step 5: Run** `uv run pytest tests/test_spec.py -v`. Expected: PASS.
- [ ] **Step 6: Commit** `pyproject.toml uv.lock .gitignore nndc experiments tests docs` with message `chore: scaffold nndc project, add model spec` (this also commits the design doc and this plan).

---

### Task 2: MLP model and NumPy reference

**Files:**
- Create: `nndc/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces (`nndc/model.py`):
  ```python
  class UnsupportedModel(ValueError): ...

  @dataclass
  class MLP:
      weights: list[np.ndarray]   # W_i, shape (out_i, in_i), float32
      biases: list[np.ndarray]    # b_i, shape (out_i,), float32
      # ReLU after every layer except the last
      @property
      def n_inputs(self) -> int
      @property
      def n_outputs(self) -> int
      def forward(self, X: np.ndarray) -> np.ndarray       # (N, n_inputs) -> (N, n_outputs), float32 math
      def preactivations(self, X: np.ndarray) -> list[np.ndarray]  # per layer (N, out_i), before ReLU

  def from_torch(module) -> MLP      # nn.Sequential: Linear, ReLU, Linear, ..., Linear
  def to_torch(mlp: MLP)             # inverse, for training/benchmarks
  def save_npz(mlp: MLP, path: Path) -> None   # keys W0,b0,W1,b1,...
  def load_npz(path: Path) -> MLP
  ```

- [ ] **Step 1: Write failing tests:**
  ```python
  def test_forward_matches_torch():
      torch.manual_seed(0)
      seq = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2))
      mlp = from_torch(seq)
      X = np.random.default_rng(0).uniform(-3, 3, (100, 3)).astype(np.float32)
      np.testing.assert_allclose(mlp.forward(X), seq(torch.from_numpy(X)).detach().numpy(), rtol=1e-6, atol=1e-6)

  def test_rejects_unsupported():
      with pytest.raises(UnsupportedModel, match="Tanh"):
          from_torch(nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)))
      with pytest.raises(UnsupportedModel):
          from_torch(nn.Sequential(nn.Linear(2, 3), nn.ReLU()))   # must end with Linear

  def test_npz_roundtrip(tmp_path):
      mlp = from_torch(nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)))
      save_npz(mlp, tmp_path / "w.npz"); m2 = load_npz(tmp_path / "w.npz")
      assert all((a == b).all() for a, b in zip(mlp.weights, m2.weights))

  def test_model_module_imports_without_torch():
      src = Path("nndc/model.py").read_text()
      assert not re.search(r"^(import torch|from torch)", src, re.M)
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_model.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `nndc/model.py`. `torch` is imported only inside `from_torch`/`to_torch`. Error messages name the offending layer type and index.
- [ ] **Step 4: Run** `uv run pytest tests/test_model.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: MLP model with numpy reference and npz io`.

---

### Task 3: Program language (parser, allowlist, interpreter, size)

**Files:**
- Create: `nndc/lang.py`, `tests/test_lang.py`

**Interfaces:**
- Produces (`nndc/lang.py`):
  ```python
  LANG_VERSION = 1
  FUNCTIONS = {"relu", "min", "max", "abs", "sum", "argmax"}

  class ProgramError(ValueError): ...   # message: "line N: <what> is not allowed; allowed: ..."

  @dataclass
  class Program:
      src: str
      arg_names: list[str]
      body: list[ast.stmt]

  def parse_program(src: str, expected_args: list[str] | None = None) -> Program
  def evaluate(prog: Program, X: np.ndarray) -> np.ndarray       # (N, n_args) -> (N,) float64
  def program_size(prog: Program) -> dict    # {"lines", "nodes", "constants", "relus", "ifs"}
  def allowlist_summary() -> str             # human text used in INSTRUCTIONS.md and errors
  ```
- Allowed, version 1:
  - Module containing exactly one `def net(<args>):`. Leading comments are fine; a leading docstring is allowed.
  - Statements: `Assign` (single `Name` target), `If`/`elif`/`else`, `Return`.
  - Expressions: `Name`, `Constant` (int/float/bool), `BinOp` with `+ - *` and `/` only when the right side is a nonzero numeric constant, `UnaryOp` (`-`, `+`, `not`), `BoolOp` (`and`/`or`), `Compare` (`< <= > >= == !=`, chains allowed), `IfExp`, `Call` to a name in `FUNCTIONS` with positional args. `sum` also accepts one tuple/list literal argument. `argmax(a, b, ...)` returns the index of the first maximum.
  - Anything else raises `ProgramError`.

- [ ] **Step 1: Write failing tests:**
  ```python
  X2 = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 0.0]])

  def ev(src, X=X2): return evaluate(parse_program(src), X)

  def test_arith_and_functions():
      np.testing.assert_allclose(ev("def net(x0, x1):\n    return relu(x0 - x1) + abs(x1) / 2"), [1.0, 4.5, 0.0])
      np.testing.assert_allclose(ev("def net(x0, x1):\n    return max(x0, x1, 0.5) + sum((x0, x1))"), [5.0, 5.0, 0.5])
      np.testing.assert_allclose(ev("def net(x0, x1):\n    return argmax(x0, x1)"), [1, 0, 0])

  def test_if_elif_else_per_row():
      src = "def net(x0, x1):\n    if x0 > 2:\n        return 10\n    elif x1 > 1:\n        y = x1 * 3\n        return y\n    else:\n        return -1"
      np.testing.assert_allclose(ev(src), [6.0, 10.0, -1.0])

  def test_bool_results_and_ifexp():
      np.testing.assert_allclose(ev("def net(x0, x1):\n    return abs(x0 - x1) < 1.5"), [1, 0, 1])
      np.testing.assert_allclose(ev("def net(x0, x1):\n    return 1 if x0 != x1 and not x0 > 2 else 0"), [1, 0, 0])

  @pytest.mark.parametrize("src, word", [
      ("import math\ndef net(x0):\n    return x0", "Import"),
      ("def net(x0):\n    return math.floor(x0)", "Attribute"),
      ("def net(x0):\n    return x0 ** 2", "Pow"),
      ("def net(x0):\n    return sum([x0 for _ in (1, 2)])", "ListComp"),
      ("def f(x0):\n    return x0\ndef net(x0):\n    return f(x0)", "one function"),
      ("def net(x0):\n    for i in (1,2):\n        pass\n    return x0", "For"),
      ("def net(x0):\n    return x0 / 0", "division"),
      ("def net(x0):\n    return x0 / x0", "division"),
      ("def net(x0):\n    return round(x0)", "round"),
  ])
  def test_rejects_with_named_construct(src, word):
      with pytest.raises(ProgramError, match=word):
          parse_program(src)

  def test_error_has_line_number():
      with pytest.raises(ProgramError, match="line 3"):
          parse_program("def net(x0):\n    y = x0\n    return y ** 2")

  def test_signature_mismatch():
      with pytest.raises(ProgramError, match="arguments"):
          parse_program("def net(a, b):\n    return a", expected_args=["x0", "x1"])

  def test_missing_return_on_some_rows():
      with pytest.raises(ProgramError, match="no return"):
          ev("def net(x0, x1):\n    if x0 > 2:\n        return 1")

  def test_unassigned_name():
      with pytest.raises(ProgramError, match="h9"):
          ev("def net(x0, x1):\n    return h9")

  def test_size():
      s = program_size(parse_program("def net(x0, x1):\n    h = relu(x0 - 0.5)\n    if h > 1:\n        return h\n    return 0"))
      assert s["relus"] == 1 and s["ifs"] == 1 and s["constants"] == 3 and s["lines"] == 4
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_lang.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `nndc/lang.py`. The parser is an `ast.NodeVisitor` whose checks match the allowlist above. Every rejection names the AST node class (or "one function", "division", "arguments") plus the line number.
  - **Interpreter algorithm** (vectorized masked execution; not determined by the tests alone):
    - `exec_block(stmts, rows)`, where `rows` is an int index array of the still-active rows.
    - `env` maps names to full-length float64 arrays, plus a per-name boolean "assigned" mask.
    - `Assign` writes `env[name][rows]` and marks those rows assigned.
    - `If` evaluates the condition on `rows`, recurses into the body with the true subset and into the else branch with the false subset, then continues with the rows that have not returned.
    - `Return` writes `out[rows]` and marks those rows done.
    - Expressions are evaluated on the `rows` subset. Reading a name on rows where it is unassigned raises `ProgramError("name h9 used before assignment")`.
    - After the body, any row not done raises `ProgramError("no return for some inputs, e.g. x0=..., x1=...")`.
    - Bools become 0.0/1.0. `and`/`or` are logical ops on truthiness. `IfExp` uses `np.where`.
  - `program_size` metrics:
    - `nodes`: all AST nodes inside the function body.
    - `constants`: numeric `Constant` nodes, counting `-c` as one.
    - `lines`: non-empty, non-comment lines of the function, excluding the `def` line.
- [ ] **Step 4: Run** `uv run pytest tests/test_lang.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: restricted python program language with allowlist and vectorized interpreter`.

---

### Task 4: Dump (net → program)

**Files:**
- Create: `nndc/dump.py`, `tests/test_dump.py`

**Interfaces:**
- Consumes: `MLP` (Task 2), `Spec` (Task 1), `parse_program`/`evaluate` (Task 3).
- Produces: `def dump_program(mlp: MLP, spec: Spec) -> str`

Format (exact):
```
# dumped from a neural network: 2 inputs -> 3 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: binary: class 1 if the returned value > 0.0, else class 0
def net(x0, x1):
    h1_0 = relu(0.9813 * x0 - 1.0127 * x1 - 0.1984)  # layer 1 neuron 0
    ...
    return -2.31 * h1_0 - 2.28 * h1_1 + 0.4507
```
- Hidden neurons are named `h<layer>_<j>`, with layers starting at 1.
- Constants are formatted with `np.format_float_positional(np.float32(c), unique=True, trim="-")`.
- Terms with weight exactly `0.0` are omitted. A zero bias is omitted. A neuron with no terms gets `relu(0)`.
- Header lines by output kind:
  - regression: `# output: regression: the returned value`
  - argmax: `# output: argmax over K outputs: the returned class index`, with outputs `o0..o{K-1}` assigned and then `return argmax(o0, ..., oK-1)`.
- No task name or real names appear anywhere in the output.

- [ ] **Step 1: Write failing tests:**
  ```python
  @pytest.mark.parametrize("sizes, kind", [([2, 3, 1], "binary"), ([3, 8, 8, 1], "regression"), ([2, 4, 3], "argmax")])
  def test_dump_evaluates_like_net(sizes, kind):
      mlp = random_mlp(sizes, seed=1)     # helper in the test file: float32 normal weights
      spec = Spec(task="secret_task", output=kind, n_outputs=sizes[-1],
                  inputs=[InputSpec(f"x{i}", f"real{i}", "float", -3, 3) for i in range(sizes[0])])
      src = dump_program(mlp, spec)
      X = np.random.default_rng(2).uniform(-3, 3, (2000, sizes[0]))
      out = evaluate(parse_program(src, [f"x{i}" for i in range(sizes[0])]), X)
      ref = mlp.forward(X.astype(np.float32))
      if kind == "argmax":
          assert (out == ref.argmax(1)).mean() > 0.999
      else:
          np.testing.assert_allclose(out, ref[:, 0], rtol=1e-4, atol=1e-5)
      assert "secret_task" not in src and "real0" not in src

  def test_zero_weights_omitted():
      mlp = MLP([np.array([[1.0, 0.0]], np.float32), np.array([[2.0]], np.float32)],
                [np.zeros(1, np.float32), np.zeros(1, np.float32)])
      spec = Spec(task="t", output="regression", inputs=[InputSpec("x0", "a", "float", 0, 1), InputSpec("x1", "b", "float", 0, 1)])
      src = dump_program(mlp, spec)
      body = src.split("def net(x0, x1):\n")[1]
      assert "x1" not in body                       # x1 appears only in the signature
      assert "h1_0 = relu(1 * x0)" in body
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_dump.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `dump_program` in `nndc/dump.py`. Coefficient `1`/`-1` stays written as `1 * x0` / `- 1 * x0`; this is a raw dump, so nothing is simplified.
- [ ] **Step 4: Run** `uv run pytest tests/test_dump.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: dump MLP as readable program`.

---

### Task 5: Checker

**Files:**
- Create: `nndc/check.py`, `tests/test_check.py`

**Interfaces:**
- Consumes: `Spec`, `MLP`, `parse_program`, `evaluate`, `ProgramError`.
- Produces (`nndc/check.py`):
  ```python
  EXHAUSTIVE_CAP = 10_000_000

  @dataclass
  class CheckResult:
      exact_pass: bool
      snap_pass: bool
      label: Literal["exhaustive", "empirical"]
      n_inputs: int
      max_abs_err: float              # regression only, else nan
      disagreements_outside: int      # outside the unsure zone (or > snap_tol for regression)
      disagreements_inside: int       # inside the zone (classification only)
      zone_fraction: float            # fraction of inputs in the unsure zone (classification), else 0
      margin: float                   # m (binary/argmax) or snap_tol (regression)
      counterexamples: list[dict]     # up to 20: {"inputs": {"x0":..}, "net": float, "program": float, "in_zone": bool}
      error: str | None               # ProgramError text; then both passes are False
      def to_json(self) -> dict

  def build_input_set(mlp: MLP, spec: Spec, seed: int = 0,
                      extra: np.ndarray | None = None) -> tuple[np.ndarray, str]   # (X, label)
  def net_output(mlp: MLP, spec: Spec, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]
      # (decision, score): regression -> (value, value); binary -> (class 0/1, out - threshold);
      # argmax -> (class index, top-2 gap)
  def program_decision(spec: Spec, raw: np.ndarray) -> np.ndarray
      # regression -> raw; binary -> (raw > threshold); argmax -> raw (class index)
  def check_program(src: str, mlp: MLP, spec: Spec, seed: int = 0,
                    extra_inputs: np.ndarray | None = None) -> CheckResult
  def format_result(r: CheckResult, spec: Spec) -> str   # human text, first line "PASS (snap)" / "PASS (exact)" / "FAIL" / "ERROR"
  ```
- Input set (`build_input_set`):
  - **Exhaustive:** if `spec.is_discrete()` and `domain_size() ≤ EXHAUSTIVE_CAP`, use the full grid.
  - **Empirical** otherwise, the union of:
    - 200_000 uniform random points (integers for discrete dims)
    - a full grid with 201 points per dim, if there are ≤ 2 continuous dims and no discrete dims
    - all corners if d ≤ 12, else 4096 random corners
    - boundary points: for 2000 random segments, evaluate every hidden preactivation and the output score at 64 points along the segment. At each sign change, bisect 40 times. Add the found point, the points ±1e-4·(hi−lo) along the segment, and `np.nextafter` neighbours in both directions on each continuous coordinate. Clip to bounds and cap at 200_000 boundary points.
  - Always add `extra` (replayed counterexamples). RNG is `np.random.default_rng(seed)`.
- Pass rules (exactly as in the Global Constraints):
  - exact: regression `|p−n| ≤ atol + rtol·|n|` everywhere; classification has no disagreements anywhere. NaN/inf program outputs are disagreements.
  - snap: regression `|p−n| ≤ snap_tol`; classification disagreements only where the score is `< m`.

- [ ] **Step 1: Write failing tests** (handcrafted nets defined inline as small `MLP`s):
  ```python
  def absdiff_net():   # relu(a-b) + relu(b-a)
      return MLP([np.array([[1, -1], [-1, 1]], np.float32), np.array([[1, 1]], np.float32)],
                 [np.zeros(2, np.float32), np.zeros(1, np.float32)])
  REG = Spec(task="t", output="regression", inputs=[InputSpec("x0","a","float",-3,3), InputSpec("x1","b","float",-3,3)])

  def test_exact_formula_passes_exact():
      r = check_program("def net(x0, x1):\n    return abs(x0 - x1)", absdiff_net(), REG)
      assert r.exact_pass and r.snap_pass and r.label == "empirical" and r.error is None

  def test_slightly_off_passes_snap_not_exact():
      r = check_program("def net(x0, x1):\n    return 1.01 * abs(x0 - x1)", absdiff_net(), REG)
      assert not r.exact_pass and r.snap_pass      # max err 0.06 <= 0.02*6 = 0.12

  def test_wrong_fails_with_counterexamples():
      r = check_program("def net(x0, x1):\n    return x0 - x1", absdiff_net(), REG)
      assert not r.snap_pass and len(r.counterexamples) > 0 and set(r.counterexamples[0]["inputs"]) == {"x0", "x1"}

  def test_program_error_is_reported_not_raised():
      r = check_program("import os\ndef net(x0, x1):\n    return 0", absdiff_net(), REG)
      assert r.error and "Import" in r.error and not r.snap_pass
      assert format_result(r, REG).startswith("ERROR")

  def test_nan_counts_as_disagreement():
      r = check_program("def net(x0, x1):\n    return abs(x0 - x1) * (1 if x0 < 2.9 else 1e308) * 10", absdiff_net(), REG)
      assert not r.exact_pass

  def test_binary_snap_zone():
      # net: 0.2 - |a-b|  (class 1 iff |a-b| < 0.2); program uses 0.21: disagrees only near the boundary
      net = MLP([np.array([[1, -1], [-1, 1]], np.float32), np.array([[-1, -1]], np.float32)],
                [np.zeros(2, np.float32), np.array([0.2], np.float32)])
      spec = Spec(task="t", output="binary", inputs=[InputSpec("x0","a","float",-2,2), InputSpec("x1","b","float",-2,2)])
      r = check_program("def net(x0, x1):\n    return abs(x0 - x1) < 0.21", net, spec)
      assert not r.exact_pass and r.snap_pass and r.disagreements_inside > 0 and r.disagreements_outside == 0
      r2 = check_program("def net(x0, x1):\n    return abs(x0 - x1) < 0.6", net, spec)
      assert not r2.snap_pass and r2.disagreements_outside > 0

  def test_exhaustive_label_for_small_discrete():
      xor = MLP([np.array([[1, 1], [1, 1]], np.float32), np.array([[1, -2]], np.float32)],
                [np.array([0, -1], np.float32), np.array([-0.5], np.float32)])
      spec = Spec(task="t", output="binary", inputs=[InputSpec("x0","p","bool",0,1), InputSpec("x1","q","bool",0,1)])
      r = check_program("def net(x0, x1):\n    return x0 != x1", xor, spec)
      assert r.label == "exhaustive" and r.n_inputs == 4 and r.exact_pass

  def test_argmax_program():
      net = MLP([np.array([[1, 0], [0, 1]], np.float32), np.array([[1, 0], [0, 1]], np.float32)],
                [np.zeros(2, np.float32), np.zeros(2, np.float32)])     # outputs relu(x0), relu(x1)
      spec = Spec(task="t", output="argmax", n_outputs=2,
                  inputs=[InputSpec("x0","a","float",0,1), InputSpec("x1","b","float",0,1)])
      assert check_program("def net(x0, x1):\n    return 1 if x1 > x0 else 0", net, spec).exact_pass
      r = check_program("def net(x0, x1):\n    return 0", net, spec)
      assert not r.snap_pass and r.disagreements_outside > 0

  def test_rare_branch_missed_by_random_but_caught_by_boundary_points():
      # bump of height 1 and width 2e-5 at x=1 on [-5, 5], exactly 0 elsewhere (no float cancellation):
      # h = relu(x-1), relu(1-x); g = relu(e - h0 - h1); out = g / e
      e = 1e-5
      net = MLP([np.array([[1], [-1]], np.float32), np.array([[-1, -1]], np.float32), np.array([[1 / e]], np.float32)],
                [np.array([-1, 1], np.float32), np.array([e], np.float32), np.zeros(1, np.float32)])
      spec = Spec(task="t", output="regression", inputs=[InputSpec("x0", "x", "float", -5, 5)])
      Xr = np.random.default_rng(0).uniform(-5, 5, (200_000, 1))
      assert (net.forward(Xr.astype(np.float32))[:, 0] > 0.1).sum() == 0      # pure random sampling misses it
      r = check_program("def net(x0):\n    return 0", net, spec)
      assert not r.snap_pass                                                   # boundary points catch it

  def test_extra_inputs_are_replayed():
      X, _ = build_input_set(absdiff_net(), REG, extra=np.array([[2.5, -2.5]]))
      assert any((row == [2.5, -2.5]).all() for row in X)
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_check.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `nndc/check.py` per Interfaces. `m = margin_frac × p95(score)` over the input set, where score is `|out − threshold|` for binary and the top-2 gap for argmax. `snap_tol = snap_tol_frac × (max − min of net value)`. A `ProgramError` from parse or evaluate is caught into `error`.
- [ ] **Step 4: Run** `uv run pytest tests/test_check.py -v`. Expected: PASS. If the rare-branch precondition (random sampling misses the bump with seed 0) does not hold, change the bump position or `e`, not the assertions.
- [ ] **Step 5: Commit** `feat: checker with exact/snap acceptance and boundary sampling`.

---

### Task 6: Tasks, handcrafted nets, training, true-task grading

**Files:**
- Create: `experiments/tasks.py`, `experiments/handcrafted.py`, `experiments/train.py`, `experiments/grade.py`, `tests/test_tasks.py`

**Interfaces:**
- Consumes: `Spec`, `InputSpec`, `MLP`, `from_torch`, `parse_program`, `evaluate`, `build_input_set`.
- Produces:
  ```python
  # experiments/tasks.py
  @dataclass
  class Task:
      name: str
      spec: Spec                                  # real names filled in
      target: Callable[[np.ndarray], np.ndarray]  # (N,d) -> (N,): value, or class 0/1
      expected: str                               # expected formula as a program-language expression in x0.., written before any run
      hidden: list[int] | None                    # trained architecture; None = hand-only
      handcrafted: Callable[[], MLP] | None
      steps: int = 3000
      lr: float = 1e-2
      sampler: Callable[[np.random.Generator, int], np.ndarray] | None = None   # default: uniform over bounds
  TASKS: dict[str, Task]
  def get_task(name: str) -> Task
  def expected_program(task: Task) -> str         # "def net(x0, ...):\n    return <expected>"

  # experiments/train.py
  def train(task: Task, seed: int) -> tuple[MLP, dict]   # dict: train_log (loss curve every 100 steps, quality, learned)

  # experiments/grade.py
  def grade_task(src: str, task: Task, seed: int = 0) -> dict
      # {"in_range": float, "out_of_range": float | None, "metric": "accuracy" | "max_abs_err", "recovered": bool, "error": str | None}
  ```
- Task table (exact). `expected` uses neutral names; "hand" = has `handcrafted`, "train" = has `hidden`:

  | name | inputs (real names) | output | expected | hand | train (hidden) |
  |---|---|---|---|---|---|
  | `xor` | p, q: bool | binary | `x0 != x1` | h0=relu(x0+x1), h1=relu(x0+x1−1); out=h0−2·h1−0.5 | [4], data = the 4 points, lr 5e-2 |
  | `max` | a, b: float [−3,3] | regression | `max(x0, x1)` | relu(a−b), relu(b), relu(−b); out=h0+h1−h2 | — |
  | `absdiff` | a, b: float [−3,3] | regression | `abs(x0 - x1)` | relu(a−b), relu(b−a); out=h0+h1 | [8] |
  | `absdiff_perturbed` | a, b: float [−3,3] | regression | `abs(x0 - x1)` | absdiff with each weight ×(1+U(−0.03,0.03)) and biases U(−0.01,0.01), rng seed 0 | — |
  | `triangle` | x: float [−2,6] | regression | `max(0, 1 - abs(x0 - 3))` | relu(x−2), relu(x−3), relu(x−4); out=h0−2h1+h2 | — |
  | `ramp` | x: float [−2,3] | regression | `min(max(x0, 0), 1)` | relu(x), relu(x−1); out=h0−h1 | — |
  | `window` | x: float [−2,2] | binary | `abs(x0) < 0.5` | relu(x), relu(−x); out=0.5−h0−h1 | [4] |
  | `neareq` | a, b: float [−2,2] | binary | `abs(x0 - x1) < 0.2` | relu(a−b), relu(b−a); out=0.2−h0−h1 | [8], sampler: 50% with b=a+U(−0.4,0.4) clipped, 50% uniform |
  | `deadredundant` | a, b: float [0,1] | regression | `x0 + x1 + max(0, x0 - x1)` | relu(a+b), relu(−a−b−1), relu(a−b), relu(a−b); out=h0+5h1+0.5h2+0.5h3 | — |
  | `sparse8` | x0..x7: float [0,1] (real names `u0..u7`) | regression | `max(0, x2 - x5)` | relu(x2−x5+0.0005·(x0+x1+x3+x4+x6+x7)); out=h0 | — |
  | `sum8` | x0..x7: int [0,255] (real `v0..v7`) | regression | `max(0, 0.02 * sum(x0, x1, x2, x3, x4, x5, x6, x7) - 20)` | relu(0.02·Σx − 20); out=h0 | — |
  | `median3` | a, b, c: float [−3,3] | regression | `max(min(x0, x1), min(max(x0, x1), x2))` | — | [16, 16] |
  | `tax` | income: float [0,200] | regression | `0.1 * min(x0, 20) + 0.2 * min(max(x0 - 20, 0), 60) + 0.4 * max(x0 - 80, 0)` | — | [8], steps 6000 |

- Training: Adam with `lr`; batch 512 fresh samples per step from `sampler`, except `xor`, which uses the 4 points every step. Loss is MSE (regression) or `BCEWithLogitsLoss` (binary). `torch.manual_seed(seed)`, `np.random.default_rng(seed)`, float32, `torch.set_num_threads(1)`.
- Quality on 20_000 fresh samples (rng seed `1000+seed`): accuracy (binary), or RMSE and R² (regression). `learned = acc ≥ 0.98` or `R² ≥ 0.99`.
- Grading (independent of any net):
  - **in_range**: the full grid if the spec is discrete and `domain_size() ≤ 10_000_000`, else 200_000 uniform points (`np.random.default_rng(seed)`).
  - **out_of_range**: for continuous inputs only, bounds widened by 50% of the range on each side, with points outside the original box; `None` if all inputs are discrete.
  - **recovered**: accuracy == 1.0 on both sets (classification), or `max_abs_err ≤ 1e-6 × target range` on both (regression).

- [ ] **Step 1: Write failing tests:**
  ```python
  @pytest.mark.parametrize("name", list(TASKS))
  def test_expected_formula_is_correct(name):
      t = get_task(name)
      g = grade_task(expected_program(t), t)
      assert g["error"] is None and g["recovered"], g

  @pytest.mark.parametrize("name", [n for n, t in TASKS.items() if t.handcrafted])
  def test_handcrafted_net_matches_target(name):
      t = get_task(name); mlp = t.handcrafted()
      X, _ = build_input_set(mlp, t.spec, seed=0)
      out = mlp.forward(X.astype(np.float32))[:, 0]
      tgt = t.target(X)
      if t.spec.output == "binary":
          assert (((out > 0).astype(int)) == tgt).mean() > 0.999
      elif name != "absdiff_perturbed":
          np.testing.assert_allclose(out, tgt, atol=0.01 if name == "sparse8" else 1e-4)

  def test_specs_use_neutral_names():
      for t in TASKS.values():
          assert [i.name for i in t.spec.inputs] == [f"x{k}" for k in range(len(t.spec.inputs))]

  def test_training_smoke():
      t = replace(get_task("absdiff"), steps=200)
      mlp, log = train(t, seed=0)
      assert mlp.n_inputs == 2 and "quality" in log and "learned" in log
      mlp2, _ = train(t, seed=0)
      assert np.array_equal(mlp.weights[0], mlp2.weights[0])   # deterministic
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_tasks.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the four modules. `targets` are written independently of `expected` (plain NumPy, e.g. `np.abs(X[:,0]-X[:,1])`), so the expected-formula test is a real cross-check.
- [ ] **Step 4: Run** `uv run pytest tests/test_tasks.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: step-1 tasks, handcrafted nets, training and task grading`.

---

### Task 7: Run pipeline and CLI (`nndc run`, `nndc check`)

**Files:**
- Create: `nndc/run.py`, `nndc/cli.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  ```python
  # nndc/run.py
  RUNS = Path("runs")
  def run_dir(task: str, seed: int | None, root: Path = RUNS) -> Path      # runs/<task>/hand or seed<k>
  def build_run(task: str, seed: int | None, root: Path = RUNS, retrain: bool = False) -> Path
      # creates the net (handcrafted or trained) only if weights.npz is missing or retrain=True;
      # writes spec.json, weights.npz, net.pt (trained only), train_log.json, dump.py; then evaluate_run()
  def evaluate_run(rd: Path) -> dict
      # checks + grades every version (dump.py and every agent_<n>.py), replaying counterexamples.json,
      # merges new counterexamples into counterexamples.json (dedup), writes and returns results.json:
      # {"task", "run", "kind": "hand"|"trained", "train": {...}, "versions": {"dump": V, "agent_1": V, ...}}
      # V = {"check": CheckResult.to_json(), "task": grade_task(...), "size": program_size(...), "meta": {...}}
  def load_run(rd: Path) -> tuple[Spec, MLP]
  def all_runs(root: Path = RUNS) -> list[Path]

  # nndc/cli.py
  def main(argv: list[str] | None = None) -> int
  ```
- CLI:
  - `nndc run <task|all> [--seed K | --hand] [--retrain]`. For a single task, the default is every applicable kind: hand if it has `handcrafted`, and seeds 0, 1, 2 if it has `hidden`. `all` does every task.
  - `nndc check <program.py> --run <run_dir>` prints `format_result` and exits 0 iff `snap_pass`.
  - Later tasks add the `agent`, `report`, and `gate` subcommands.
- `build_run` raises if the dump fails `exact_pass`. That would be a decompiler bug.
- With `retrain=True` and existing `agent_*` files, refuse with a message: they belong to the old weights.

- [ ] **Step 1: Write failing tests:**
  ```python
  def test_build_hand_run(tmp_path):
      rd = build_run("absdiff", None, root=tmp_path)
      assert rd == tmp_path / "absdiff" / "hand"
      for f in ["spec.json", "weights.npz", "dump.py", "results.json"]:
          assert (rd / f).exists()
      res = json.loads((rd / "results.json").read_text())
      assert res["versions"]["dump"]["check"]["exact_pass"]

  def test_rerun_keeps_weights_and_agent_attempts(tmp_path):
      rd = build_run("absdiff", None, root=tmp_path)
      (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
      w_before = (rd / "weights.npz").read_bytes()
      build_run("absdiff", None, root=tmp_path)
      assert (rd / "weights.npz").read_bytes() == w_before and (rd / "agent_1.py").exists()
      res = json.loads((rd / "results.json").read_text())
      assert res["versions"]["agent_1"]["task"]["recovered"]

  def test_retrain_refuses_with_agent_attempts(tmp_path):
      rd = build_run("absdiff", 0, root=tmp_path)   # uses short steps via monkeypatch in the test
      (rd / "agent_1.py").write_text("def net(x0, x1):\n    return 0\n")
      with pytest.raises(RuntimeError, match="agent"):
          build_run("absdiff", 0, root=tmp_path, retrain=True)

  def test_counterexamples_saved_and_replayed(tmp_path):
      rd = build_run("absdiff", None, root=tmp_path)
      (rd / "agent_1.py").write_text("def net(x0, x1):\n    return x0 - x1\n")
      evaluate_run(rd)
      assert json.loads((rd / "counterexamples.json").read_text())

  def test_cli_check_exit_codes(tmp_path, capsys):
      rd = build_run("absdiff", None, root=tmp_path)
      good = tmp_path / "g.py"; good.write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
      bad = tmp_path / "b.py"; bad.write_text("def net(x0, x1):\n    return x0\n")
      assert main(["check", str(good), "--run", str(rd)]) == 0
      assert main(["check", str(bad), "--run", str(rd)]) == 1
  ```
  (`test_retrain_refuses_with_agent_attempts` monkeypatches `experiments.tasks.TASKS["absdiff"]` to `steps=50`.)
- [ ] **Step 2: Run** `uv run pytest tests/test_run.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `nndc/run.py` and `nndc/cli.py` (argparse subcommands).
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: run pipeline and nndc run/check CLI`.

---

### Task 8: Agent prepare / collect / mock, instruction file, HEURISTICS.md

**Files:**
- Create: `nndc/agent/__init__.py`, `nndc/agent/instruction.md`, `nndc/agent/check_template.py`, `nndc/agent/prepare.py`, `nndc/agent/collect.py`, `nndc/agent/mock.py`, `tests/test_agent.py`
- Modify: `nndc/cli.py` (add `agent prepare|collect|mock`)

**Interfaces:**
- Consumes: `run_dir`, `load_run`, `evaluate_run`, `Spec.blind`, `save_npz`, `check_program`, `format_result`.
- Produces:
  ```python
  # nndc/agent/prepare.py
  BLIND_ROOT = Path(tempfile.gettempdir()) / "nndc-blind"
  INDEX = Path("runs") / "blind_index.json"      # {blind_id: {"run": "<posix path of run dir>", "prepared": iso}}
  def prepare(rd: Path, blind_root: Path = BLIND_ROOT, index: Path = INDEX) -> Path   # returns blind folder
  def suggested_commands(blind: Path) -> list[str]

  # nndc/agent/collect.py
  def mangle(path: Path) -> str          # re.sub(r"[^A-Za-z0-9]", "-", str(path.resolve()))
  def find_transcripts(blind: Path, projects: Path = Path.home() / ".claude" / "projects") -> list[Path]
  def audit_transcripts(paths: list[Path], blind: Path) -> dict
      # {"transcript_found": bool, "outside_paths": [str], "model": str | None,
      #  "duration_s": float | None, "tokens": {"input": int, "output": int} | None}
  def collect(blind: Path, index: Path = INDEX, heuristics: Path = Path("HEURISTICS.md"),
              projects: Path = Path.home() / ".claude" / "projects") -> tuple[Path, int | None]
      # returns (run dir, attempt number or None if identical attempt already collected)

  # nndc/agent/mock.py
  def mock_agent(blind: Path, program_src: str, notes: str) -> None
      # writes agent.py, agent_notes.md, and runs the blind check.py once (appending to agent_checks.log)
  ```
- **Blind folder contents** (`<BLIND_ROOT>/<id>/`, where `id = "m" + secrets.token_hex(3)`):
  - `dump.py`
  - `spec.json` (from `spec.blind()`)
  - `weights.npz`
  - `INSTRUCTIONS.md` (copy of `nndc/agent/instruction.md`, with `{ALLOWLIST}` replaced by `allowlist_summary()`)
  - `check.py` (copy of `check_template.py`)
  - `_checker/` containing `__init__.py` plus copies of `nndc/spec.py`, `model.py`, `lang.py`, `check.py`
  - nothing else
- **`check.py` in the blind folder:** `python check.py agent.py`
  - first inserts its own directory at `sys.path[0]`, so `import _checker` also works under `python -I`, which does not add the script directory
  - loads `spec.json` and `weights.npz` via `_checker`
  - runs `check_program`
  - prints `format_result`
  - appends `"<iso time>\t<first line of result>\t<file>"` to `agent_checks.log`
  - exits 0 iff `snap_pass`
- **`nndc/agent/instruction.md`:** the author's wording from design §5 step 2, then a "Practical details" section with exactly these points:
  - Read `dump.py` and `spec.json`.
  - Write your version to `agent.py`, keeping `def net(x0, ...)` with the same arguments.
  - Check it with `python check.py agent.py` (PASS (snap) is enough; PASS (exact) is better).
  - The language your code must use: `{ALLOWLIST}`.
  - Write `agent_notes.md`: what you changed and why, then a section titled exactly `## Heuristics` listing heuristics that would apply to other networks.
  - Work only with files in this folder.
- **collect:**
  - requires `agent.py`
  - `n` = next free attempt number
  - if the sha256 of `agent.py` equals an existing `agent_<k>.py`, return `(rd, None)` and change nothing
  - copies to `agent_<n>.py`, `agent_<n>_notes.md`, `agent_<n>_checks.log`
  - writes `agent_<n>_meta.json`: `{"blind_id", "collected", "transcript_found", "outside_paths", "model", "duration_s", "tokens"}`. Time and cost per run come from here; the report shows them.
  - runs `evaluate_run(rd)`
  - appends to `HEURISTICS.md`: `## <task>/<hand|seedK> — agent_<n> (<blind_id>, <date>)` followed by the text under `## Heuristics` from the notes, or the whole notes prefixed with `(no "## Heuristics" section found)`.
- **audit:**
  - Each transcript is a JSONL file. Look in every `tool_use` block's `input` for strings that match `[A-Za-z]:[\\/]`, `/[a-z]/`, or contain `..`.
  - Flag any that do not resolve under the blind folder.
  - `model` is the first `message.model` value found.
  - `duration_s` is last minus first `timestamp` (ISO strings) across lines.
  - `tokens` sums `message.usage.input_tokens` and `message.usage.output_tokens`.
  - If the transcript is missing or unparseable, return `transcript_found: false` without raising.

- [ ] **Step 1: Write failing tests:**
  ```python
  def setup_run(tmp_path):
      rd = build_run("neareq", None, root=tmp_path / "runs")
      return rd, tmp_path / "blind", tmp_path / "runs" / "blind_index.json"

  def test_prepare_is_blind_and_self_contained(tmp_path):
      rd, broot, idx = setup_run(tmp_path)
      b = prepare(rd, blind_root=broot, index=idx)
      names = sorted(p.name for p in b.iterdir())
      assert names == ["INSTRUCTIONS.md", "_checker", "check.py", "dump.py", "spec.json", "weights.npz"]
      blob = "".join(p.read_text(errors="ignore") for p in b.rglob("*") if p.suffix in {".py", ".md", ".json"})
      assert "neareq" not in blob and str(rd) not in blob
      r = subprocess.run([sys.executable, "-I", "check.py", "dump.py"], cwd=b, capture_output=True, text=True)
      assert r.returncode == 0 and r.stdout.startswith("PASS")
      assert (b / "agent_checks.log").read_text().count("\n") == 1

  def test_blind_folder_with_spaces(tmp_path):
      rd, _, idx = setup_run(tmp_path)
      b = prepare(rd, blind_root=tmp_path / "dir with spaces", index=idx)
      assert subprocess.run([sys.executable, "-I", "check.py", "dump.py"], cwd=b).returncode == 0

  def test_collect_good_and_wrong_attempts(tmp_path):
      rd, broot, idx = setup_run(tmp_path)
      heur = tmp_path / "HEURISTICS.md"
      b1 = prepare(rd, blind_root=broot, index=idx)
      mock_agent(b1, "def net(x0, x1):\n    return abs(x0 - x1) < 0.2\n", "Snapped.\n## Heuristics\n- relu pair -> abs\n")
      rd_, n = collect(b1, index=idx, heuristics=heur, projects=tmp_path / "noproj")
      assert rd_ == rd and n == 1
      b2 = prepare(rd, blind_root=broot, index=idx)
      mock_agent(b2, "def net(x0, x1):\n    return x0 > x1\n", "Wrong on purpose.")
      _, n2 = collect(b2, index=idx, heuristics=heur, projects=tmp_path / "noproj")
      res = json.loads((rd / "results.json").read_text())
      assert res["versions"]["agent_1"]["check"]["snap_pass"] and res["versions"]["agent_1"]["task"]["recovered"]
      assert not res["versions"][f"agent_{n2}"]["check"]["snap_pass"]
      meta = json.loads((rd / "agent_1_meta.json").read_text())
      assert meta["transcript_found"] is False
      text = heur.read_text()
      assert "relu pair -> abs" in text and '(no "## Heuristics" section found)' in text

  def test_collect_same_attempt_twice_is_noop(tmp_path):
      rd, broot, idx = setup_run(tmp_path)
      b = prepare(rd, blind_root=broot, index=idx)
      mock_agent(b, "def net(x0, x1):\n    return abs(x0 - x1) < 0.2\n", "x")
      collect(b, index=idx, heuristics=tmp_path / "H.md", projects=tmp_path / "np")
      assert collect(b, index=idx, heuristics=tmp_path / "H.md", projects=tmp_path / "np")[1] is None

  def test_audit_flags_outside_paths(tmp_path):
      b = tmp_path / "blind" / "m1"; b.mkdir(parents=True)
      proj = tmp_path / "projects" / mangle(b); proj.mkdir(parents=True)
      lines = [{"timestamp": "2026-10-27T10:00:00Z",
                "message": {"model": "claude-x", "usage": {"input_tokens": 100, "output_tokens": 10},
                            "content": [{"type": "tool_use", "name": "Read", "input": {"file_path": str(b / "dump.py")}}]}},
               {"timestamp": "2026-10-27T10:01:30Z",
                "message": {"usage": {"input_tokens": 50, "output_tokens": 5},
                            "content": [{"type": "tool_use", "name": "Read",
                                         "input": {"file_path": str(tmp_path / "runs" / "secret.json")}}]}}]
      (proj / "s.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
      a = audit_transcripts(find_transcripts(b, projects=tmp_path / "projects"), b)
      assert a["transcript_found"] and a["model"] == "claude-x"
      assert a["duration_s"] == 90 and a["tokens"] == {"input": 150, "output": 15}
      assert a["outside_paths"] == [str(tmp_path / "runs" / "secret.json")]
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_agent.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the modules and the CLI:
  - `nndc agent prepare <run_dir>... | --all`: `--all` means every run in `all_runs()`. Print each blind path plus `suggested_commands`: `cd "<blind>"` and `claude "Read INSTRUCTIONS.md and follow it."`.
  - `nndc agent collect <blind_dir>`.
  - `nndc agent mock <blind_dir> --program FILE [--notes FILE]`.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: blind agent folders, collect with audit, mock agent, instruction file`.

---

### Task 9: Report, spike gate, README stub

**Files:**
- Create: `nndc/report.py`, `README.md`, `tests/test_report.py`
- Modify: `nndc/cli.py` (add `report`, `gate`)

**Interfaces:**
- Consumes: `all_runs`, `results.json` schema (Task 7), `agent_<n>_meta.json` (Task 8), `Spec` real names.
- Produces:
  ```python
  def build_report(root: Path = RUNS) -> str          # markdown; `nndc report` writes REPORT.md
  def with_real_names(src: str, spec: Spec) -> str    # replace \bx<k>\b with real names for display
  SPIKE_RUNS = ["xor/seed0", "absdiff/seed0", "neareq/seed0"]
  def spike_gate(root: Path = RUNS) -> tuple[bool, list[str]]
  ```
- `REPORT.md` contains:
  - **One table**, one row per run × version (`dump`, `agent_1`, …), with columns: run | training (acc or R², learned) | version | check (PASS exact / PASS snap / FAIL / ERROR) | label | max err | disagreements outside / inside (zone %) | task in-range / out-of-range | recovered | nodes (and % of dump) | audit (ok / outside paths / no transcript) | Claude time and tokens.
  - **One section per run that has agent attempts**, showing `dump.py` and the best attempt (snap pass first, then fewest nodes), both via `with_real_names`, plus the attempt's notes.
- Spike gate:
  - An attempt is **ok** if `snap_pass` and `nodes ≤ 0.5 × dump nodes`.
  - The gate passes iff at least 2 of the 3 spike runs have an ok attempt, and at least 1 has an attempt with `recovered`.
  - Attempts flagged by the audit (`outside_paths` non-empty) don't count.
  - Each output line names the run, its best attempt, and which criterion it met or failed.
- `README.md` stub: one paragraph of intent (from design §1), the commands (`uv sync`, `uv run pytest`, `uv run nndc run all`, the prepare → run Claude → collect flow, `uv run nndc report`), and a "Prior work" placeholder line pointing to the design doc.

- [ ] **Step 1: Write failing tests:**
  ```python
  def test_real_names_substitution():
      spec = Spec(task="t", output="regression", inputs=[InputSpec("x0","income","float",0,1), InputSpec("x1","rate","float",0,1)])
      # whole tokens only, single pass: x10 is not an input and stays
      assert with_real_names("return x0 + x1 * x10", spec) == "return income + rate * x10"

  def test_report_lists_versions(tmp_path):
      rd = build_run("absdiff", None, root=tmp_path)
      (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n"); evaluate_run(rd)
      md = build_report(tmp_path)
      assert "absdiff/hand" in md and "agent_1" in md and "abs(a - b)" in md

  def test_spike_gate(tmp_path):
      for t in ["xor", "absdiff", "neareq"]:
          rd = tmp_path / t / "seed0"; rd.mkdir(parents=True)
          ok = t != "xor"
          (rd / "results.json").write_text(json.dumps({"versions": {
              "dump": {"size": {"nodes": 100}, "check": {"snap_pass": True}, "task": {"recovered": False}},
              "agent_1": {"size": {"nodes": 20 if ok else 90}, "check": {"snap_pass": True}, "task": {"recovered": t == "absdiff"}}}}))
      passed, lines = spike_gate(tmp_path)
      assert passed and len(lines) >= 3
  ```
- [ ] **Step 2: Run** `uv run pytest tests/test_report.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `nndc/report.py`, the CLI `report` (writes `REPORT.md`) and `gate` (prints lines, exits 0 iff passed), and `README.md`.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: report, spike gate, README stub`.

---

### Task 10: Generate all step-1 runs and prepare the spike

**Files:**
- Create (generated): `runs/**`, `REPORT.md`

- [ ] **Step 1: Run** `uv run nndc run all`. Expected:
  - 11 hand runs and 18 trained runs (xor, absdiff, neareq, window, median3, tax × seeds 0–2).
  - Every dump `exact_pass`.
  - Training quality is printed for each trained run. Runs that fail to learn are kept and marked `learned: false`, never replaced.
- [ ] **Step 2: Run** `uv run nndc report`. Then open `REPORT.md` and verify that every row's dump says PASS exact, and that training quality is filled in for the trained runs.
- [ ] **Step 3: Commit** `runs/ REPORT.md`, with message `data: step-1 runs (hand + trained seeds 0-2)`.
- [ ] **Step 4: Prepare the spike** with `uv run nndc agent prepare runs/xor/seed0 runs/absdiff/seed0 runs/neareq/seed0`. Expected: three blind paths printed, each with the suggested `claude` command.
- [ ] **Step 5: HUMAN STEP.** The user runs Claude Code in each of the three blind folders (as many attempts as they like). After each run: `uv run nndc agent collect <blind path>`.
- [ ] **Step 6: Run** `uv run nndc report`, then `uv run nndc gate`.
  - If the gate passes, steps 3–6 get their own plans.
  - If it fails: adjust `nndc/agent/instruction.md` once, record the change in `REPORT.md`, prepare fresh blind folders, and repeat Steps 5–6.
  - If it still fails, stop and write it up as a result.
- [ ] **Step 7: Commit** `runs/ REPORT.md HEURISTICS.md nndc/agent/instruction.md`, with message `data: spike-gate Claude Code attempts`.
