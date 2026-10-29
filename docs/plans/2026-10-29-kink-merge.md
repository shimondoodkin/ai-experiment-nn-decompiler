# Kink Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `kink-merge` AST route for one-layer sigmoid nets, written as a new version `ast2` with its own
counterexample file. It is developed on the existing sigmoid seeds 0–2, frozen, and then run once on 15 fresh
held-out nets (seeds 3–5), with a report section that shows which kinks cancel and which stay.

**Architecture:**
- **Route:** a new module, `nndc/heuristics/kinks.py`. It expands every sigmoid into ReLU kinks, merges the
  kinks by line, picks the strongest lines, refines and snaps them, and writes the program through the existing
  ReLU motif code. `refactor` calls it only for nets whose hidden units are all sigmoid.
- **Versioning:** `run_ast` gets a version name (`ast2`), and `evaluate_run` checks `ast2` against its own
  `counterexamples_ast2.json`.
- **Report:** `report.py` gains a "Kink merge (ast2)" section in compare mode.
- **Data:** development runs, the freeze, held-out builds, agents and the held-out tool run each go in their
  own commits, in the design's order.

**Tech Stack:** Python 3.13, NumPy, SciPy `linprog` (HiGHS, already used by `knot_fit`), pytest, `uv`.

**Spec:** `docs/design/2026-10-29-kink-merge-design.md`

## Global Constraints

- **Older runs stay.** No `ast.py`, `ast_trace.json`, `agent_*`, checks log or `counterexamples.json` under
  `runs*/` is rewritten by this work.
  - New files are fine: `ast2.py`, `ast2_trace.json`, `counterexamples_ast2.json`, new seed folders.
  - `results.json` and `REPORT.md` are regenerated files.
  - Destructive commands only on scratch copies, never on the repo, its parent or `runs*/`.
- **ReLU results unchanged:** `experiments/equivalence.py` (`copy`, `evaluate`, `ast`, `compare`, `redump`) on
  a scratch copy reports 0 differences.
- **Sigmoid results unchanged:** re-evaluating an existing sigmoid run leaves every version's verdict
  except `ast2` as it was.
- **Fixed parameters (design Goal 1):**
  - pieces `K = 17`;
  - cosine `0.995`;
  - merge offsets `(0.02, 0.1)`;
  - `m = 1..8`;
  - offset search ±0.1 in 21 steps, 3 rounds;
  - 20,000 fit samples with a fixed seed, plus the box corners;
  - `SNAP_TOLS = (0.05, 0.02, 0.01, 0.005)`;
  - "nearly cancelled": at least 2 kinks, and net strength below a quarter of the sum of absolute strengths.
- **The route applies only** to nets with exactly one hidden layer, all units sigmoid, and a regression or
  binary output. Otherwise it returns `None` with a trace reason.
- **Blindness:** the route reads only `spec` (blind) and `mlp`.
- **Commits:** code and `data:` commits are separate; no attribution lines.

## Review Focus

1. **A line that leaves the box after snapping** (an edge kink at 199.6 snaps to 200). It must fold into the
   affine part, not produce a dead `max(...)` term or an error. Test: `test_snap_line_leaving_box_folds`.
   (`nicest_in` returns the integer nearest the middle, so the tax edge kink at 193 stays at 193 at every
   tolerance.)
2. **A kink whose normal points the "wrong" way,** with `max(y,0) = max(−y,0) + y`. The expanded function must
   equal the net's chord interpolation everywhere in the box. Test: `test_expand_matches_chords`.
3. **Inputs with very different ranges** (tax 0–200 vs absdiff −3..3). Positions are snapped with
   `tol·span`, and coefficients with a relative tolerance. Test: `test_snap_real_coordinates_tax`.
4. **A net the route must refuse:** two hidden layers, argmax output, a ReLU or tanh layer. It returns `None`,
   and ReLU `refactor` output is unchanged. Test: `test_kink_merge_refuses`.
5. **`ast2` counterexamples leaking** into `counterexamples.json`, or old versions being checked on them. Test:
   `test_ast2_cex_separate`.

---

### Task 1: Expand and merge (`nndc/heuristics/kinks.py`)

**Files:**
- Create: `nndc/heuristics/kinks.py`
- Test: `tests/test_kinks.py`

**Interfaces:**
- Produces:
  - `@dataclass Kink: normal: np.ndarray; offset: float; strength: float`. In unit-box coordinates the term
    is `strength * max(normal @ u + offset, 0)`, with `|normal| = 1` and the first nonzero component positive.
  - `@dataclass Expansion: const: float; linear: np.ndarray; kinks: list[Kink]`. Unit-box affine part plus
    kinks.
  - `@dataclass Line: normal; offset; strength; gross: float; count: int`.
  - `applies(spec: Spec, mlp: MLP) -> str | None`. `None` when the route applies, else a reason string.
  - `expand(spec: Spec, mlp: MLP, pieces: int = PIECES) -> Expansion`.
  - `evaluate_expansion(e: Expansion, U: np.ndarray) -> np.ndarray`.
  - `merge_lines(kinks: list[Kink], off: float, cos_min: float = COS_MIN) -> list[Line]`. Sorted by
    `|strength|`, descending.
  - `nearly_cancelled(line: Line) -> bool`.
  - `unit_box(spec) -> tuple[np.ndarray, np.ndarray]`. Returns `(lo, span)`.
  - Constants: `PIECES=17`, `COS_MIN=0.995`, `MERGE_OFFS=(0.02, 0.1)`, `MAX_LINES=8`, `OFF_RANGE=0.1`,
    `OFF_STEPS=21`, `OFF_ROUNDS=3`, `FIT_SAMPLES=20_000`, `FIT_SEED=0`, `CANCEL_FRAC=0.25`.

- [ ] **Step 1: Write the failing tests** in `tests/test_kinks.py`

```python
def test_expand_matches_chords():
    # 2-input, 3-unit sigmoid net with weights of mixed signs (some normals flip)
    mlp = MLP([np.array([[2., -1.], [-1.5, 0.5], [0., 3.]], np.float32), np.array([[1., -2., 0.5]], np.float32)],
              [np.array([0.3, -0.2, 0.1], np.float32), np.array([0.4], np.float32)], ["sigmoid"])
    spec = Spec(task="", output="regression", inputs=[InputSpec(f"x{i}", f"x{i}", "float", -3, 3) for i in range(2)])
    e = expand(spec, mlp, pieces=17)
    U = np.random.default_rng(0).uniform(0, 1, (5000, 2))
    want = chord_reference(mlp, spec, U, 17)        # helper in the test: np.interp per unit on its knot grid
    np.testing.assert_allclose(evaluate_expansion(e, U), want, atol=1e-9)
    assert all(k.normal[np.flatnonzero(np.abs(k.normal) > 1e-9)[0]] > 0 for k in e.kinks)

def test_expansion_close_to_net_absdiff():
    spec, mlp = load_run(Path("runs_sigmoid/absdiff/seed0")); blind = Spec.from_json(spec.blind())
    lo, span = unit_box(blind); X = uniform_sample(blind, np.random.default_rng(1), 50_000)
    net = mlp.forward(X.astype(np.float32))[:, 0]
    err = np.abs(evaluate_expansion(expand(blind, mlp), (X - lo) / span) - net).max()
    assert err <= 0.02 * np.ptp(net)

def test_merge_absdiff_lines_parallel():
    spec, mlp = load_run(Path("runs_sigmoid/absdiff/seed0")); blind = Spec.from_json(spec.blind())
    lines = merge_lines(expand(blind, mlp).kinks, off=0.02)
    assert 8 <= len(lines) <= 64
    assert all(abs(abs(l.normal @ np.array([1, -1]) / np.sqrt(2)) - 1) < 0.01 for l in lines[:4])
    assert any(nearly_cancelled(l) for l in lines)

def test_merge_same_line_adds():
    k = [Kink(np.array([1., 0.]), -0.5, 2.0), Kink(np.array([1., 0.]), -0.51, -1.5)]
    (line,) = merge_lines(k, off=0.02)
    assert line.strength == pytest.approx(0.5) and line.gross == pytest.approx(3.5) and line.count == 2
    assert nearly_cancelled(line)                      # 0.5 < 0.25 * 3.5
    k2 = [Kink(np.array([1., 0.]), -0.5, 2.0), Kink(np.array([1., 0.]), -0.51, -0.2)]
    assert not nearly_cancelled(merge_lines(k2, off=0.02)[0])   # 1.8 >= 0.25 * 2.2
```

Also add `test_box_folding`: a unit whose line never crosses the box leaves no kink, and the expansion still
matches the chords.

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_kinks.py -v`
Expected: FAIL, ImportError `nndc.heuristics.kinks`.

- [ ] **Step 3: Implement it in `nndc/heuristics/kinks.py`**

Follow the spike (`pieces.py` in the session scratchpad):
- **Expand:** knots uniform in `z` over the unit's corner `[z_min, z_max]`, with chord slopes. A unit with
  `z_min == z_max` is constant.
- **Flip** (`max(y,0) = max(−y,0) + y`) into the affine part. Lines that never cross the box (corner
  evaluation) are folded or dropped.
- **Merge:** greedy first match on `normal·normal > cos_min` and `|Δoffset| < off`. Strength is the sum, and
  normal and offset are the |strength|-weighted means, with the normal renormalised.
- `applies`:
  - `"not one hidden layer"` when `len(mlp.weights) != 2`;
  - `"not all sigmoid"` when `mlp.acts != ["sigmoid"]`;
  - `"argmax output"` when `spec.output == "argmax"`.

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_kinks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/heuristics/kinks.py tests/test_kinks.py
git commit -m "feat: kink expansion and line merging for one-layer sigmoid nets"
```

### Task 2: Refine, snap, write (`kink_merge`)

**Files:**
- Modify: `nndc/heuristics/kinks.py`
- Test: `tests/test_kinks.py`

**Interfaces:**
- Consumes: Task 1's API; `_minimax(A, y)` and `SNAP_TOLS` from `nndc/heuristics/sigmoid.py`; `nicest_in`
  from `nndc/heuristics/decision_list.py`; `nice` from `nndc/heuristics/snap.py`; `as_relu_sum`,
  `motif_expr` (`nndc/heuristics/motifs.py`); `program_from_expr` (`nndc/heuristics/cleanup.py`);
  `SymNet`, `Unit`, `Affine`, `to_program` (`nndc/symnet.py`); `check_program` (`nndc/check.py`).
- Produces:
  - `refine(lines: list[Line], U: np.ndarray, y: np.ndarray) -> tuple[list[Line], np.ndarray, float]`. Returns
    the lines with refined offsets, the coefficients `[c, a…, s…]`, and the max error.
  - `to_real(lines, coefs, spec) -> RealFit`, where `@dataclass RealFit: const: float; linear: dict[str, float];
    terms: list[tuple[dict[str, float], float, float]]`. Each term is `(normal over input names, t, strength)`,
    meaning `strength * max(normal·x − t, 0)`, and each normal's largest |component| is 1.
  - `snap_fit(fit: RealFit, spec, tol: float, U, y) -> RealFit`. The design's Goal 1.7 sub-steps 1–4, in order:
    direction, position (`nicest_in(t − tol·span, t + tol·span)` with `span` the largest input range),
    LP refit, coefficients (`nicest_in(s − tol·|s|, s + tol·|s|)`), then the constant as the midrange of the
    residual. Terms whose line leaves the box are folded into the affine part.
  - `fit_program(fit: RealFit, spec) -> str`. Builds a one-layer `SymNet` of `relu` units, then
    `program_from_expr(motif_expr(rs), spec)` when `as_relu_sum` accepts it, else `to_program`.
  - `kink_merge(spec: Spec, mlp: MLP) -> tuple[str | None, dict]`. Returns the first passing program and the
    trace dict:
    - `applies` (reason or `None`);
    - `per_offset`: a list of `{off, kinks, lines, cancelled}`;
    - `winner`: `{off, m, tol, lines: [{normal, t, strength}]}`, or `None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_kink_merge_tax_passes():
    spec, mlp = load_run(Path("runs_sigmoid/tax/seed0")); blind = Spec.from_json(spec.blind())
    src, trace = kink_merge(blind, mlp)
    assert src is not None and check_program(src, mlp, blind).snap_pass
    assert trace["winner"]["m"] <= 8 and ("relu" in src or "max" in src)

def test_snap_real_coordinates_tax():
    spec, U, y, fit = _tax_like()        # helper: 1-input spec x0 in [0, 200], data and an exact RealFit
    snapped = snap_fit(fit, spec, 0.005, U, y)
    assert 192 < snapped.terms[-1][1] < 194
    assert {round(t, 6) for _, t, _ in snapped.terms[:2]} == {20.0, 80.0}

def test_snap_line_leaving_box_folds():
    spec, U, y, fit = _tax_like(edge=199.6)
    snapped = snap_fit(fit, spec, 0.005, U, y)  # (198.6, 200.6): nicest_in picks 200, the box edge
    assert len(snapped.terms) == 2
    assert fit_program(snapped, spec).count("relu(") + fit_program(snapped, spec).count("max(") <= 2

def test_kink_merge_refuses():
    for sizes, acts, out in [([2, 4, 4, 1], ["sigmoid", "sigmoid"], "regression"),
                             ([2, 4, 3], ["sigmoid"], "argmax"),
                             ([2, 4, 1], ["relu"], "regression"),
                             ([2, 4, 1], ["tanh"], "regression")]:
        src, trace = kink_merge(_spec(2, out, n_out=sizes[-1]), _net(sizes, acts))
        assert src is None and trace["applies"]
```

The two snap tests use a test helper, `_tax_like(edge=193.3)`. It builds a 1-input spec `x0 ∈ [0, 200]` and
`y = 0.1x + 0.1·max(x−20.2,0) + 0.2·max(x−79.8,0) − 0.3·max(x−edge,0)` on 2001 points, and returns that `y`
with the exact `RealFit` for it.

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_kinks.py -v`
Expected: the four new tests FAIL (names missing).

- [ ] **Step 3: Implement it**

The route's order:
1. For each `off` in `MERGE_OFFS` and each `m = 1..MAX_LINES`, refine the top `m` lines.
2. Try the snap ladder `SNAP_TOLS`, coarsest first, then the unsnapped fit.
3. Each candidate goes through `check_program(src, mlp, spec)`.
4. Return the first passing one, smallest `m` first; for equal `m`, `off = 0.02` before `0.1`.

Fit data: `FIT_SAMPLES` uniform samples (`np.random.default_rng(FIT_SEED)`) plus the box corners, in unit
coordinates. `y` is the net's raw output.

- [ ] **Step 4: Runtime check** (design Constraint)

Run: `uv run python -c "import time; ...kink_merge on runs_sigmoid/absdiff/seed0..."`
Expected: under 2 minutes.

If it is slower, set `SEARCH_SAMPLES = 2000`, a fixed-seed subset used only inside the offset search; the final
LP and the snap stay on all 20,000. Record this as a ledger ruling and in the plan's Amendments. This is the
one parameter change the design allows, and it is made on runtime, not results.

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/test_kinks.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nndc/heuristics/kinks.py tests/test_kinks.py
git commit -m "feat: kink-merge route — refine, snap in real coordinates, motif-written program"
```

### Task 3: Route in `refactor`, `ast2` version, own counterexamples

**Files:**
- Modify: `nndc/ast_tool.py` (`ROUTES`, `refactor`, `run_ast`)
- Modify: `nndc/run.py` (`evaluate_run`)
- Modify: `nndc/cli.py` (`ast --name`)
- Test: `tests/test_kinks.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: `kink_merge(spec, mlp) -> (str | None, dict)`.
- Produces:
  - `ROUTES` gains `"kink-merge"`, appended last;
  - the `refactor` trace gains a key `kink_merge` (the route's trace dict) for sigmoid nets;
  - `run_ast(rd: Path, name: str = "ast") -> dict` writes `{name}.py` and `{name}_trace.json`, and calls
    `record_git(rd, name)`;
  - `evaluate_run` checks `ast2` when `ast2.py` exists. Its extra inputs are the union of
    `counterexamples.json` and `counterexamples_ast2.json`; its new counterexamples go only to
    `counterexamples_ast2.json`;
  - CLI: `nndc ast RUNS… --name ast2` (default `ast`).

- [ ] **Step 1: Write the failing tests**

```python
def test_ast2_cex_separate(tmp_path):            # tests/test_run.py
    rd = build_run("absdiff", 0, root=tmp_path, act="sigmoid")
    before = (rd / "counterexamples.json").read_bytes() if (rd / "counterexamples.json").exists() else b""
    (rd / "ast2.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")   # fails vs the net
    res = evaluate_run(rd)
    assert not res["versions"]["ast2"]["check"]["snap_pass"]
    assert json.loads((rd / "counterexamples_ast2.json").read_text())
    after = (rd / "counterexamples.json").read_bytes() if (rd / "counterexamples.json").exists() else b""
    assert after == before

def test_run_ast_name_keeps_ast(tmp_path):        # tests/test_run.py
    rd = build_run("tax", 0, root=tmp_path, act="sigmoid")
    run_ast(rd); old = (rd / "ast.py").read_bytes()
    run_ast(rd, name="ast2")
    assert (rd / "ast.py").read_bytes() == old and (rd / "ast2.py").exists()
    assert "ast2:" in (rd / "run_git.txt").read_text()

def test_refactor_trace_has_kink_merge():         # tests/test_kinks.py
    spec, mlp = load_run(Path("runs_sigmoid/tax/seed0")); blind = Spec.from_json(spec.blind())
    _, trace = refactor((Path("runs_sigmoid/tax/seed0") / "dump.py").read_text(), blind, mlp)
    assert "kink_merge" in trace and any(c["route"] == "kink-merge" for c in trace["candidates"])
```

Extend `test_kink_merge_refuses` with the ReLU case through `refactor`. There must be no `kink-merge`
candidate and no `kink_merge` trace key.

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_kinks.py tests/test_run.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement it**

In `refactor`, inside the existing `if any(u.act not in ("relu", "linear") …)` block, add:

```python
src_k, tr_k = kink_merge(spec, mlp); trace_extra["kink_merge"] = tr_k
add("kink-merge", lambda: src_k, ["K1"])
```

Merge `trace_extra` into the returned trace. `cmd_ast` prints `res["versions"][args.name]` and reads
`{name}_trace.json`.

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: ReLU equivalence**

Run, with `C` a scratch directory under the session scratchpad:

```bash
uv run python -m experiments.equivalence copy --copy C
uv run python -m experiments.equivalence evaluate --copy C
uv run python -m experiments.equivalence ast --copy C
uv run python -m experiments.equivalence compare --copy C
uv run python -m experiments.equivalence redump
```

Expected: `compare` and `redump` report 0 differences.

- [ ] **Step 6: Commit**

```bash
git add nndc/ast_tool.py nndc/run.py nndc/cli.py tests/test_kinks.py tests/test_run.py
git commit -m "feat: kink-merge route in refactor; ast2 version with its own counterexamples"
```

### Task 4: Agent scratch directory

**Files:**
- Modify: `nndc/agent/prepare.py`, `nndc/agent/instruction.md`
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces: `prepare` creates `<blind>/scratch/`. `instruction.md` gets one line under "Work only with the
  files in this folder": `- Put any scratch scripts or files in the scratch/ folder here.`

- [ ] **Step 1: Write the failing test**

`test_prepare_scratch_dir(tmp_path)`. Prepare a tmp run into a tmp blind root, then assert that
`(blind / "scratch").is_dir()` and that `"scratch/"` is in `INSTRUCTIONS.md`.

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_agent.py::test_prepare_scratch_dir -v`
Expected: FAIL.

- [ ] **Step 3: Implement it.** Then run `uv run pytest -q`. Expected: all pass. The collect audit must not
  flag `scratch/` paths, since they are inside the folder.

- [ ] **Step 4: Commit**

```bash
git add nndc/agent/prepare.py nndc/agent/instruction.md tests/test_agent.py
git commit -m "feat: blind folders get their own scratch directory"
```

### Task 5: Report section "Kink merge (ast2)"

**Files:**
- Modify: `nndc/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `results.json` versions `ast2` and `formula`; `ast2_trace.json` (`trace["kink_merge"]`,
  `trace["winner"]`).
- Produces:
  - `kink_md(root: Path) -> str`, added to `build_report` in compare mode after `activation_compare_md`. It
    returns `""` when no run has `ast2.py`.
  - It has a table with one row per run that has `ast2`:
    - run, split (development = seeds 0–2, held-out = seeds 3–5);
    - the winning route, its verdict vs the net, nodes, recovered;
    - kinks → lines → nearly cancelled → kept, for the winner's `off`;
    - the kept lines in real coordinates, `t` and strength rounded to 3 significant figures.
  - Then the reproduction count for `ast2` (passes and recovered, development vs held-out). Then, per formula
    net, the `ast2` program next to `formula.py` in two code blocks.

- [ ] **Step 1: Write the failing tests**

- `test_kink_md_rows(tmp_path)`: write two fake run folders with `results.json` and `ast2_trace.json`. Assert
  one row each, the split labels, and the counts text `128 → 32 → 14 → 4`.
- `test_kink_md_empty(tmp_path)`: no `ast2` gives `""`.
- `test_default_report_unchanged` (existing, slow) must still pass.

- [ ] **Step 2: Run them and see them fail.** Then **Step 3: implement it** and **Step 4: run**
  `uv run pytest tests/test_report.py -v`; expected PASS.

- [ ] **Step 5: Commit**

```bash
git add nndc/report.py tests/test_report.py
git commit -m "feat: report section for kink-merge (ast2)"
```

### Task 6: Development run, then the freeze

**Files:**
- Data: `runs_sigmoid/{xor,window,neareq,absdiff,tax}/seed{0,1,2}/ast2.py`, `ast2_trace.json`,
  `counterexamples_ast2.json`, `results.json`, `run_git.txt`
- Modify: this plan's "Amendments" section

- [ ] **Step 1: Run the tool on the development set**

```bash
uv run nndc ast --name ast2 runs_sigmoid/xor/seed0 runs_sigmoid/xor/seed1 runs_sigmoid/xor/seed2 \
  runs_sigmoid/window/seed0 … runs_sigmoid/tax/seed2
```

That is all 15 folders: 5 tasks × seeds 0–2. Expected: one line per run, with a verdict and a route.

- [ ] **Step 2: Check that old results did not change.** For every development run, every version except
  `ast2` keeps its verdict and nodes in `results.json`. Compare against `git show HEAD:<path>`.

- [ ] **Step 3: Bug fixes only.** If a run shows a bug, fix it with a failing test first:
  - a crash;
  - a program that does not parse;
  - a trace that contradicts the program;
  - a refused net that should apply.

  Record each fix in Amendments. No parameter or move changes; the one exception is Task 2 Step 4's
  `SEARCH_SAMPLES`.

- [ ] **Step 4: Data commit**

```bash
git add runs_sigmoid
git commit -m "data: ast2 (kink-merge) on sigmoid development runs"
```

- [ ] **Step 5: Freeze.** Commit any remaining code (`freeze: kink-merge tool`), or note the current HEAD as the
  freeze if there is none. Record the commit in the plan's Amendments.

### Task 7: Held-out nets, agents, frozen tool run

**Files:**
- Data: `runs_sigmoid/{xor,window,neareq,absdiff,tax}/seed{3,4,5}/`, `runs_sigmoid/blind_index.json`,
  `runs_sigmoid/HEURISTICS_heldout.md`

- [ ] **Step 1: Build the 15 nets** (only after Task 6's freeze)

```bash
for t in xor window neareq absdiff tax; do for s in 3 4 5; do
  uv run nndc run $t --seed $s --act sigmoid --root runs_sigmoid; done; done
```

Expected: 15 new folders, each with `formula.py`. Commit: `data: sigmoid held-out nets for kink-merge
(seeds 3–5)`.

- [ ] **Step 2: Blind agents.** For each of the 15 folders:
  - `uv run nndc agent prepare <run> --index runs_sigmoid/blind_index.json`;
  - one fresh general-purpose subagent, one wave of 15. The prompt is only "Work only inside `<folder>`. Read
    INSTRUCTIONS.md there and follow it.";
  - keep a folder → agent-id map;
  - when every agent has finished, run `uv run nndc agent collect <blind> --transcript <jsonl> --index
    runs_sigmoid/blind_index.json --heuristics runs_sigmoid/HEURISTICS_heldout.md`;
  - check that no `agent.py` changed after collection.

  Commit: `data: blind agents on kink-merge held-out nets`.

- [ ] **Step 3: The frozen tool, once.** Run `uv run nndc ast --name ast2` on the 15 folders. Check that each
  `run_git.txt` `ast2:` line names the freeze commit, or a data-only commit on top of it. Commit:
  `data: ast2 (frozen kink-merge) on held-out nets`.

### Task 8: Report and docs

**Files:**
- Data: `runs_sigmoid/REPORT.md`
- Modify: `CLAUDE.md` (freeze line; `ast2` and `counterexamples_ast2.json` in Sigmoid runs), this plan's
  Amendments

- [ ] **Step 1:** Run `uv run nndc report --root runs_sigmoid --out runs_sigmoid/REPORT.md --compare-to runs`
  in the background, since it re-runs the checker. Expected: the "Kink merge (ast2)" section has 30 rows: 15
  development and 15 held-out.
- [ ] **Step 2:** Re-run `evaluate_run` over all `runs_sigmoid` runs before the data commit (CLAUDE.md
  concurrent writes). Then run `uv run pytest -q`; expected all pass.
- [ ] **Step 3:** Update CLAUDE.md:
  - Freezes: add `kink-merge: <commit>`;
  - Sigmoid runs: add `ast2`, `counterexamples_ast2.json`, the held-out seeds 3–5 and the route;
  - the Unit.act line is unchanged.

  Commit the docs (`docs: kink-merge freeze and notes`), then the report (`data: runs_sigmoid/REPORT.md
  with kink merge`).

## Amendments during execution


- **Task 2, runtime (2026-10-29):** `kink_merge` on absdiff seed0 took over 2 minutes, so `SEARCH_SAMPLES = 2000`
  was set. The offset search fits on the first 2000 fit samples; the final LP and the snap use all 20,000 plus
  the corners. With it, absdiff seed0 takes 40 s and tax seed0 8 s. The change was made on runtime alone,
  before the development run.
- **Task 2:** `fit_program` builds the motif input (`ReluSum`) directly instead of a `SymNet` plus
  `as_relu_sum`; the result is the same.
- **Task 6, development run (2026-10-29):** 15 runs, all pass.
  - Kink-merge won on neareq seeds 0–2 (78/116/78 nodes), absdiff seeds 0–1 (80/95) and tax seed 2 (20).
  - Absdiff seed 2 fell back to the dump (structural-exact, 238), as the spike's 4.0% suggested.
  - Tax seeds 0–1 keep knot-fit (22). Xor and window keep truth-table and boundary-scan.
  - No bug fixes were needed.
  - The committed `results.json` of 6 runs were stale (`n_inputs` +20 against their committed
    `counterexamples.json`). Verdicts, nodes and recovered are unchanged for every older version.
- **Freeze: `93d7199`** (the last code commit; the development `ast2:` lines record it). Held-out nets are built
  after this point.
