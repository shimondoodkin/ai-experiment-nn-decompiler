# Activation Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every hidden layer of a net can be `relu`, `sigmoid` or `tanh`. The activation is carried
through model, dump, language, checker, symbolic net and AST tool. Pieces that only work for ReLU
refuse other nets. Nothing that exists today changes.

**Architecture:**
- `nndc/model.py` holds the one definition of each activation (`activate`, `LIPSCHITZ`) and the
  per-layer list `MLP.acts`. Every other module reads activations from there.
- The symbolic net replaces `Unit.relu: bool` with `Unit.act: str`, so a missed site crashes instead
  of silently treating sigmoid as linear.
- A small equivalence script proves nothing changed. It is run on the current code first (Task 1) and
  again at the end (Task 9).

**Tech Stack:** Python 3.13, NumPy, torch (CPU, training only), pytest, `uv`.

**Spec:** `docs/design/2026-10-28-activations-design.md`

## Global Constraints

- **Blind-folder modules:** `nndc/model.py`, `nndc/lang.py`, `nndc/check.py` and `nndc/spec.py` stay
  stdlib + NumPy only.
- **Stable sigmoid:** no overflow warnings for large `|z|`.
- **Nothing changes for ReLU:**
  - ReLU dumps, `weights.npz` written for all-ReLU nets, and every existing `results.json`,
    `ast.py`, `ast_trace.json`, `repair/results.json` and `*_net.npz` stay identical. Judge with
    `git diff`, or with CRLF stripped.
  - The size statistic keys stay `lines`, `nodes`, `constants`, `relus`, `ifs`.
- **Held-out runs:** the AST tool is not run on held-out runs (formula seed 2, Iris seeds 3–5), not even
  in a throwaway copy, and not on `runs_bugs/`.
- **Commits:** code commits only, no `data:` commit. No attribution lines in commit messages.

## Review Focus

1. **A ReLU-path float changes by one ulp:** a rewritten `np.maximum` (dtype, operand order) changes
   a dump, a rounding bound or the LP features. Pinned by Task 1's baseline and Task 9's comparison,
   and by the rounding-bound reference test in Task 4.
2. **A sigmoid unit is treated as linear, or as ReLU, by a heuristic:** for example
   `prune_dead_and_linear` inlining it, or `fold` / `as_relu_sum` reading it. Pinned by the Task 6
   tests `test_prune_keeps_sigmoid_units` and `test_relu_only_routes_skip_sigmoid`.
3. **Large `|z|` in sigmoid:** overflow warnings or NaN in `model.activate` and in the language.
   Pinned by `test_sigmoid_extremes` in Task 2 and `test_sigmoid_tanh_eval` in Task 3.
4. **An old `weights.npz` without `acts`:** it must load as all-ReLU, and an all-ReLU net must save
   without the key. Pinned by `test_npz_relu_has_no_acts_key` and `test_npz_roundtrip_acts` in Task 2.
5. **A net rebuilt from another net's layers loses `acts`:** the LP repair, fine-tune
   (`to_torch` / `from_torch`). Pinned by `test_lp_repair_keeps_acts` and
   `test_torch_roundtrip_acts` (Task 8, Task 2).

---

### Task 1: Equivalence script and the "before" baseline

**Files:**
- Create: `experiments/equivalence.py`

**Interfaces:**
- Produces: `uv run python -m experiments.equivalence <command> --copy DIR`. The commands:
  - **`copy`:** copies `runs/` and `runs_bugs/` into `DIR` (deleting `DIR` first).
  - **`compare`:** for every `results.json`, `ast.py`, `ast_trace.json`, `repair/results.json`,
    `repair/*_net.npz` and `dump.py` under `DIR`, compares it to the repo's file at the same relative
    path. Text files are compared with `\r` stripped; `.npz` files by array equality per key. It prints
    each differing path and exits 1 if any differ. With `--skip-glob PATTERN` (repeatable), matching
    files are left out.
  - **`redump`:** for every run under `runs/` and `runs_bugs/`, compares
    `dump_program(*load_run(rd))` with `dump.py` (CRLF stripped). It also compares `save_npz` of the
    loaded net, written to a temp file, byte for byte with `weights.npz`. It prints differences and
    exits 1 if any.
  - **`evaluate`:** runs `nndc.run.evaluate_run` on every run of `all_runs(DIR/runs)` and
    `all_runs(DIR/runs_bugs)`.
  - **`ast`:** runs `nndc.ast_tool.run_ast` on the non-held-out runs in `DIR/runs`:
    - formula runs where `is_step1(rd)` and `rd.name != "seed2"`;
    - `iris` and `iris_petal` seeds 0–2.
  - **`repair`:** runs `run_repair_baselines` and then `evaluate_repair` on every run under
    `DIR/runs` and `DIR/runs_bugs` that has a `repair/` folder in the repo.
- [ ] **Step 1:** Write `experiments/equivalence.py` with the commands above. Use `argparse`. Keep the
  same one-bad-run-does-not-stop-the-batch handling as `nndc/cli.py`.
- [ ] **Step 2:** Run the full baseline on the current code (background; the repair step trains):

```bash
uv run python -m experiments.equivalence copy --copy %TEMP%/nndc-eq/a
uv run python -m experiments.equivalence evaluate --copy %TEMP%/nndc-eq/a
uv run python -m experiments.equivalence ast --copy %TEMP%/nndc-eq/a
uv run python -m experiments.equivalence repair --copy %TEMP%/nndc-eq/a
uv run python -m experiments.equivalence compare --copy %TEMP%/nndc-eq/a
uv run python -m experiments.equivalence redump
```

  Expected: `redump` exits 0. `compare` exits 0, or lists only `repair/` files of fine-tune arms.
- [ ] **Step 3:** Repeat the `copy` and `repair` commands into `%TEMP%/nndc-eq/b`, then compare `a` with `b`
  for `repair/` files only. Record in the commit message which repair files differ between two runs
  of the same code: those are the non-reproducible fine-tune outputs. They become `--skip-glob`
  patterns in Task 9.
  - If anything other than the fine-tune arms differs in Step 2, stop and report it: the baseline
    is not clean.
- [ ] **Step 4:** Commit.

```bash
git add experiments/equivalence.py
git commit -m "feat: equivalence script for the activation refactor (baseline: <result>)"
```

### Task 2: Activations in the model

**Files:**
- Modify: `nndc/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces in `nndc/model.py`:
  - `ACTIVATIONS = ("relu", "sigmoid", "tanh")` and `LIPSCHITZ = {"relu": 1.0, "sigmoid": 0.25, "tanh": 1.0}`.
  - `activate(act: str, z: np.ndarray) -> np.ndarray`, which keeps `z`'s dtype:
    - `relu` is exactly `np.maximum(z, z.dtype.type(0))`;
    - `sigmoid` is the stable form `e = exp(-|z|); where(z >= 0, 1/(1+e), e/(1+e))`;
    - `tanh` is `np.tanh`.
  - `MLP.acts: list[str] | None = None`. `__post_init__` sets it to `["relu"] * (len(weights) - 1)`
    when it is `None`, and raises `ValueError` on a wrong length or an unknown name.
  - `MLP.hidden(X) -> np.ndarray`: the float64 output of the last hidden layer, or `X` as float64
    when there is no hidden layer.
  - `from_torch` accepts `nn.ReLU`, `nn.Sigmoid` and `nn.Tanh`; `to_torch` emits them per `acts`.
  - `save_npz` writes the key `acts` (a str array) only when some act is not `relu`. `load_npz`
    reads it when present, and otherwise uses all-ReLU.
- [ ] **Step 1:** Write the failing tests:
  - `test_rejects_unsupported` now uses `nn.GELU()` with `match="GELU"`; the must-end-with-Linear
    case is unchanged.
  - `test_forward_matches_torch_acts`: parametrise over `[nn.Sigmoid, nn.Tanh]`, and also a mixed
    net `Linear-ReLU-Linear-Sigmoid-Linear`. Check `mlp.acts` and assert the forward pass matches
    torch at `rtol=1e-6, atol=1e-6`.
  - `test_torch_roundtrip_acts`: `from_torch(to_torch(m)).acts == m.acts`, and the weights are
    equal.
  - `test_npz_roundtrip_acts`: a sigmoid/tanh net survives save/load, `acts` included.
  - `test_npz_relu_has_no_acts_key`: for an all-ReLU net, `"acts" not in np.load(path).files`.
  - `test_sigmoid_extremes`: `activate("sigmoid", np.array([-1e4, 0, 1e4], np.float32))` equals
    `[0, 0.5, 1]` under `np.errstate(all="raise")`, and keeps dtype float32.
  - `test_mlp_rejects_bad_acts`: a wrong length and the name `"gelu"` both raise `ValueError`.
- [ ] **Step 2:** Run `uv run pytest tests/test_model.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement the interfaces above. `preactivations` applies `activate(self.acts[i], z)`
  on hidden layers.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit with the message `feat: per-layer activations in MLP (relu default; sigmoid, tanh)`.

### Task 3: `sigmoid` and `tanh` in the program language

**Files:**
- Modify: `nndc/lang.py` (`FUNCTIONS`, the description string near line 35, the arity check near
  line 111, and the evaluator near line 279)
- Test: `tests/test_lang.py`

**Interfaces:**
- Produces: the language functions `sigmoid(x)` and `tanh(x)`, one argument each. They are evaluated
  in float64 with the same formulas as `model.activate`. They count as nodes; no new size key is added.
- [ ] **Step 1:** Write the failing tests:
  - `test_sigmoid_tanh_eval`: `def net(x0):\n    return sigmoid(x0) + tanh(x0)` on
    `[-1e4, -1, 0, 1, 1e4]` equals `1/(1+exp(-x)) + tanh(x)`, with no warning under
    `np.errstate(all="raise")`.
  - `test_sigmoid_one_arg`: `sigmoid(x0, x1)` raises `ProgramError` matching "exactly one argument".
- [ ] **Step 2:** Run them. Expected: FAIL ("function sigmoid is not allowed").
- [ ] **Step 3:** Implement. Add `sigmoid` and `tanh` to the arity check together with `relu` and
  `abs`, and to the description string.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit with the message `feat: sigmoid and tanh in the program language`.

### Task 4: Dump and checker

**Files:**
- Modify: `nndc/dump.py` (the hidden-unit line), `nndc/check.py` (`_rounding_bound`)
- Test: `tests/test_dump.py`, `tests/test_check.py`

**Interfaces:**
- Consumes: `MLP.acts`, `activate` and `LIPSCHITZ` (Task 2); `sigmoid` and `tanh` in the language
  (Task 3).
- Produces:
  - **The dump** writes `{act}({affine})` per hidden unit. The ReLU text is unchanged and the header
    is unchanged.
  - **`_rounding_bound`:** for a `relu` layer, the code path is exactly today's. For `sigmoid` and
    `tanh` layers, after the affine error update, `e = LIPSCHITZ[act] * e + 4 * eps`, and
    `h = activate(act, z)` in float64.
- [ ] **Step 1:** Write the failing tests:
  - In `test_dump.py`: parametrise `test_dump_evaluates_like_net` over
    `acts in [None, ["sigmoid"], ["tanh"]]` for the one-hidden-layer cases, and
    `["relu", "sigmoid"]` for `[3, 8, 8, 1]`. Use the same tolerances.
  - `test_dump_relu_text_unchanged`: the dump of `hand("absdiff")` equals the committed
    `runs/absdiff/hand/dump.py`, CRLF stripped.
  - In `test_check.py`:
    - `test_sigmoid_dump_passes_exact`: dump a random `[2, 4, 1]` sigmoid net (regression) and a
      `[2, 4, 3]` tanh net (argmax), check each dump against its own net, and assert `exact_pass`.
    - `test_rounding_bound_relu_unchanged`: compare `_rounding_bound` on a random `[3, 8, 8, 2]`
      ReLU net with a copy of today's function pasted into the test as `_old_bound`. Assert
      `np.array_equal`.
- [ ] **Step 2:** Run them. Expected: the sigmoid/tanh ones FAIL; the two "unchanged" ones PASS
  (they guard the refactor).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit with the message `feat: dump and checker handle sigmoid/tanh layers`.

### Task 5: Symbolic net with explicit activations

**Files:**
- Modify: `nndc/symnet.py`
- Test: `tests/test_symnet.py`, `tests/test_heuristics_structure.py` (lines 16 and 32: `u.relu` becomes
  `u.act == "relu"`)

**Interfaces:**
- Produces in `nndc/symnet.py`:
  - `Unit(name, layer, pre, act: str = "relu")` with `act` one of `relu`, `linear`, `sigmoid`,
    `tanh`. The `relu` attribute no longer exists.
  - `post_interval(act: str, lo: float, hi: float) -> tuple[float, float]`:
    - `relu` gives `(max(lo, 0.0), max(hi, 0.0))`;
    - `linear` gives `(lo, hi)`;
    - `sigmoid` and `tanh` give `(f(lo), f(hi))`, using `math`, with sigmoid stable for large
      `|z|`.
  - `from_dump` reads `relu(...)`, `sigmoid(...)` and `tanh(...)` around an affine expression;
    other assignments are `linear`. The argmax outputs must be `linear` units.
  - `evaluate` uses `activate` for relu/sigmoid/tanh (float64) and the identity for linear.
  - `unit_bounds` uses `post_interval`. `substitute` keeps `u.act`.
  - `to_program` writes `{act}(...)` for non-linear units.
- [ ] **Step 1:** Write the failing tests:
  - `test_roundtrip_sigmoid_tanh`: dump a random `[2, 4, 4, 1]` net with `acts=["sigmoid", "tanh"]`.
    `from_dump` gives units with those acts, and `evaluate` matches `mlp.forward` at `rtol=1e-5`.
    `to_program`, run through `lang.evaluate`, matches as well.
  - `test_post_interval`: the four activations on `(-2.0, 3.0)`, plus sigmoid on `(-1e4, 1e4)` gives
    `(0.0, 1.0)` without an exception.
  - Update the two lines in `test_heuristics_structure.py`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Then grep the whole repo for `\.relu\b` and `relu=`; the only hits left
  must be in `compile_back.py` (its own `_Builder.relu`).
- [ ] **Step 4:** Run `uv run pytest tests/test_symnet.py tests/test_heuristics_structure.py -v`.
  Expected: PASS. The full suite still fails in the heuristics, which Task 6 fixes; do not commit
  yet.

### Task 6: Heuristics and the AST tool

**Files:**
- Modify:
  - `nndc/heuristics/bounds.py` (`prune_dead_and_linear`)
  - `nndc/heuristics/merge.py` (`normalize`, `merge_units`)
  - `nndc/heuristics/snap.py` (`snap_net`)
  - `nndc/heuristics/motifs.py` (`as_relu_sum`)
  - `nndc/heuristics/decision_list.py` (`fold`, `_patterns`, `decision_list`)
  - `nndc/ast_tool.py` (`scan_directions`)
- Test: `tests/test_heuristics_sigmoid.py` (new)

**Interfaces:**
- Consumes: `Unit.act` and `post_interval` (Task 5).
- Produces these rules:

| Site | New rule |
|---|---|
| `prune_dead_and_linear` | relu: dead/always-on as today. linear: inlined as today. sigmoid/tanh: left alone. |
| `normalize`, `merge_units` (unit list) | Only `act == "relu"` units. Post bounds via `post_interval`. |
| `snap_net` | Post bounds via `post_interval`. |
| `as_relu_sum` | `None` if any unit's act is sigmoid or tanh. The old `not u.relu` branch becomes `u.act == "linear"`. |
| `decision_list` | Returns `None` if any unit's act is sigmoid or tanh. `fold` and `_patterns` test `act == "relu"` / `act == "linear"` where they tested `u.relu` / `not u.relu`. |
| `scan_directions` | `u.act == "relu"`. |

- [ ] **Step 1:** Write the failing tests in `tests/test_heuristics_sigmoid.py`:
  - `test_prune_keeps_sigmoid_units`: a symnet with one sigmoid unit whose pre-activation is always
    negative on the box. After `prune_dead_and_linear` the unit is still there, and the net's
    evaluation is unchanged.
  - `test_relu_only_routes_skip_sigmoid`: for a dumped sigmoid net, `as_relu_sum(...)` is `None` and
    `decision_list(...)` (argmax, `[2, 4, 3]`) is `None`.
  - `test_refactor_sigmoid_net_passes`: `refactor(dump, blind_spec, mlp)` on a random `[2, 4, 1]`
    sigmoid regression net and a `[2, 4, 3]` tanh argmax net. The winning program passes the
    checker (snap at least), and the trace lists "decision-list: not applicable" for the argmax net.
- [ ] **Step 2:** Run them. Expected: FAIL (`AttributeError: relu`, or a wrong program).
- [ ] **Step 3:** Implement the table above.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit Tasks 5 and 6 together:

```bash
git add nndc/symnet.py nndc/heuristics nndc/ast_tool.py tests/test_symnet.py tests/test_heuristics_structure.py tests/test_heuristics_sigmoid.py
git commit -m "refactor: explicit unit activations in the symbolic net; ReLU-only heuristics skip sigmoid/tanh"
```

### Task 7: Training with an activation

**Files:**
- Modify: `experiments/train.py` (`train`)
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces: `train(task, seed, target=None, act: str = "relu") -> tuple[MLP, dict]`. Hidden layers use
  the matching torch module. The log gets `"act"` only when `act != "relu"`, so ReLU logs are
  unchanged.
- [ ] **Step 1:** Write the failing test `test_train_sigmoid_window`. Train `get_task("window")` with
  `act="sigmoid"` for the task's steps, seed 0. Assert `mlp.acts == ["sigmoid"]`, that `log["act"]`
  is `"sigmoid"`, and accuracy ≥ 0.98. The spike measured 0.999 in a few seconds.
- [ ] **Step 2:** Run it. Expected: FAIL (unexpected keyword `act`).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit with the message `feat: train() takes a hidden activation`.

### Task 8: ReLU-only pieces refuse; the LP repair keeps activations

**Files:**
- Modify:
  - `nndc/compile_back.py` (`compile_program`)
  - `nndc/baselines/regions.py` (`regions_program`)
  - `nndc/repair/weights.py` (`_hidden`, `lp_repair`)
- Test: `tests/test_compile_back.py`, `tests/test_baselines_regions.py`, `tests/test_repair_weights.py`

**Interfaces:**
- Produces:
  - **`compile_program`:** raises `ValueError("compile_back: sigmoid/tanh cannot be compiled into a ReLU net")`
    when the program calls `sigmoid` or `tanh`.
  - **`regions_program`:** raises `ValueError` naming the net's `acts` when any act is not `relu`.
  - **`_hidden(mlp, X)`:** returns `mlp.hidden(X)`.
  - **`lp_repair`:** builds `MLP(..., acts=mlp.acts)`.
- [ ] **Step 1:** Write the failing tests:
  - `test_compile_back_refuses_sigmoid`
  - `test_regions_refuses_sigmoid`
  - `test_lp_repair_keeps_acts`: a small sigmoid regression net where the LP is feasible, for
    example repair cases that equal the net's own outputs plus 0. Assert `fixed.acts == mlp.acts`.
  - `test_hidden_matches_old_relu`: `mlp.hidden(X)` equals
    `np.maximum(mlp.preactivations(X)[-2].astype(np.float64), 0.0)` exactly for a ReLU net.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit with the message `feat: ReLU-only pieces refuse sigmoid/tanh; LP repair keeps activations`.

### Task 9: Equivalence after the refactor, smoke test, docs

**Files:**
- Test: `tests/test_activations_smoke.py` (new)
- Modify: `CLAUDE.md` (Architecture: one sentence on `MLP.acts` / `Unit.act`, and that compile_back,
  regions and the ReLU motif/decision-list routes are ReLU-only)

**Interfaces:**
- Consumes: everything above; `experiments/equivalence.py` (Task 1).
- [ ] **Step 1:** Write `test_activations_smoke.py::test_sigmoid_and_tanh_end_to_end`. For `act` in
  `["sigmoid", "tanh"]`:
  - train `window`, seed 0, with `train(..., act=act)`;
  - save and load it through `save_npz` / `load_npz` in `tmp_path`;
  - dump it, and assert `check_program(dump)` gives `exact_pass`;
  - run `refactor` with the blind spec, and assert that the winner passes the checker (snap at
    least).
- [ ] **Step 2:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 3:** Run the equivalence check on the new code (background):

```bash
uv run python -m experiments.equivalence copy --copy %TEMP%/nndc-eq/c
uv run python -m experiments.equivalence evaluate --copy %TEMP%/nndc-eq/c
uv run python -m experiments.equivalence ast --copy %TEMP%/nndc-eq/c
uv run python -m experiments.equivalence repair --copy %TEMP%/nndc-eq/c
uv run python -m experiments.equivalence compare --copy %TEMP%/nndc-eq/c --skip-glob <patterns from Task 1>
uv run python -m experiments.equivalence redump
```

  Expected: both `compare` and `redump` exit 0. Any difference is a bug in the refactor; fix it, and
  do not accept the difference.
- [ ] **Step 4:** Re-evaluate in place and confirm that nothing in the repo changed:

```bash
uv run python -c "from pathlib import Path; from nndc.run import all_runs, evaluate_run; [evaluate_run(rd) for root in ('runs', 'runs_bugs') for rd in all_runs(Path(root))]"
rtk proxy git status --short runs runs_bugs
```

  Expected: no output from `git status`.
- [ ] **Step 5:** Update `CLAUDE.md`, then commit with the message
  `test: sigmoid/tanh smoke test; docs: activation types in CLAUDE.md`.
