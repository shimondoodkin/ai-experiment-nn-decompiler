# Sigmoid Nets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run steps 1–3 on 30 sigmoid nets in `runs_sigmoid/`: build, blind agents, AST tool with
sigmoid moves frozen before the held-out run. Show the intended formula next to each net, and write
`runs_sigmoid/REPORT.md` with "Formula vs net" and "Sigmoid vs ReLU" sections.

**Architecture:** The code changes come first:
- `build_run` / `cli` gain `--act`, and sigmoid runs get a `formula.py` reference version;
- the agent CLI gets `--index` / `--heuristics`;
- the AST tool gains a sigmoid module (`nndc/heuristics/sigmoid.py`), with greedy unit replacement
  and a linear-scores route, wired into `refactor` only for nets with non-ReLU units;
- `report.py` gains the two new sections.

Data work (training, agent passes, the tool runs) goes in separate `data:` commits, in the order the
design fixes.

**Tech Stack:** Python 3.13, NumPy, torch (CPU), scikit-learn (already in the default `baselines`
group), pytest, `uv`.

**Spec:** `docs/design/2026-10-28-sigmoid-nets-design.md`

## Global Constraints

- **Nothing under `runs/` or `runs_bugs/` is rewritten.** Older runs stay. Destructive commands run
  only on scratch copies in the session scratchpad, never on the repo, its parent or `runs*/`.
  Tests that delete use `tmp_path` or patch the delete.
- **ReLU results unchanged:** `experiments/equivalence.py` (`copy`, `evaluate`, `ast`, `compare`,
  `redump`) must report 0 differences after the AST tool changes.
- **Training:** same sizes, steps, learning rates and seeds as ReLU. No per-task tuning. Nets that
  don't learn stay.
- **Held-out sets:** formula seed 2 and Iris seeds 3–5. Tool development uses formula seeds 0–1,
  Iris 0–2, and `runs_sigmoid/HEURISTICS.md` only, never `HEURISTICS_heldout.md`.
- **Blindness:** agents get the standard instructions. Nothing about sigmoid, rounding or snapping.
- **Commits:** code commits and `data:` commits are separate. No attribution lines.
- **Blind-folder modules** (`model.py`, `lang.py`, `check.py`, `spec.py`) stay stdlib + NumPy.

## Review Focus

1. **Formula counterexamples leak into `counterexamples.json`.** They would harden every other
   check. Pinned by `test_formula_cex_not_shared` (Task 1).
2. **A ReLU run gets `formula.py`, or `--act sigmoid` builds a hand net.** Pinned by
   `test_formula_only_for_sigmoid` and `test_sigmoid_skips_hand` (Task 1).
3. **A replaced unit emits a program the language can't parse, or one that evaluates differently from
   the symbolic net.** Pinned by `test_step_clamp_program_matches_symnet` (Task 4).
4. **New AST routes run on a ReLU net.** That would break equivalence. Pinned by
   `test_sigmoid_routes_skip_relu_nets` (Task 4), plus the Task 4 equivalence run.
5. **A held-out agent's heuristics land in the development file.** Pinned operationally in Task 3,
   where every held-out collect names `HEURISTICS_heldout.md`, and by `test_collect_heuristics_path`
   (Task 2).

---

### Task 1: Build sigmoid runs, with `formula.py` as a reference version

**Files:**
- Modify: `nndc/run.py` (`build_run`, `evaluate_run`), `nndc/cli.py` (`run` subcommand)
- Test: `tests/test_run.py`

**Interfaces:**
- Produces:
  - `build_run(task, seed, root=RUNS, retrain=False, bug=False, act="relu") -> Path`.
    - With `act != "relu"`: it raises `ValueError` for a hand net (`seed is None`) and for
      `bug=True`, and calls `train(t, seed, act=act)`.
    - For a task in `TASKS` (formula tasks), it writes `formula.py = expected_program(t)`.
  - In `evaluate_run`:
    - `"formula"` is added to the versions when `formula.py` exists.
    - Its result is computed like the others, but its counterexamples are not added to
      `counterexamples.json`. They stay in its own `results.json` entry.
  - CLI: `nndc run <task|all> --act sigmoid --root runs_sigmoid`. With `--act` other than relu,
    `all` and default seeds skip hand nets.
- [ ] **Step 1:** Write failing tests in `tests/test_run.py`. They use `tmp_path` as the root, and
  `xor` and `window` (fast to train):
  - `test_formula_only_for_sigmoid`:
    - `build_run("window", 0, tmp_path / "s", act="sigmoid")` writes `formula.py`, equal to
      `expected_program(get_task("window"))`;
    - `results.json` has a `"formula"` version;
    - `load_run(rd)[1].acts == ["sigmoid"]`;
    - a ReLU `build_run("window", 0, tmp_path / "r")` writes no `formula.py`.
  - `test_formula_cex_not_shared`: on a sigmoid `absdiff` seed 0 run (its formula fails near the kink):
    1. delete `counterexamples.json`;
    2. run `evaluate_run`;
    3. let F be the inputs in `results["versions"]["formula"]["check"]["counterexamples"]`, and O the
       inputs of every other version's counterexamples;
    4. assert that F is not empty, and that no input in F − O appears in `counterexamples.json`.
  - `test_sigmoid_skips_hand`: `build_run("xor", None, tmp_path, act="sigmoid")` raises `ValueError`,
    and so does `build_run("xor", 0, tmp_path, bug=True, act="sigmoid")`.
- [ ] **Step 2:** Run `uv run pytest tests/test_run.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement. Keep the ReLU code path byte-for-byte the same in behaviour: the
  `train` call for relu stays `train(t, seed)`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit with the message `feat: build sigmoid runs (--act) with the intended formula as a reference version`.

### Task 2: Blind index and heuristics paths for another root

**Files:**
- Modify: `nndc/cli.py` (`agent prepare`, `agent collect`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces:
  - `nndc agent prepare ... --index PATH`, defaulting to `runs/blind_index.json`.
  - `nndc agent collect <blind> --index PATH --heuristics PATH`, defaulting to today's paths.
    `cmd_agent_collect` looks the kind up in `--index`, not in `prep.INDEX`.
  - `prepare()` and `collect()` already take `index` / `heuristics`. The change is CLI plumbing only.
- [ ] **Step 1:** Write failing test `test_collect_heuristics_path`.
  - Use `nndc.cli.main([...])` or the cmd functions with `tmp_path` paths, following the existing
    mock-agent tests in `tests/test_agent.py`.
  - Prepare one run with `--index tmp/idx.json`, mock-write an attempt, and collect with
    `--index tmp/idx.json --heuristics tmp/H.md`.
  - Assert that `tmp/H.md` holds the section, and that `runs/blind_index.json` is unchanged:
    compare its bytes before and after.
- [ ] **Step 2:** Run it. Expected: FAIL (unrecognised argument `--index`).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit with the message `feat: agent prepare/collect take --index and --heuristics`.

### Task 3: Build the 30 nets and run the blind agents (data)

No code; this task is operations.
- [ ] **Step 1:** Run `uv run nndc run all --act sigmoid --root runs_sigmoid` in the background.
  - Expected: 30 lines, each `dump PASS exact`.
  - Record which nets did not learn (`learned: false` in `train_log.json`).
  - Commit: `git add runs_sigmoid`, message `data: 30 sigmoid nets (same tasks, seeds and settings as ReLU)`.
- [ ] **Step 2:** Run `uv run nndc agent prepare --all --root runs_sigmoid --index runs_sigmoid/blind_index.json`.
  Keep the printed folder→run map.
- [ ] **Step 3:** Dispatch one fresh general-purpose subagent per folder, at most 20 at once, so two
  waves. Each prompt is only: "Work only inside `<folder>`. Read INSTRUCTIONS.md there and follow it."
  Keep a folder→agent-id map.
- [ ] **Step 4:** Collect each finished folder. Check first that its `agent.py` no longer changes.
  - Development runs (formula seeds 0–1, Iris 0–2):
    `uv run nndc agent collect <folder> --index runs_sigmoid/blind_index.json --heuristics runs_sigmoid/HEURISTICS.md --transcript <agent jsonl>`
  - Held-out runs (formula seed 2, Iris 3–5): the same command, with
    `--heuristics runs_sigmoid/HEURISTICS_heldout.md`.
  - **Do not open `HEURISTICS_heldout.md` afterwards** until the freeze in Task 5.
  - Review the audit flags by hand, as in step 2.
- [ ] **Step 5:** Re-evaluate every sigmoid run (`evaluate_run` over `all_runs(Path("runs_sigmoid"))`).
  Commit `runs_sigmoid` with the message `data: blind agent passes on 30 sigmoid nets`.

### Task 4: AST tool moves for sigmoid units (development set only)

**Files:**
- Create: `nndc/heuristics/sigmoid.py`
- Modify:
  - `nndc/symnet.py` (new unit acts `step` and `clamp`)
  - `nndc/ast_tool.py` (`refactor`)
  - this plan (Step 0)
- Test: `tests/test_heuristics_sigmoid.py`

**Interfaces:**
- **Symbolic net, new unit acts:**
  - `step`: evaluates `1.0 if z > 0 else 0.0`, and `to_program` writes `(<pre> > 0)`.
  - `clamp`: evaluates `min(max(0.5 + 0.25 z, 0), 1)`, and `to_program` writes
    `min(max(<0.5 + 0.25 pre>, 0), 1)`, with the inner affine simplified via `Affine`.
  - `post_interval` is monotone for both.
  - `UNIT_ACTS` gains both. The fail-closed guards from project A leave them out of the ReLU routes.
- **`nndc/heuristics/sigmoid.py`:**
  - `soft_share(net: SymNet, X: np.ndarray) -> dict[str, float]`: per sigmoid unit, the share of
    rows where its output is in (0.05, 0.95).
  - `replace_units(net: SymNet, spec: Spec, mlp: MLP) -> tuple[SymNet, list[str]]`:
    - It goes through sigmoid units in ascending `soft_share`, so the most saturated come first.
    - For each unit it tries, in order:
      - step;
      - line: the unit becomes `linear`, with `pre := 0.5 + 0.25 * pre`;
      - clamp.
    - It keeps the first option for which `check_program(to_program(candidate), mlp, spec).snap_pass`
      holds, and otherwise leaves the unit as sigmoid.
    - The trace has one line per unit, for example `h1_3: step` or `h1_5: kept sigmoid`.
    - The builder may reuse one input set per net (`build_input_set`) for speed. Acceptance must
      equal the checker's verdict.
  - `linear_scores(spec: Spec, mlp: MLP) -> str | None`, for argmax only:
    - Fit `LogisticRegression(C=1e4, max_iter=5000)` to the net's own decisions on
      `build_input_set(mlp, spec, seed=1)`.
    - Subtract class 0's score from all the scores, and divide by the largest |coefficient|.
    - Snap the coefficients with `nice(v, tol)` for `tol` in `(0.05, 0.02, 0.01, 0.005)`.
    - Return the first program `argmax(0, s1, …)` that passes `check_program`, otherwise the
      unsnapped one if it passes, otherwise `None`.
- **New routes in `refactor`**, active only when `any(u.act not in ("relu", "linear") for u in net0.units)`:
  - `sigmoid-replace`: `replace_units(n1)`, then `prune_dead_and_linear`, then merge, then `_emit`.
  - `sigmoid-replace-snap`: the same net through `snap_net`, prune and merge.
  - For argmax: `decision_list(<replaced net>, spec, mlp)`. Its guard lets it run only when every
    unit is relu or linear. Its routes are added as `decision-list` / `decision-list-snap`, as today.
  - `linear-scores`: for argmax nets.
- [ ] **Step 0:** Read `runs_sigmoid/HEURISTICS.md` (development set only).
  - Append to this plan a section "Task 4 move list". It lists the moves above plus any move at least
    two development-set agents name, each as an `Interfaces` line like the ones above.
  - Commit this plan with the message `docs: sigmoid AST move list (fixed before tool code)`.
  - Implement only that list.
- [ ] **Step 1:** Write the failing tests:
  - `test_step_clamp_program_matches_symnet`: a hand-built symnet with one `step`, one `clamp` and
    one linear unit. `to_program` parses under `lang`, and `lang.evaluate` equals `symnet.evaluate`
    on 1,000 random points.
  - `test_replace_units_on_window`: train `window` with `act="sigmoid"`, seed 0, using `train()`.
    `replace_units` turns all 4 units into `step`, and the emitted program passes `check_program`.
    That matches the spike's result.
  - `test_linear_scores_iris_petal`: a sigmoid `iris_petal` net, seed 0, trained in the test. Its
    `linear_scores` returns a program that passes `check_program`. That matches the spike.
  - `test_sigmoid_routes_skip_relu_nets`: `refactor` on a ReLU hand net (`hand("absdiff")`) has no
    candidate whose route starts with `sigmoid-` or equals `linear-scores`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the move list.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Check that ReLU is unchanged (in the background), on a scratchpad copy:

```bash
uv run python -m experiments.equivalence copy --copy <scratchpad>/nndc-eq/d
uv run python -m experiments.equivalence evaluate --copy <scratchpad>/nndc-eq/d
uv run python -m experiments.equivalence ast --copy <scratchpad>/nndc-eq/d
uv run python -m experiments.equivalence compare --copy <scratchpad>/nndc-eq/d
uv run python -m experiments.equivalence redump
```

  Expected: both exit 0 with 0 differences.
- [ ] **Step 6:** Run the tool on the development set, in the background:
  `uv run nndc ast runs_sigmoid/{xor,window,neareq,absdiff,median3,tax}/seed{0,1} runs_sigmoid/{iris,iris_petal}/seed{0,1,2}`.
  - Bugs in the listed moves may be fixed and re-run, but no move may be added.
  - Then commit the code (message `feat: AST tool moves for sigmoid units (freeze)`). This is the
    freeze.
  - Then commit the development-set `runs_sigmoid` data separately, with the message
    `data: AST tool on sigmoid development nets`.

### Task 5: Held-out AST run (data)

- [ ] **Step 1:** Check that `git status` shows no uncommitted change in `nndc/` or `experiments/`, so
  the freeze commit is what runs.
- [ ] **Step 2:** Run `uv run nndc ast runs_sigmoid/{xor,window,neareq,absdiff,median3,tax}/seed2 runs_sigmoid/{iris,iris_petal}/seed{3,4,5}`.
  Expected: each `run_git.txt` gets an `ast:` line with the freeze commit and no `+dirty`.
- [ ] **Step 3:** Commit with the message `data: frozen sigmoid AST tool on held-out nets`. No re-runs
  and no tool changes after this point.

### Task 6: Report sections and `runs_sigmoid/REPORT.md`

**Files:**
- Modify: `nndc/report.py` (`build_report`, plus new `formula_md` and `activation_compare_md`),
  `nndc/cli.py` (`report --compare-to`), `README.md` (one link)
- Test: `tests/test_report.py`

**Interfaces:**
- **`build_report(root, compare_to: Path | None = None) -> str`.**
  - When `compare_to` is given, the step-4 comparison section is left out. `formula_md(root)` and
    `activation_compare_md(compare_to, root)` are added in its place.
  - The default call (`compare_to=None`) produces exactly today's `REPORT.md`.
- **`formula_md(root) -> str`:** one row per run that has a `formula` version. The columns:
  - the run;
  - the formula's check verdict against the net;
  - its max error, as a % of the net's output range (regression), or disagreements outside / inside
    the zone (binary);
  - its task grade (in / out);
  - whether the best agent program and the AST program pass against the net, and whether each
    recovered the formula.

  Below the table, per run: up to 5 counterexamples of the formula, each with inputs, net, formula and
  true answer. The true answer is computed with `get_task(task).target`.
- **`activation_compare_md(relu_root, root) -> str`:** one row per task and seed present in both
  roots. The columns:
  - the net learned (relu / sigmoid);
  - the best agent program, as verdict, recovered and nodes, for each activation;
  - the AST program, as verdict, recovered and nodes, for each activation.
- **CLI:** `nndc report --root runs_sigmoid --out runs_sigmoid/REPORT.md --compare-to runs`.
- [ ] **Step 1:** Write the failing tests in `tests/test_report.py`. Use small fake `results.json` trees
  under `tmp_path`, following the existing report tests:
  - `test_formula_md_rows`: one row per formula run, with its counterexamples listed;
  - `test_activation_compare_md_pairs_runs`: a run present in both roots gives one row; a run in only
    one root gives none;
  - `test_default_report_unchanged`: `build_report(RUNS)` equals the committed `REPORT.md`, CRLF
    stripped.

    If `REPORT.md` cannot be regenerated byte-identically today, for example because of timing, the
    test compares the section headings only. Note that in the commit message.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Generate the report:
  `uv run nndc report --root runs_sigmoid --out runs_sigmoid/REPORT.md --compare-to runs`.
  Add one README line linking `runs_sigmoid/REPORT.md`.
  - Commit the code: `feat: formula-vs-net and sigmoid-vs-ReLU report sections`.
  - Commit the report: `data: runs_sigmoid/REPORT.md`.
- [ ] **Step 6:** Final check:
  - run the whole-branch review (one fresh reviewer, the most capable model, read-only);
  - send the report to the user;
  - the `CONCLUSIONS.md` section is written with the user after they have read it, outside this plan.

## Task 4 move list (fixed before tool code, from `runs_sigmoid/HEURISTICS.md`, development set only)

Each move ships with the tests named here. A move counts only when the program it produces passes `check_program`.
Moves 1–3 come from the plan. Moves 4–7 are each named by at least two development-set agents (listed in brackets).

1. **step**, 2. **line**, 3. **clamped line**: per-unit replacements in `replace_units`, as specified in Task 4.
4. **const** [neareq/seed0, neareq/seed1, absdiff/seed0, median3/seed1]: a sigmoid unit whose output range over the input
   set is under 0.02 becomes a constant. Its value is the mean output, written as a `linear` unit with an empty
   coefficient map, and pruning inlines it.
   - `replace_units` tries the options in the order const, step, line, clamp.
   - Test: `test_const_unit_inlined`.
5. **step duplicates** [window/seed0, window/seed1, xor/seed0, xor/seed1]: `step(z)` does not change when `z` is scaled
   by a positive number. So after replacement, each step unit's pre-activation is scaled so that its largest |coef|
   is 1, and steps with identical pre-activations (to 1e-9) are merged by adding up their consumers' coefficients.
   - This lives in `nndc/heuristics/sigmoid.py` (`merge_steps(net) -> tuple[SymNet, list[str]]`); the ReLU merge is
     untouched.
   - Test: `test_merge_steps_on_window`: the window net goes from 4 steps to at most 3, and the program still passes.
6. **linear cascade** for argmax nets [iris_petal/seed0–2, iris/seed0–2]:
   `linear_cascade(spec, mlp) -> str | None` in `sigmoid.py`.
   - On `build_input_set(mlp, spec, seed=1)`, keep the confident points (outside the unsure zone), and count which
     class pairs are the top two.
   - If the classes form a chain a | b | c (only the pairs a-b and b-c are adjacent), fit `LogisticRegression(C=1e4)`
     for "a vs rest" and "c vs rest" on the confident points.
   - Normalise each cut by its largest |coef|, then snap the coefficients and the intercept with the same `nice`
     ladder as `linear_scores`.
   - Emit `if s_a > 0: return a`, then `if s_c > 0: return c`, then `return b`, using the first snapping level that
     passes, else the unsnapped version if it passes, else `None`.
   - Route name `linear-cascade`; it runs under the same condition as `linear-scores`.
   - Test: `test_linear_cascade_iris_petal`.
7. **knot fit for 1-input regression** [tax/seed0, tax/seed1, absdiff/seed0]: `knot_fit(spec, mlp) -> str | None` in
   `sigmoid.py`, for regression nets with one input only.
   - For k = 1, 2, 3 knots placed on the grid `lo + (hi - lo) * j / 20`, choose knots greedily: each new knot is the
     grid point that most reduces the max error.
   - The fit itself is a minimax LP (`scipy.optimize.linprog`) of `c0 + c1*x + Σ ck*relu(x - knot_k)` against the net,
     on 2,001 grid points.
   - Snap the slopes with `nice(v, 0.05 * max|slope|)`, refit the constant, and emit the result. The first k that
     passes the checker wins.
   - Route name `knot-fit`; it runs only for nets with a unit that is not relu/linear.
   - Test: `test_knot_fit_tax_seed0`: the sigmoid tax net from `runs_sigmoid/tax/seed0` (a development net) gives a
     program that passes.

Not included, because only one agent named them or they are already covered by an existing route:
- boundary scanning for binary nets (the existing `boundary-scan` route already reads the net as a black box);
- rotating to difference/sum coordinates (absdiff only);
- grouping units by direction for multi-input regression (median3);
- exact bisected thresholds (window/seed1 only).
