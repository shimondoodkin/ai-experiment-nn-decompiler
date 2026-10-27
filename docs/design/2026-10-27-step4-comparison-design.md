# Design: step 4, comparison with existing methods

Date: 2026-10-27
Status: draft for approval
Parent design: `docs/design/2026-10-27-nn-decompiler-design.md` (§5 step 4)
Literature: `reports/Neural net decompilation prior work.md` (notes in `research_notes/`)

## 1. Intent

The literature search found that every component of this repo has a precedent:
- MIPS already turns small ReLU nets into Python with integer snapping.
- Exact ReLU-to-tree conversion exists.
- Symbolic regression finds formulas.
- LLMs describe functions from input/output behaviour.

The combination was not found anywhere: a task-blind coding agent reading the net's weights as code,
non-integer "intended" snapping, and a distilled deterministic tool.

Step 4 measures the repo's two methods against runnable representatives of each existing
approach, on the same 29 models, with the same checker and the same grading. Each baseline
answers a specific question:

| Baseline | Question it answers |
|---|---|
| **B1: PySR on samples of the net** (raw) | Does an established formula search find the same programs from behaviour alone? |
| **B2: PySR plus this repo's snapping** | Is snapping the missing ingredient, or the search? (The report's suggested split: "separates search from snapping".) |
| **B3: decision-tree distillation** (sklearn) | How does the classic approximate rule-extraction baseline (TREPAN-style) do on size and fidelity? |
| **B4: exact activation-region decomposition** | How big is the exact, faithful but unreadable program, i.e. the Aytekin/EC-DT/RENTT family? |
| **B5: Claude Code from input/output only** (no dump, no weights) | Is reading the weights as code what makes the agent succeed? The report calls this the most important experiment. |

The straight-line compiled-C speed baseline belongs to step 5.

## 2. Goals, non-goals, anti-goals

### Goals
- Every baseline produces a program in the repo's language, so the existing checker,
  task grading, size metric, report and (in step 5) the C backend apply unchanged.
- One comparison table: method × {passes check, exact, recovered the true formula inside and
  outside the training range, median size, cost}, with held-out (seed 2) and in-sample reported
  separately.
- A per-model matrix showing which method recovered which algorithm.
- The README's "Prior work" section is rewritten with the positioning sentence from the report and
  a link to it.

### Non-goals
- KAN, gplearn, AI Feynman, ruleex/REMIX, parx, MIPS's own code. They are cited, not run: install
  friction on Windows/Python 3.13, or a different model class.
- Tuning any baseline per task.
- Speed benchmarks (step 5).

### Anti-goals
- Giving one method more information than another. All non-agent baselines get the same
  samples of the net (inputs, net outputs, blind spec) and the same checker oracle the AST tool
  uses for selection. They never get the task name or target. Raw PySR (B1) is the variant
  selected without the checker.
- Choosing baseline hyperparameters after looking at held-out results. They are fixed in this
  design, before any run.
- Presenting "not found in our searches" as "does not exist". The report's wording is kept.

## 3. Shared setup

- **Samples:** 50,000 points per model.
  - continuous inputs: uniform over the spec box
  - discrete inputs: uniform integers (the full grid when it is at most 50k points)
  - fixed seed 0
  - the net's raw output: the value for regression, the score `out − threshold` for binary
- **Blind spec only** (neutral names, types, bounds, output kind).
- **Output files** go in each run folder as program versions: `tree.py`, `pysr.py`,
  `pysr_snap.py`, `regions.py`, and `io_agent_<n>.py` for B5. `evaluate_run` checks and grades
  them like `dump.py`, `agent_n.py` and `ast.py`.
- **Side files:** each baseline also writes `<name>_info.json` with runtime, hyperparameters,
  and anything it skipped and why.
- **Commands:** `nndc baseline <name> <run>... | --all`, with name one of `tree | pysr | regions | all`.
  B5 uses the agent commands (§4.5).

## 4. Baselines

### 4.1 B3: decision-tree distillation
- `DecisionTreeRegressor` (regression), or `DecisionTreeClassifier` on the net's class (binary),
  fitted on the 50k samples.
- Sweep `max_leaf_nodes` ∈ {2, 4, 8, …, 512}. Keep the **smallest tree that passes the checker**
  (snap); if none passes, keep the largest. This is the same checker-guided "simplest passing"
  rule the AST tool uses, so the comparison is fair.
- Emitted as nested `if x_i <= t: … else: …` returning the leaf value (regression) or 0/1
  (binary). If no tree in the sweep passes, the largest is reported as a failing program, not as
  "no program".

### 4.2 B1/B2: PySR
- Settings, fixed here:
  - binary operators `+ - * max min greater`
  - unary operators `abs relu`
  - `cond` if the installed version has it
  - `maxsize=25`, `niterations=40`, `timeout_in_seconds=120`, `deterministic=True`,
    `parallelism="serial"`, `random_state=0`
  - 5,000 of the 50k samples (PySR is slow on large datasets)
  - for binary nets PySR fits the score `out − threshold`, and the emitted program returns that
    score (class = score > 0, the same convention as `dump.py`). The tree fits the class instead.
- **B1 (raw):** PySR's own chosen best equation (`model.get_best()`), converted to the program
  language.
- **B2 (snapped):** every equation on PySR's Pareto front, with each numeric constant snapped by
  the repo's `nice()`:
  - multiplicative coefficients: 5% relative
  - additive constants and comparison thresholds: 1% of the relevant output or input range

  Then the smallest one that passes the checker is chosen, exactly like the AST tool's selection.
- **Conversion:** a SymPy-to-program printer.
  - `Abs` → `abs`, `Max`/`Min` → `max`/`min`
  - PySR's `greater`/`cond` piecewise forms → `a if cond else b` / comparisons
  - integer powers → repeated multiplication
  - anything else → the candidate is invalid, and the reason is recorded
- **Installation:** an optional dependency group `baselines`. The first import downloads Julia
  through juliacall. Task 1 is a feasibility spike: install PySR, do one warm-up import (so Julia
  compile time is not counted), then one run on `absdiff/hand`.
  - If installation fails, or that run returns no equation within the timeout, B1/B2 are dropped,
    the report says why, and step 4 continues with the others.
  - The settings above do not change after the spike.

### 4.3 B4: exact activation-region decomposition
- `nndc/regions.py` (planned for step 5 in the parent design, moved here).
- Recursive enumeration of feasible activation patterns. At each node, take the next unit in
  topological order; its pre-activation is affine inside the current region. Split it into `> 0`
  and `<= 0` and check each half for feasibility with an LP (`scipy.optimize.linprog`, HiGHS).
  Infeasible branches are pruned.
- Budget: 4,096 regions or 60 s per model. Over budget, the run records "budget exceeded" and no
  program, and the report shows that. This is itself a result: exact methods don't scale.
- Emitted as a nested if/else on the units' pre-activation signs (an exact decision tree) with an
  affine return per leaf. It is exact by construction, real-arithmetic, and still checked.
- For integer inputs the LP works on the real relaxation, so a few regions without an integer
  point may be counted. This is accepted and noted.

### 4.4 B5: Claude Code from input/output only
- `nndc agent prepare --io <run>` builds a blind folder containing:
  - `samples.csv`: 2,000 seeded samples of inputs and the net's output, plus the class for binary
  - the blind spec
  - `INSTRUCTIONS.md`: the same refactoring intent, adapted to "this is a black-box function given
    by samples"; no snapping hints beyond the main instruction
  - `check.py`: the same verdict and counterexample output as before, but its input set has **no
    points derived from the net's units** (uniform samples, grid and corners only). The full
    checker's boundary points come from the net's activation boundaries and would leak structure
    (the kink locations) to an agent that queries repeatedly. The final evaluation of
    `io_agent_<n>.py` still uses the full checker.
- The weights must stay out of reach.
  - They are needed by `check.py`, so they are stored obfuscated as `oracle.bin`.
  - The instruction says to use only `samples.csv` and `check.py`.
  - The audit additionally flags any transcript access to `oracle.bin` or `_checker/`. A flagged
    run is excluded from the headline, as before.
  - This is best effort, stated as such in the report.
  - If more than a third of B5 runs are audit-flagged, B5 is reported as invalid rather than
    summarized.
- Run exactly like the step-2 agent pass: one fresh blind subagent per model, all 29 models, one
  attempt each. `nndc agent collect --transcript` copies back `io_agent_<n>.py`, notes, checks
  log and meta.
- Cost: about 29 agent runs, roughly the same as step 2.

## 5. Reporting

- **Population:** the 29 models in `runs/`. Every method is reported with the same split:
  held-out (6 seed-2 runs) and in-sample (23). The split only matters for the AST tool, but a
  uniform layout keeps rows comparable. `runs_datasets/` (dataset tasks, separate work in
  progress) is out of scope.
- **Existing rows are reused, not re-run:**
  - Claude (dump): the best valid `agent_n` attempt, chosen by the same rule as the reproduction
    section.
  - AST tool: the committed `ast.py`, produced by the frozen tool `1076f21`.
- **Columns:** "passes check" = `snap_pass`, "exact" = `exact_pass`.

`REPORT.md` gains a "Comparison (step 4)" section:
- **Method table:** Claude (dump), Claude (input/output only), AST tool, PySR, PySR+snap, tree,
  regions. Rows show passes check, exact, recovered (in and out of range), median nodes, median
  time, and the number of runs with no program. Held-out and in-sample are shown separately.
  Audit-flagged runs are excluded.
- **Recovery matrix:** run × method, ✓/✗.
- **Three short "what this shows" lines,** computed, not hand-written:
  - B5 vs Claude-dump: the effect of reading weights
  - B2 vs B1: the effect of snapping on search
  - B4 size vs the others: exact vs short

## 6. Calls made by Claude

- All baseline hyperparameters in §4 (sample counts, PySR settings, leaf sweep, region budget,
  snapping tolerances).
- Checker-guided "smallest passing" selection for the tree and PySR+snap, for parity with the AST
  tool. Raw PySR uses its own selection.
- Moving `regions.py` from step 5 to step 4.
- B5 on all 29 models with one attempt each. Weights are obfuscated plus audited rather than
  sandboxed.
- Dropping B1/B2 if PySR's Julia install fails here, rather than switching to gplearn.
- B5's checker gives feedback without net-derived boundary points, so its feedback doesn't leak
  the net's structure. This is the most consequential call: it makes B5 a cleaner test of "does
  reading the weights help", at the cost of slightly weaker feedback than the dump condition had.
- Binary targets: PySR fits the score, the tree fits the class.
- Reuse of the existing Claude and AST results; uniform held-out/in-sample split for all methods.
- Literature report and notes stay committed in `reports/` and `research_notes/`.
