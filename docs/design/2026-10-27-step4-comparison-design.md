# Design: step 4, comparison with existing methods

Date: 2026-10-27
Status: draft for approval
Parent design: `docs/design/2026-10-27-nn-decompiler-design.md` (§5 step 4)
Literature: `reports/Neural net decompilation prior work.md` `reports/Counterexample guided program refinement.md` and `reports/Decompile fix and reinsert network blocks.md` (notes in `research_notes/`)

## 1. Intent

The literature search found that every component of this repo has a precedent:
- MIPS already turns small ReLU nets into Python with integer snapping.
- Exact ReLU-to-tree conversion exists.
- Symbolic regression finds formulas.
- LLMs describe functions from input/output behaviour.

The combination was not found anywhere: a task-blind coding agent reading the net's weights as code,
non-integer "intended" snapping, and a distilled deterministic tool.

Step 4 measures the repo's two methods against runnable representatives of each existing
approach, on the same models (29 formula nets, 12 Iris nets), with the same checker and the same grading. Each baseline
answers a specific question:

| Baseline | Question it answers |
|---|---|
| **B1: PySR on samples of the net** (raw) | Does an established formula search find the same programs from behaviour alone? |
| **B2: PySR plus this repo's snapping** | Is snapping the missing ingredient, or the search? (The report's suggested split: "separates search from snapping".) |
| **B3: decision-tree distillation** (sklearn) | How does the classic approximate rule-extraction baseline (TREPAN-style) do on size and fidelity? |
| **B4: exact activation-region decomposition** | How big is the exact, faithful but unreadable program, i.e. the Aytekin/EC-DT/RENTT family? |
| **B5: Claude Code from input/output only** (no dump, no weights) | Is reading the weights as code what makes the agent succeed? The report calls this the most important experiment. |
| **R: decompile, fix, reinsert** (the user's idea, §4.5) | When the net itself is wrong, can the agent fix the decompiled program from failing task cases and put it back into a net, and does that beat fixing the weights directly? |

The straight-line compiled-C speed baseline belongs to step 5.

### The checker loop (noted, not ablated)

The Iris runs show the agent repairing its own program from the checker's failing inputs: it moves a
threshold, or splits one line into two half-planes joined by `or`. The checker logs
(`agent_1_checks.log`, one line per `check.py` call) give its size:

| Set | Runs | Runs with any FAIL | Started with FAIL | Most checks in one run |
|---|---|---|---|---|
| Formula tasks (`runs/`) | 29 | 0 | 0 | 3 |
| Iris (`runs_datasets/`) | 12 | 7 | 5 | 16 (iris_petal seed2: 6 FAILs, then PASS) |

This is counterexample-guided synthesis with a sampling checker; prior work is in
`reports/Counterexample guided program refinement.md`. The user judged that a full ablation of it
would add little, so step 4 only reports these counts for every agent method. No extra agent runs.

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
- For the fix experiment (§4.5): efficacy, generalization and drawdown for every fixer, against
  weight-space repair; and a reinserted net checked against the fixed program.
- For every agent method: checker calls and FAILs before the first PASS (from the existing logs), time and tokens.

### Non-goals
- KAN, gplearn, AI Feynman, ruleex/REMIX, parx, MIPS's own code. They are cited, not run: install
  friction on Windows/Python 3.13, or a different model class.
- Tuning any baseline per task.
- Speed benchmarks (step 5).
- A budget-matched resampling baseline (many one-shot agents at the same token cost, as in Olausson et
  al.). Noted as a limitation instead.
- A formal equivalence check (MILP/SMT against the ReLU net) as the final verdict. A PASS stays evidence
  from sampling, not proof; a certifying check is a candidate for step 5.
- Fixing one block inside a bigger net (a composite net or a transformer MLP block). Here the block is
  the whole small net; multi-block nets come with the bigger models in step 6.
- A repair loop inside the AST tool. Tool v2 already does a batch version (thresholds centred in the range
  the confident points allow); a counterexample-driven tool loop is step-3 follow-up work.

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
  B5 uses the agent commands (§4.4).

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
- Run exactly like the step-2 agent pass: one fresh blind subagent per model, all 41 models (29 formula, 12 Iris), one
  attempt each. `nndc agent collect --transcript` copies back `io_agent_<n>.py`, notes, checks
  log and meta.
- Cost: about 41 agent runs (29 formula nets and 12 Iris nets).

### 4.5 R: decompile, fix, reinsert
The user's idea: take a block out of a net, decompile it, run it on the cases where **the net** is wrong
on the task, have the agent fix the algorithm in the code, and put it back. Here the block is the whole
small net. Prior work fixes nets from failing cases only by changing weights (PRDNN, Arachne, ROME/MEMIT)
or fixes logic only in models that were readable to begin with (GAM Changer). The full cycle was not
found (`reports/Decompile fix and reinsert network blocks.md`).

The existing results already show part of it. Snapping alone fixes the small errors of most trained
formula nets: their program recovers the true formula where the net does not. So the fix experiment needs
nets whose faithful program is still wrong.

**Populations.** All are wrong on the task in a way the program copies faithfully.
- **P1, planted bug (main):** 5 formula tasks (absdiff, median3, tax, window, xor), seed 0, trained by
  `experiments/train.py` with one defect. During training, the target is wrong inside one axis-aligned
  sub-box of 10% of the training box, drawn with seed 0 and recorded in `train_log.json`:
  - regression: true value plus 20% of the output range;
  - binary: the label is flipped.

  The net learns the bug, the truth is known, and the true formula gives unlimited fresh data. Each P1
  net first gets the normal decompile: the AST tool and one step-2 agent pass.
- **P2, natural bugs:** the 3 existing nets whose agent program still misses the formula:
  - tax seed0 and seed1: the net learned misplaced brackets;
  - xor seed2: the net never learned xor.
- **P3, Iris (stress test):** the 12 nets. On the 150 flowers they get 2–7 wrong, many in the
  versicolor/virginica overlap, where no clean rule exists. This tests overfitting, not repair.

**Data.**
- **P1/P2:**
  - repair set: 2,000 points from the training distribution, labelled with the true task;
  - held-out set: 20,000 fresh points;
  - out-of-range set: 20,000 points.
- **P3:** a stratified 100/50 split of the flowers (seed 0) is the repair/held-out pair. Non-agent
  fixers also repeat 5 two-fold splits, because one flower is 2 points of held-out accuracy.
- The fixer sees only the repair set. The held-out failures are never shown.

**Fixers.** Each one starts from the same net and the same repair set.

| Arm | What it is |
|---|---|
| **R-agent** | Blind Claude. Its folder holds the faithful program (`program.py`, the best valid `agent_n` or, for P1, the new decompile), `repair.csv` (inputs, net output, true label, failing flag) and `score.py`, which reports efficacy and drawdown on the repair set only. Instruction: fix the program so it is right on the task data, keep it unchanged where the net is right, keep it short. No task name. |
| **R-io** (P1, P2) | Same as R-agent, but with no program. Does the decompiled program help, or does the agent just learn the task from the labels? |
| **R-tree** | A decision tree fitted on the repair set, with the B3 sweep selected on the repair set. A cheap "learn from data only" control. |
| **W-ft** | Fine-tune the net on the repair set (Adam, lr 1e-3, 300 steps, full batch). |
| **W-ft-anchor** | Fine-tune on the repair set plus 2,000 anchor points labelled by the original net where it is right, to limit drawdown. |
| **W-lp** | PRDNN-style one-layer repair: an L∞-minimal change to the last layer's weights such that every repair point is right, solved as an LP with scipy (HiGHS). Infeasible is recorded as a result. |
| **W-retrain** (P1) | Retrain from scratch on clean labels. The ceiling. |

**Reinsertion.** `nndc/compile_back.py` turns a fixed program into a ReLU MLP.
- relu/min/max/abs/affine programs are continuous and piecewise linear, so they compile exactly
  (min(a,b) = a − relu(a − b), and so on).
- Binary and argmax programs compile to scores whose sign or argmax gives the same decision. A decision
  list `if t1: 0 elif t2: 1 else 2` becomes scores built from relu of the test margins. It is exact except
  for exact ties.
- A regression program with a jump (an `if` whose branches differ at the threshold) gets a ramp of width
  1e-4 of the input range. The report gives the fraction of inputs inside ramps.
- The reinserted net is checked against the fixed program with the existing checker (net vs program).
  It must PASS.
- Not every construct compiles. Unsupported ones (for example `argmax` over non-linear expressions) are
  recorded, and that arm reports the program only.
- The fixed program running as code is the CPU path; its speed is step 5.

**Metrics.** For every arm, on P1/P2 and on P3 separately:
- **Efficacy:** the repair-set failures that are now right.
- **Generalization (headline):** the held-out failures of the original net that are now right.
- **Drawdown:** the held-out points the original net got right that are now wrong. For P3 it is measured
  on flowers only, because off the data the net's answers are arbitrary.
- **Task:** accuracy or error in range and out of range (P1/P2).
- **Program side:** nodes, tests added by the fix, and how many repair points support each added test.
  An added test supported by fewer than 5 repair points is flagged as a likely memorised special case.
- **Reinsertion:** the reinserted net passes the checker against the fixed program; the reinserted net's
  task metrics equal the program's.
- Agent time and tokens.

**Cost.** Agent runs:
- P1: 5 decompile + 5 fix + 5 io = 15
- P2: 3 fix + 3 io = 6
- P3: 12 fix
- Total: 33

## 5. Reporting

- **Population:**
  - **Formula tasks:** the 29 models in `runs/`. Every method is reported with the same split:
    held-out (6 seed-2 runs) and in-sample (23). The split only matters for the AST tool, but a uniform
    layout keeps rows comparable.
  - **Iris:** the 12 nets in `runs_datasets/` (iris and iris_petal, seeds 0–5), in a separate table.
    "Recovered" is meaningless there (1-NN target), so the columns are checker verdict, extra-seed
    disagreements, size and accuracy on the 150 flowers. The AST row is frozen tool v2, which is
    in-sample on seeds 0–2 and held-out on seeds 3–5. The tree baseline is the natural textbook
    comparison on Iris.
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
- **Fix table (§4.5):** arm × {efficacy, generalization, drawdown, task in/out of range, nodes, flagged
  special cases, reinsertion PASS}, as medians over nets, for P1+P2 and P3 separately. Per-net rows
  show the original program next to the fixed one: the readable diff of a network repair.
- **Loop columns for every agent method:** checker calls and FAILs before the first PASS, from
  `*_checks.log`.

## 6. Calls made by Claude

- All baseline hyperparameters in §4 (sample counts, PySR settings, leaf sweep, region budget,
  snapping tolerances).
- Checker-guided "smallest passing" selection for the tree and PySR+snap, for parity with the AST
  tool. Raw PySR uses its own selection.
- Moving `regions.py` from step 5 to step 4.
- B5 on all 41 models (formula and Iris) with one attempt each. Weights are obfuscated plus audited rather than
  sandboxed.
- Dropping B1/B2 if PySR's Julia install fails here, rather than switching to gplearn.
- B5's checker gives feedback without net-derived boundary points, so its feedback doesn't leak
  the net's structure. This is the most consequential call: it makes B5 a cleaner test of "does
  reading the weights help", at the cost of slightly weaker feedback than the dump condition had.
- Binary targets: PySR fits the score, the tree fits the class.
- Reuse of the existing Claude and AST results; uniform held-out/in-sample split for all methods.
- Literature report and notes stay committed in `reports/` and `research_notes/`.
- (The user's fix-and-reinsert idea) Experiment R (§4.5): planted-bug nets as the main population;
  3 natural-bug nets and the Iris nets as secondary ones. Fixers: the blind agent, an input/output-only
  agent, a tree, and weight-space repair (fine-tune, anchored fine-tune, one-layer LP, retrain).
  Exact compile-back with ramps for jumps. Multi-block nets are left to step 6. PRDNN's own code is
  not run; its last-layer LP is re-implemented with scipy (APRNN needs Gurobi).
- The planted defect (10% sub-box; +20% of output range, or a flipped label) and all fine-tuning
  settings are fixed here, before any run.
- Total new agent runs: 41 (B5) + 33 (R) = 74, one attempt each.
- (After the Iris runs) The Iris nets join the population as a second table. The checker loop is
  reported from logs only; budget-matched resampling, a formal equivalence check, and a repair loop
  in the AST tool are left out of step 4.
