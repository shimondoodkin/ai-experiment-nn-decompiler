# Sigmoid nets (project B)

Status: draft, 2026-10-28. Follows project A (`2026-10-28-activations-design.md`, merged), which
taught the code that a hidden layer can be `sigmoid`.

## Why

The user asked whether the decompiler can handle sigmoid as well as ReLU. A sigmoid that feeds a
decision reads as an `if`: `sigmoid(z) > 0.5` is exactly `z > 0`. A throwaway spike showed four
behaviours:

- **Sharp units become steps:** xor and window nets, and neareq after longer training. Each unit can
  become `1 if z > 0 else 0`.
- **Soft but nearly straight units become lines:** for example Iris-petal. A unit can become
  `0.5 + z/4`, or that line clamped to 0..1.
- **Real curves stay sigmoid:** absdiff and tax regression need some units to stay `sigmoid()`.
- **Rounded corners:** the net rounds the corners of the intended formula. The sharp formula
  (`abs(x0 - x1)`, the tax brackets) fails the checker against the net, but only near the kinks: up to
  11% of the output range for absdiff, and 2.2% against a 2% limit for tax.

**The user's direction:**
- Do for sigmoid nets what was done for ReLU nets in steps 1–3.
- **Do not decide the faithful-or-snapped question up front.** It "needs trial and learning". Build
  the experiment so the results show what happens; the user and Claude decide a policy afterwards.
- Sigmoid first. Tanh follows later as its own short project.
- The code is written natively in the session. Blind agent passes are still fresh subagents,
  because that is what keeps them blind.

## Goals

1. **Sigmoid nets for every trained task.**
   - Formula tasks: xor, window, neareq, absdiff, median3 and tax, seeds 0–2.
   - Iris and Iris-petal: seeds 0–5.
   - That is 30 nets. Same sizes, steps, learning rates and seeds as the ReLU runs; only the
     hidden activation changes.
   - They live under a new root, `runs_sigmoid/<task>/seed<k>/`, with the same run-folder layout.
2. **Build command:** `nndc run <task|all> --act sigmoid --root runs_sigmoid` builds them.
   - Handcrafted (`hand`) nets are ReLU by construction, so they are skipped for sigmoid.
3. **The intended formula as a version.** For each formula-task sigmoid run, `formula.py` holds the
   task's `expected` formula (from `experiments/tasks.py`). `evaluate_run` checks it against the net
   and grades it against the task like any other version.
   - This is the "snapped" reference. It answers: how far is the intended formula from what this
     net does, and where?
   - Iris has no formula, so there is no `formula.py` there.
4. **Blind Claude agent passes (step 2 again):** one fresh subagent per sigmoid net, all 30, with the
   same instructions, blind folder and collection as step 2. Their notes and `## Heuristics` sections
   show what an agent does with soft units, and whether it snaps or stays faithful when the checker
   pushes back. Heuristics go to a separate file, `runs_sigmoid/HEURISTICS.md`, not the ReLU
   `HEURISTICS.md`.
5. **AST tool moves for sigmoid units (step 3 again).** These are per-unit replacements, tried with
   the most saturated unit first. A replacement is kept only if the whole program still passes the
   checker:
   - **step:** `(z > 0)`, which the language evaluates to 1/0;
   - **line:** `0.5 + 0.25 * z`, folded into the consumers as an affine expression;
   - **clamped line:** `min(max(0.5 + 0.25 * z, 0), 1)`.

   Plus whatever moves the seed 0–1 agents' heuristics suggest. After replacement, the existing
   structural routes (pruning, merging, motif rewriting) work on the result where they apply.
6. **Held-out discipline, as in step 3.**
   - Tool development uses formula seeds 0–1 and Iris seeds 0–2 only.
   - The sigmoid moves are committed, which is the freeze, before the tool runs on the held-out
     sigmoid nets: formula seed 2 and Iris seeds 3–5.
   - The freeze commit is recorded in `run_git.txt` as usual.
7. **A report for the sigmoid runs:** `runs_sigmoid/REPORT.md`, written by
   `nndc report --root runs_sigmoid --out runs_sigmoid/REPORT.md`. It contains:
   - the results table and the per-net sections (dump, best agent program, AST program, notes), as in
     `REPORT.md`;
   - the step-3 reproduction section, for the sigmoid tool on held-out vs in-sample nets;
   - **a new section, "Formula vs net":** for each formula-task net, the formula's verdict against the
     net, its worst error (or disagreements outside the unsure zone), its grade against the task, and
     the top counterexamples (inputs, the net's output, the formula's output, the true answer). Next
     to it, whether the agent's program and the AST program are faithful (pass against the net) and
     whether they recover the formula;
   - **a new section, "Sigmoid vs ReLU":** one row per task and seed. It shows whether the net
     learned the task, the best agent program (pass? recovered? nodes) and the AST program (pass?
     recovered? nodes) for both activations, read from `runs/` and `runs_sigmoid/`.

   The step-4 comparison and fix sections are not part of this report.
8. **Afterwards, not part of the build:** the user and Claude read the report together and write a
   sigmoid section in `CONCLUSIONS.md`, including what to do about rounded corners.

## Non-goals

- Tanh. It comes later, reusing everything here.
- Step-4 baselines (PySR, tree, regions), samples-only agents, and fix/reinsert experiments on
  sigmoid nets.
- Deciding the faithful-vs-snapped policy, or changing the checker's tolerances for smooth nets.
- `compile_back` for sigmoid programs.
- A decision-list route that reads sigmoid or step units directly. It runs only once every unit is
  relu or linear; see Decisions. Making the decision list understand steps is left for later.
- Re-running anything on ReLU nets.

## Anti-goals

- **Any change to an existing ReLU result.** Nothing under `runs/` or `runs_bugs/` is rewritten. The
  equivalence script from project A (`evaluate` and `ast` on a copy, then `compare`) must report 0
  differences after the AST tool changes.
- **Tuning training for sigmoid** (steps, learning rate) per task to make nets look better. Nets that
  fail to learn stay, and are marked, like ReLU xor/seed2.
- **Leaking the experiment into blind folders.** Agents must not be told this is about sigmoid
  rounding or snapping. The dump itself shows `sigmoid(...)`, which is fine.
- **Tool development seeing held-out nets or their agents' notes.**
- **Deleting or overwriting older runs**, per CLAUDE.md: only new files and the new root.

## Constraints

- Everything from CLAUDE.md:
  - experimental discipline;
  - `data:` commits are separate from code commits;
  - blind-folder modules stay stdlib + NumPy;
  - older runs stay;
  - destructive commands only on scratch copies.
- **Agent passes:** at most 20 subagents at once, so 30 nets means two waves. Collect with
  `--transcript` from each subagent's JSONL, as in step 2.
- **A new AST unit kind needs a symbolic-net activation.** A unit that became a step or a clamped
  line needs new values in `Unit.act` (for example `step` and `clamp`), with evaluation, intervals and
  `to_program` support. Project A's fail-closed guards keep the ReLU-only routes from misreading them.
  A plain line becomes a `linear` unit, so the existing pruning inlines it.

## Decisions (my calls; the user can override)

- **Run root:** `runs_sigmoid/`, separate from `runs/`, so ReLU runs stay untouched and both reports
  can be compared directly.
- **`formula.py` only in sigmoid runs.** Adding it to ReLU runs would change their `results.json`.
  The ReLU formula comparison can come later.
- **Agents run on all 30 nets, held-out seeds included,** as in step 2. "Held out" applies to AST-tool
  development only.
- **Order:**
  1. CLI and `formula.py`, then build the 30 nets (code commit, then data commit).
  2. Agent passes on all 30 (data commit).
  3. AST moves, developed on seeds 0–1 and Iris 0–2 (code commit = freeze).
  4. The held-out tool run (data commit).
  5. The report.
- **Greedy order for unit replacement:** most saturated first, measured as the share of checker inputs
  where the unit's output is outside (0.05, 0.95). That is the spike's measure.

- **The spike touched only the development set.** It trained seeds 0 and 1 (window and neareq also
  seed 0 at 4x steps), and Iris and Iris-petal seeds 0–1. Training is deterministic, so those are
  the same nets as development seeds 0–1. No held-out net (formula seed 2, Iris 3–5) was seen.
- **Held-out agents' heuristics go to a separate file.** `collect` writes development-set runs'
  `## Heuristics` sections to `runs_sigmoid/HEURISTICS.md`. Held-out runs' sections go to
  `runs_sigmoid/HEURISTICS_heldout.md`, which tool development does not read. The step-3 report may
  then claim the held-out result honestly.
- **`formula.py` is a reference, not a candidate.** Its counterexamples are not added to
  `counterexamples.json`, and the checker never reuses them for other versions. Otherwise the
  formula's deliberate kink failures would make agent and AST checks harder on sigmoid runs than on
  ReLU runs. The formula's own counterexamples are kept in its `results.json` entry, as for every
  version, and the report reads them from there.
- **Iris gets tool support, as in the ReLU runs.**
  - The decision-list route runs when every remaining unit, after replacement, is `relu` or
    `linear`. The fail-closed guard already allows that.
  - A new black-box route for argmax nets fits one linear score per class to the net's own
    decisions on the checker's input set, snaps the coefficients with the existing `nice()` ladder,
    and emits `argmax(...)`. It is kept only if it passes the checker.
  - The spike showed three linear scores pass on Iris-petal and miss narrowly on 4-feature Iris.
  - This route works for any activation, but it runs only on argmax nets whose units are not all
    ReLU, so ReLU results cannot change.
- **The stopping rule for tool development** is a fixed list of moves, written into the build plan
  before any tool code:
  - the three unit replacements;
  - the linear-scores route;
  - any move that at least two development-set agents name in their heuristics.

  Development may fix bugs in these moves, judged on the development set only, but adds no moves
  after seeing their results. Then comes the freeze commit.
- **Blind index and heuristics paths follow `--root`.** For sigmoid runs that means
  `runs_sigmoid/blind_index.json` and `runs_sigmoid/HEURISTICS*.md`, so nothing under `runs/`
  changes. `prepare`/`collect` get a `--root` option; the default keeps today's paths.

## Left to the builder

- Names, module layout, CLI flag spelling beyond `--act`.
- How `formula.py` is generated: once at build time, or by a small command.
- How the report's new sections are laid out, within the content listed in Goal 7.
- The exact selection rule among AST candidates for sigmoid nets. The default is the existing one:
  fewest nodes, with exactness breaking ties.
