# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research experiment (`nndc`). It decompiles small trained ReLU MLPs (1–8 inputs) into readable
programs, refactors them, and checks every program against the original network. The plan and
rationale live in `docs/design/` (one design doc per step) and `docs/plans/` (one build plan per step).
Results are in `REPORT.md`, and for the Iris dataset runs in `runs/iris/REPORT.md`. Literature
reviews are in `reports/`, with raw notes in `research_notes/`.

## Commands

```bash
uv sync                                          # Python 3.13; torch comes from the CPU wheel index
uv run pytest -q                                 # full suite (no Claude needed; uses a mock agent)
uv run pytest tests/test_check.py::test_name -v  # a single test
uv run nndc run all                              # build/train every model (formula tasks and Iris) into runs/
uv run nndc run iris --seed 0                    # one model
uv run nndc check prog.py --run runs/absdiff/seed0
uv run nndc ast --all --split distill|heldout|all [--root R]   # deterministic refactoring tool
uv run nndc agent prepare <run>                  # blind folder for a Claude refactoring run (outside the repo)
uv run nndc agent collect <blind> --transcript <jsonl>         # bring it back, audit, re-check
uv run nndc report                               # regenerate REPORT.md from runs/
```

There is no linter configured.

## Architecture

**Run folders** are the unit of everything: `runs/<task>/<hand|seedK>/`. Each holds:
- `spec.json`, `weights.npz` (and `net.pt`), `train_log.json`;
- `dump.py`: the net as code;
- program versions: `agent_<n>.py`, `ast.py`, plus later baselines;
- `results.json`.

`nndc/run.py:evaluate_run` checks and grades every version present and rewrites `results.json`.
Adding a new kind of program means adding it to the version list there.

Each version gets two separate judgements:
- **Check against the net**, never the target (`nndc/check.py`). This decides acceptance.
  - PASS exact means equal within tolerance or a float32 tie.
  - PASS snap means regression error ≤ 2% of the output range, or binary/argmax disagreements only in
    the net's "unsure zone" (|score| < 5% of p95).
  - The input set is exhaustive when the domain is ≤ 1e7 points. Otherwise it is empirical: 200k
    uniform points + grid + corners + points bisected onto the net's activation boundaries.
  - Counterexamples accumulate in `counterexamples.json` and are re-used.
- **Grading against the true task** (`experiments/grade.py`). Report only; never used for acceptance.

**Program language** (`nndc/lang.py`): an allowlist subset of Python, parsed and evaluated vectorised.
Every program (dump, agent, tool, baseline) must parse under it. Output conventions:
- regression returns the value;
- binary: the decision is `raw > spec.threshold`;
- argmax returns a class index, or `argmax(...)`.

**Activations:** a hidden layer is `relu`, `sigmoid` or `tanh`. `MLP.acts` holds one name per hidden layer
(default all `relu`), and `weights.npz` stores an `acts` key only when a net is not all-ReLU. The symbolic net's
`Unit.act` is one of these or `linear` (no activation: inlined units and argmax scores), and dumps call
`sigmoid(...)` / `tanh(...)`. `compile_back`, the `regions` baseline and the ReLU motif and decision-list routes
are ReLU-only: they skip or refuse such nets.
Train one with `train(task, seed, act="sigmoid")`. The all-ReLU path must stay byte-identical;
`experiments/equivalence.py` checks that.

**Tasks:** `experiments/tasks.py` holds the formula tasks (`TASKS`, with a true target and an
`expected` formula written before any run). `experiments/datasets.py` holds the dataset tasks
(`DATASET_TASKS`, Iris with a 1-NN target). Dataset tasks are kept out of `TASKS`, so the step-1
tests don't see them; `get_task` resolves both. Their runs live in `runs/iris/` and `runs/iris_petal/`
next to the formula runs. `nndc run all` builds them (seeds 0–5), and `nndc ast --all` includes them only
with `--split all`. In `REPORT.md` they appear in the results table with `recovered` n/a, and are left
out of the step-3 reproduction and the beats-the-network list (`nndc/run.py: is_step1`).

**Run versions:** each run folder has `run_git.txt`, one line per step (`build`, `ast`, `agent_<n>`)
with the git commit the step ran with (`+dirty` for uncommitted code in `nndc/` or `experiments/`).
`nndc run`, `nndc ast` and `nndc agent collect` write it (`record_git`); lines marked `backfilled` were
reconstructed from git history.

**Blind agent runs** (`nndc/agent/`):
- `prepare` copies into `%TEMP%/nndc-blind/m<hex>/`:
  - the dump and the weights;
  - a blind spec (neutral names `x0..`, no task name);
  - `INSTRUCTIONS.md` (`nndc/agent/instruction.md`);
  - `check.py` plus a `_checker/` copy of `spec.py`, `model.py`, `lang.py`, `check.py`.

  Those four modules must stay **stdlib + NumPy only**, because they run inside blind folders.
- The folder→run mapping is in `runs/blind_index.json`.
- `collect` copies `agent.py` back as `agent_<n>.py` (plus notes, checks log and meta). It audits the
  transcript for paths outside the folder, re-evaluates, rolls back on failure, and appends the
  agent's `## Heuristics` section to `HEURISTICS.md`.
- Agent passes are run as one fresh general-purpose subagent per folder. Their transcripts are the
  subagent JSONL files, passed with `--transcript`.

**AST tool** (`nndc/ast_tool.py`, `nndc/symnet.py`, `nndc/heuristics/`):
- Parses `dump.py` into a symbolic net (units = `relu(affine)`).
- Builds candidate programs along routes: structural (prune, merge, snap with `nice()`, ReLU motifs),
  spline, boundary rule / boundary scan, truth table, fit, and the decision list for argmax nets.
- Keeps the checker-passing candidate with the fewest nodes, exactness breaking ties. Writes `ast.py`
  and `ast_trace.json`.
- It sees only the dump, the blind spec and the weights; it never sees the task.

**Step 4 pieces:**
- **Baselines** (`nndc/baselines/`: tree, regions, PySR) write extra program versions (`tree.py`,
  `regions.py`, `pysr.py`, `pysr_snap.py`) that `evaluate_run` picks up. Run them with `nndc baseline`.
  PySR needs `uv run --group pysr`; the real fit test is marked `slow` and deselected by default.
- **Agent folder kinds** (`prepare(kind=...)`, `nndc agent prepare --io|--fix|--fixio`):
  - `io`: samples plus a scrambled `oracle.bin`, checked without boundary points;
  - `fix` / `fixio`: labelled repair cases, with or without the decompiled program, scored by
    `score.py`.

  `collect` reads the kind from `runs/blind_index.json` and writes `io_agent_<n>` / `fix_<n>` /
  `fixio_<n>`.
- **Fix experiment** (`nndc/repair/`):
  - planted-bug nets live in `runs_bugs/` (`nndc run TASK --seed S --bug --root runs_bugs`);
  - `nndc repair baselines|evaluate` writes `repair/results.json` (efficacy, generalization,
    drawdown, reinsertion);
  - `nndc/compile_back.py` turns a program back into an exact ReLU net.
- **Comparison section:** `nndc/compare.py` builds the "Comparison (step 4)" section of `REPORT.md`.
- **Hand-written conclusions** per step are in `CONCLUSIONS.md`. Add to it when a step ends.

**Sigmoid runs** (project B, `runs_sigmoid/`):
- **Build:** `nndc run all --act sigmoid --root runs_sigmoid` builds them; `build_run` refuses a folder whose net
  has other activations. Runs of non-ReLU formula tasks get `formula.py`, the intended formula as a reference
  version. Its counterexamples are not added to `counterexamples.json`.
- **Agents:** `nndc agent prepare/collect --index runs_sigmoid/blind_index.json --heuristics runs_sigmoid/HEURISTICS.md`.
  Held-out runs' heuristics go to `runs_sigmoid/HEURISTICS_heldout.md`, which tool development must not read.
- **AST tool:** the sigmoid moves are in `nndc/heuristics/sigmoid.py`:
  - per-unit const / step / line / clamp, and near-duplicate step merging;
  - `linear-scores` and `linear-cascade` for argmax nets;
  - `knot-fit` for 1-input regression.

  They run only for nets with non-ReLU units.
- **Kink merge (`ast2`):** `nndc/heuristics/kinks.py` expands a one-layer sigmoid net into ReLU kinks, merges
  kinks on one line, keeps the strongest lines and refits them (design `docs/design/2026-10-29-kink-merge-design.md`).
  - `nndc ast --name ast2 RUNS…` writes `ast2.py` and `ast2_trace.json` next to the older `ast.py`, which stays.
  - `ast2` has its own `counterexamples_ast2.json` (`OWN_CEX` in `nndc/run.py`), so older verdicts never change.
  - Runs: development seeds 0–2; held-out seeds 3–5 of the five one-layer formula tasks.
- **Report:** `nndc report --root runs_sigmoid --out runs_sigmoid/REPORT.md --compare-to runs` adds the
  "Formula vs net", "Compared with the ReLU runs" and "Kink merge (ast2)" sections.
- **Agent scratch files:** `prepare` gives each blind folder a `scratch/` directory, and `INSTRUCTIONS.md` asks the
  agent to use it. Before that, agents wrote to the shared session scratchpad and the audit flagged it.

## Practical notes

- **Running agent passes:**
  - One fresh general-purpose subagent per blind folder; the prompt is only "Work only inside
    `<folder>`. Read INSTRUCTIONS.md there and follow it."
  - At most 20 subagents run at once.
  - Transcripts are at `~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl`. Keep a
    folder→agent-id map, and collect only agents that have finished.
  - Agents sometimes keep a background job running after their final report. Before relying on a
    collected attempt, check that its `agent.py` did not change.
- **Concurrent writes:** runs that are evaluated while another job writes the same `results.json` can
  race. Re-run `evaluate_run` over all runs before a data commit.
- **Shell:**
  - The rtk shell hook filters `git diff` output (it can come back empty). Use PowerShell or `rtk proxy`
    for raw diffs.
  - A global `*.log` gitignore exists, so `.gitignore` re-includes `runs*/**/*_checks.log`.
- **Long jobs:** PySR's first import downloads Julia (about 8 min). A PySR pass over 41 nets takes
  about 1.5 h, and `nndc report` re-runs the checker for the Iris table; run both in the background.
- **Audits:** outside-path flags are mostly false positives (`/len`, `/print`, `/50`). Review them
  rather than auto-excluding runs. Forbidden-file access (`oracle.bin`, `_checker/`) by `io` agents
  is the one hard exclusion.

## Experimental discipline (do not break)

- **Held-out data:**
  - formula runs: seed 2 is held out from AST-tool development (`--split distill` excludes it);
  - Iris: seeds 3–5 were held out from tool v2.
  - kink merge (`ast2`): sigmoid seeds 3–5 of xor, window, neareq, absdiff and tax, built after its freeze.
- **Freezes:** the tool version used on each held-out run is the `ast:` line of its `run_git.txt`
  (step 3: `1076f21`; Iris v1: `e62d74b`, later overwritten by v2; Iris v2: `363b8c9`; sigmoid: `15b943b`, whose
  held-out runs record `797e0ae`, a data-only commit on top of it; kink merge (`ast2`): `93d7199`, whose held-out
  runs record `fad7e52`, data and docs on top of it). Commit the tool
  before running held-out models. Changing the AST tool or heuristics means a new freeze and a fresh
  held-out set, not re-running old held-out models.
- **Blindness:** no task name, real input names, `expected` formula or target may enter a blind
  folder, or the AST tool's inputs.
- **Agent attempts belong to specific weights:** `build_run --retrain` refuses to retrain over
  existing `agent_*` files.
- **Baselines:** hyperparameters are fixed in the design doc before any run, never tuned per task.
- **Commits:** data commits (new run results) are separate from code commits and use the `data:`
  prefix. Existing `agent_n` / `ast` results are reused, not regenerated.
- **Older runs stay:** never delete or overwrite older runs (`runs/`, `runs_bugs/`, their
  `agent_*`, `ast`, baseline and `repair/` files, and checks logs). They are the record
  of the work and what the reports cite. New work goes in new files or a new root. To try a command
  that rewrites or deletes run files, work on a copy in a scratch folder outside the repo, and never
  point `rm -rf`, `shutil.rmtree`, `git clean` or `experiments/equivalence.py copy` at the repo, its
  parent or `runs*/`. Tests that exercise a delete must patch it out or use `tmp_path`.
