# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research experiment (`nndc`). It decompiles small trained ReLU MLPs (1–8 inputs) into readable
programs, refactors them, and checks every program against the original network. The plan and
rationale live in `docs/design/` (one design doc per step) and `docs/plans/` (one build plan per step).
Results are in `REPORT.md`, and for the Iris side run in `runs_datasets/REPORT.md`. Literature
reviews are in `reports/`, with raw notes in `research_notes/`.

## Commands

```bash
uv sync                                          # Python 3.13; torch comes from the CPU wheel index
uv run pytest -q                                 # full suite (no Claude needed; uses a mock agent)
uv run pytest tests/test_check.py::test_name -v  # a single test
uv run nndc run all                              # build/train every formula model into runs/
uv run nndc run iris --seed 0 --root runs_datasets
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

**Tasks:** `experiments/tasks.py` holds the formula tasks (`TASKS`, with a true target and an
`expected` formula written before any run). `experiments/datasets.py` holds the dataset tasks
(`DATASET_TASKS`, Iris with a 1-NN target). Dataset tasks are deliberately kept out of `TASKS`, so
`nndc run all`, `REPORT.md` and the step-1 tests don't see them. `get_task` resolves both.

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

## Experimental discipline (do not break)

- **Held-out data:**
  - formula runs: seed 2 is held out from AST-tool development (`--split distill` excludes it);
  - Iris: seeds 3–5 were held out from tool v2.
- **Freezes:** the commits are recorded in `runs/ast_freeze.txt` and
  `runs_datasets/ast_freeze*.txt`. Changing the AST tool or heuristics means a new freeze and a fresh
  held-out set, not re-running old held-out models.
- **Blindness:** no task name, real input names, `expected` formula or target may enter a blind
  folder, or the AST tool's inputs.
- **Agent attempts belong to specific weights:** `build_run --retrain` refuses to retrain over
  existing `agent_*` files.
- **Baselines:** hyperparameters are fixed in the design doc before any run, never tuned per task.
- **Commits:** data commits (new run results) are separate from code commits and use the `data:`
  prefix. Existing `agent_n` / `ast` results are reused, not regenerated.
