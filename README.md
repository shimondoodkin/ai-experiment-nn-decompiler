# nndc: decompiling small ReLU networks into code

A ReLU neuron is an `if`: `max(0, z)` means "if z > 0 then z else 0". Combinations of neurons form
program structures such as `a > b`, `|a - b| < eps`, ranges, max/min and comparisons against stored
values. Gradient descent learns these with blurry constants (`0.0013` where the algorithm means `0`).
This experiment treats a trained network as **a dump of code that roughly resembles some
functionality**: it decompiles the net into a readable program, lets Claude Code refactor it
(snap constants, detect fixtures, form formulas, name variables, remove what doesn't help), and
checks every version against the original network.

Status: steps 1–2 (decompiler, checker, Claude Code refactoring). See
[the design](docs/design/2026-10-27-nn-decompiler-design.md) for the full plan, and
[REPORT.md](REPORT.md) for results.

## Step 3: how much of this is mechanical?

`nndc ast` is a deterministic tool with no LLM. It implements the heuristics Claude Code wrote down
in [HEURISTICS.md](HEURISTICS.md): dead and duplicate units, snapping, ReLU motifs (abs, max, min,
clamp, hat), 1-D spline reading, decision-boundary measurement, truth tables, and a small
candidate-formula fit. It never sees the task. Heuristics were distilled only from the handcrafted,
seed-0 and seed-1 runs. The tool was frozen before it was run once on the held-out seed-2 models.

- **Held-out:** it reproduced Claude's refactoring on 5 of 6 models (absdiff, median of 3, window,
  near-equality, and the failed XOR net, where both find `not x0`).
- **Where only Claude succeeded:** the tax brackets. Claude recognized a progressive tax rule and
  snapped it to `0.1 / 0.2 / 0.4` with brackets at 20 and 80. The tool produced a valid but blurred
  `max(0.12 x, 0.2 x - 1.68, 0.4 x - 17.68)`.
- **In-sample:** 21 of 21.

Details are in the "Reproduction (step 3)" section of [REPORT.md](REPORT.md).

## A real dataset: Iris

The same pipeline on nets trained on the 150 Iris flowers (all 4 measurements, and petal only), 3 classes,
seeds 0–5. There is no known formula, so success means a short program that passes the checker against
the net. Every net reduces to an if-chain of two linear tests. The AST tool got a decision-list route for
argmax nets; its second version produced passing lists on all 6 fresh held-out nets, as did blind Claude
agents. Details are in [runs/iris/REPORT.md](runs/iris/REPORT.md).

Each run folder records the git commit each step ran with in `run_git.txt`.

## Usage

```bash
uv sync
uv run pytest                      # all tests; no Claude needed (uses a mock agent)
uv run nndc run all                # build/train every model (step-1 and Iris) into runs/<task>/<hand|seedK>/
uv run nndc check my.py --run runs/absdiff/seed0
uv run nndc ast --all              # deterministic heuristics tool (no LLM); --split heldout for seed 2
```

Refactoring with Claude Code (you run Claude yourself; the tool never launches it):

```bash
uv run nndc agent prepare runs/absdiff/seed0     # prints a blind folder outside the repo
# in that folder:  claude "Read INSTRUCTIONS.md and follow it."
uv run nndc agent collect <blind folder>         # copies the attempt back, audits, re-checks
uv run nndc report                               # writes REPORT.md
uv run nndc gate                                 # spike gate for steps 1-2
```

The blind folder contains only the dump, a spec with neutral input names (`x0, x1, …`), the
weights, the instructions and a self-contained `check.py`. It contains no task name, no real input
names, and no target.

## Prior work

Rule extraction from neural networks is an established field (TREPAN, DeepRED, symbolic execution of
ReLU networks, linear-region analysis, symbolic regression). A comparison with these methods is
planned for step 4 of the design.
