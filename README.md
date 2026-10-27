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

## Usage

```bash
uv sync
uv run pytest                      # all tests; no Claude needed (uses a mock agent)
uv run nndc run all                # build/train every step-1 model into runs/<task>/<hand|seedK>/
uv run nndc check my.py --run runs/absdiff/seed0
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
