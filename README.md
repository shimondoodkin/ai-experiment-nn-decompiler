# nndc: decompiling small ReLU networks into code

A ReLU neuron is an `if`: `max(0, z)` means "if z > 0 then z else 0". Combinations of neurons form
program structures such as `a > b`, `|a - b| < eps`, ranges, max/min and comparisons against stored
values. Gradient descent learns these with blurry constants (`0.0013` where the algorithm means `0`).
This experiment treats a trained network as **a dump of code that roughly resembles some
functionality**: it decompiles the net into a readable program, lets Claude Code refactor it
(snap constants, detect fixtures, form formulas, name variables, remove what doesn't help), and
checks every version against the original network.

Status: steps 1–4 done:
1. the decompiler and checker;
2. Claude Code refactoring;
3. a deterministic tool distilled from the agents' notes;
4. a comparison with existing methods, and decompile → fix → reinsert.

See [the design](docs/design/2026-10-27-nn-decompiler-design.md) for the full plan, [REPORT.md](REPORT.md)
for results, and [CONCLUSIONS.md](CONCLUSIONS.md) for what each step taught us.

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

## Sigmoid nets

The same pipeline on 30 nets with sigmoid hidden units, the same tasks, seeds and training settings: blind Claude agents, and the AST tool with sigmoid moves (step, line, clamped line, constant; linear scores and a linear cascade for argmax; a knot fit for 1-input regression), frozen before the held-out run. Each formula task also carries its intended formula as a reference program. Results, the formula-vs-net comparison and the comparison with the ReLU runs are in [runs_sigmoid/REPORT.md](runs_sigmoid/REPORT.md).

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

Like MIPS, nndc automatically distills small trained ReLU networks into Python with snapped constants;
unlike it, nndc targets feed-forward MLPs on real-valued piecewise tasks, has a task-blind LLM agent
refactor a per-neuron code dump into a program with non-integer intended constants, accepts a program
only by comparison with the network itself, and tests how much of that agent's work a frozen
deterministic tool can reproduce on held-out models — a combination we did not find in prior work.

Step 4 also tests decompile → fix → reinsert. A network that is wrong on its task is decompiled, an
agent fixes the program from failing cases, and the program is compiled back into a ReLU net. It is
compared with fixing the weights directly (fine-tuning, a last-layer LP repair). Prior repair methods
change weights; we did not find one that fixes the network through its decompiled code.

Literature reviews:
- [Neural net decompilation prior work](reports/Neural%20net%20decompilation%20prior%20work.md)
- [Counterexample guided program refinement](reports/Counterexample%20guided%20program%20refinement.md)
- [Decompile, fix and reinsert network blocks](reports/Decompile%20fix%20and%20reinsert%20network%20blocks.md)

The step-4 comparison against these methods (PySR, decision trees, exact region decomposition, Claude
from input/output only) is in the "Comparison (step 4)" section of [REPORT.md](REPORT.md).
