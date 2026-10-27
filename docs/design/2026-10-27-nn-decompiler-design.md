# Design: decompiling ReLU networks into code, and compiling that code for the CPU

Date: 2026-10-27
Status: draft for approval (rewritten around the discovery-first order)

## 1. Intent

A ReLU neuron is an `if`: `max(0, z)` means "if z > 0 then z else 0". Combinations of neurons
form recognizable program structures, for example:

- `a > b` (one neuron on `a - b`)
- `|a - b|` and approximate equality `|a - b| < eps`
- ranges, AND/OR, max/min, clipping, sums and counts
- comparisons against stored values ("fixtures"), such as the bytes of an ASCII keyword

Gradient descent learns these structures with blurry constants (`0.0013` where the algorithm
means `0`, `0.47` where it means "halfway between two integers"). The idea behind this project:
**a trained network is a dump of code that roughly resembles some functionality.** Refactoring
that dump, by snapping constants, detecting fixtures, forming formulas, naming variables, and
removing what doesn't help, may recover a program that is cleaner, smaller, faster, and sometimes
*more correct* than the network it came from.

The project is **discovery-first**. We don't design the refactoring rules up front. Claude Code
refactors the decompiled nets and writes down what it learned. The recurring heuristics are then
turned into a deterministic AST tool. Both are compared against existing methods, and only after
that does the work move on to compiling and speed.

It is published as a public git repo and should interest people from four angles:

1. **Decompilation / "what did the net really learn":** readable programs recovered from trained
   nets, compared with the known true algorithm or a known human-written heuristic.
2. **LLM as a decompiler:** what an LLM notices in decompiled nets, captured in `HEURISTICS.md`,
   and how much of its judgment can be turned into mechanical rules.
3. **Explainability:** a decompiled program is a global, checkable explanation over the whole
   input range, unlike local tools such as saliency maps.
4. **Optimization (later step):** compiling the recovered program into cheaper native code than
   the original net, by measured cost.

Rule extraction from neural nets is an established field (TREPAN, DeepRED, symbolic execution of
ReLU nets, linear-region analysis, symbolic regression). The repo must say so plainly, compare
against those methods (step 4), and not claim novelty it hasn't checked.

## 2. Goals, non-goals, anti-goals

### Goals
- A decompiler: trained PyTorch ReLU MLP → readable Python program, plus a checker that compares
  any program with the original net.
- Claude Code refactorings of those programs, with its notes and a shared heuristics log.
- A deterministic AST heuristics tool distilled from those notes, measured by how much of the
  Claude Code result it reproduces.
- A fair comparison with existing rule-extraction and formula-discovery methods.
- Later: C compilation, exact pruning, and honest speed benchmarks.
- A clear record of where the approach **fails**: nets that don't learn clean logic, refactorings
  that don't hold, region explosion, code slower than matrices.
- Built in steps; the repo is publishable after each step.

### Non-goals (v1)
- Arbitrary user models, CNN/transformer layers, GPU code generation, a polished CLI.
- Formal floating-point proofs. Z3 is not used in v1.
- Beating optimized inference libraries on large networks.

### Anti-goals (failure even if the code works)
- Claiming a result (decompiled, equivalent, faster, better) that wasn't checked or measured.
  If refactored code is bigger, slower, or wrong, the report says so.
- Calling random sampling a proof.
- Silently replacing a trained net with a handcrafted one when training fails.
- **Leaking the answer to the refactorer.** Neither Claude Code nor the AST tool may see the task
  name, the target function, or the training data. Targets are only for generating data and
  grading results.
- Hiding the main result under machinery. The README's first screen shows weights → program.

## 3. Constraints and environment

Checked on the author's machine (Windows 11, Intel i7-1165G7, 4 cores/8 threads):

- Python 3.13.1, PyTorch 2.11.0 (CPU), NumPy 2.3.5 installed; `uv` available.
- To add as needed: `scikit-learn` (decision-tree baseline), `scipy` (LP for region decoding),
  a symbolic-regression package (PySR or gplearn; PySR needs Julia), `gymnasium` and
  `stable-baselines3` (step 6 RL). `gymnasium[box2d]` may not install on Windows with Python 3.13.
- C (step 5): LLVM clang 21.1.8 at `C:\Program Files\LLVM\bin` needs the MSVC environment;
  MSVC 14.44 is installed (VS 2022). No `gcc`. The build must find the toolchain itself
  (e.g. `vswhere` + `vcvars64`) and produce a DLL loaded with `ctypes`. It should also work with
  `gcc`/`clang` on Linux/macOS.
- Claude Code CLI (`claude`) is on PATH; it is the LLM, with no separate API key.
- CPU only, fixed seeds, laptop-friendly run times. Later C output: C99, `-O2`, no fast-math.

## 4. The core idea of the structure: one program format

**Every version of a model is a readable program file in the same small language**: the raw dump,
Claude Code's refactoring, the AST tool's output, the pruned version, and the baselines where
they can be expressed in it. As a result:

- one checker (`check`) checks any version against the original net
- `diff` between any two versions shows what changed
- one C backend (step 5) compiles any version
- readers see code, not an internal data structure

### 4.1 The program language

A restricted subset of Python, parsed with Python's `ast` module against an **allowlist** of node
types and evaluated by the project's own interpreter. Programs are never `eval`'d or imported.

- One function per model. A header comment states input names, types (float/int/byte/bool),
  bounds, and the output meaning (regression, or class with threshold).
- Statements: assignment, `if`/`elif`/`else`, `return`.
- Expressions: numeric constants, `+ - *`, division by a constant, comparisons, `and`/`or`/`not`,
  `relu`, `min`, `max`, `abs`, `sum` over named inputs, bytes/tuple literals for fixtures
  (e.g. `x[0:4] == b"HTTP"`), and later `dense(W, x)` for blocks that should stay matrices.
- The allowlist is versioned. It grows only when the heuristics need it (step 3), and every
  addition gets an interpreter test.

Example dump (`dump.py`) of a small net:

```python
# inputs: a: float [-3, 3], b: float [-3, 3]
# output: class 1 if out > 0 else 0
def net(a, b):
    h0 = relu(0.9813*a - 1.0127*b - 0.1984)   # layer 1, neuron 0
    h1 = relu(-1.0021*a + 0.9967*b - 0.2011)  # layer 1, neuron 1
    h2 = relu(0.0013*a + 0.0008*b - 0.0041)   # layer 1, neuron 2
    return -2.31*h0 - 2.28*h1 - 0.0102*h2 + 0.4507
```

What a refactoring might look like:

```python
def net(a, b):
    return 1 if abs(a - b) < 0.2 else 0   # "near-equal"
```

### 4.2 The spec

Each model has `spec.json`: input names, types, bounds; output kind; classification threshold and
tie rule; tolerances `atol`/`rtol`; snap margin (see 4.3). The spec is visible to the
refactorer. The task name and the target are not.

**Output kinds:** `regression` (one number), `binary` (one output and a threshold), and `argmax`
(several outputs; the program returns a class index). `argmax` is needed for multi-action RL
policies and the argmax-of-4 task.

**Input names are neutral (`x0, x1, …`) in everything the refactorer sees.** Names like `alive`,
`dealer_card`, or `pole_angle` would give the task away. Types and bounds stay visible, since they
are legitimate context. `spec.json` also holds the real names, which are substituted only when
programs are shown in the report and README, so readers still see `if pole_angle + …`.

### 4.3 The checker and acceptance rule

`check <program>` compares a program with the original net (`net.pt`, evaluated in float32 by an
independent NumPy reference) and prints a verdict, a verification label, statistics, and
counterexamples. The tests, Claude Code, the AST tool, the comparison, and the report all use the
same checker.

**Acceptance rule (uses only the original net, never the target):**

- **Exact mode:** regression within `atol + rtol·|ref|`; zero classification disagreements.
- **Snap mode:** the program may disagree with the net **only where the net itself was unsure**.
  Outside that zone, zero disagreements are allowed. The report shows the zone's size (the
  fraction of the input domain inside it) and the disagreements there.
  - **binary:** the zone is inputs whose output is within margin `m` of the threshold.
  - **argmax:** the zone is inputs whose top-2 output gap is below `m`.
  - **regression:** a separate, looser snap tolerance: the program may deviate from the net by
    up to `snap_tol` (default 2% of the net's output range over the domain). `snap_tol` is
    looser than exact mode, so `0.98·a − 1.01·b → a − b` can pass.
  - **How `m` is set:** computed from the net alone, before any refactoring is seen.
    `m = 0.05 × (95th percentile of |output − threshold|)` over the checker's input set (for
    argmax, the same on the top-2 gap). Defaults are project-wide and fixed in advance. A model
    may override them only with a reason written in its spec, and the report lists every
    override. A loose `m` would make "better than the net" free, so this setting is not tuned
    after the fact.

The net's confident answers always constrain a refactoring; its fuzzy edges do not. That is how a
refactoring can be *better* than the net on the true task without the refactorer ever seeing the
target.

Every program is also graded, separately and only in the report, **against the true task**,
including inputs outside the training distribution. This is where "the refactored program beats
the net" would show up.

**Verification labels.** Every verdict states exactly what was done:
- **Exhaustive:** every input of a finite domain, when it has at most ~10⁷ inputs in Python
  (truth tables, 9-bit grids, Blackjack states, 2–3 byte windows). Larger byte windows (2³² inputs
  for 4 bytes) are labeled empirical until step 5. Their empirical set always includes the
  structured cases where errors hide: every single-byte and two-byte change to each fixture, plus
  random and printable-ASCII samples. From step 5 on, 4-byte windows can be checked exhaustively in
  compiled C.
- **Empirical:** seeded random samples, grids, domain corners, points on and just beside every
  activation and decision boundary, and adjacent float values around thresholds. Reported as
  "tested", never "proven".
- **Region-certified (real arithmetic, LP):** added in step 5 with region decoding. It compares
  affine formulas region by region, real arithmetic only, never a floating-point proof. If the
  region budget runs out, the label falls back to "empirical".

Counterexamples are saved and replayed on every later run. A deliberate test shows a rare branch
that random sampling misses.

## 5. Steps

Each step ends with something runnable, results files, and a section in `REPORT.md`.

### Step 1: decompiler + checker (pipeline works)

- Load an MLP (Linear + ReLU, linear output) and its spec. Reject unsupported layers explicitly.
- `dump`: net → `dump.py` in the program language, one line per neuron, with layer/neuron comments.
- Interpreter + NumPy reference. `check` with the labels above.
- **Models** (all small: 1–3 inputs, or up to 9 bits/8 bytes; 1–2 hidden layers; 2–16 neurons):
  - *Handcrafted* (known weights, to test the tool): XOR, max, abs-difference, triangle response,
    clipped ramp, window classifier, near-equality `|a-b| < 0.2`, a net with dead and redundant
    neurons, and two small-weight traps:
    - an 8-input neuron where only 2 weights matter
    - an 8-input neuron whose tiny weights each look like 0 but add up to a real effect
  - *Trained*, seeds 0, 1, 2: XOR, abs-difference, near-equality, window, median of 3,
    tax-bracket piecewise function. Report any that fail to learn.
- Training quality (net vs task) is recorded separately from the checker's program-vs-net results.

**Spike gate:** before going further, run step 2 on 3 of these models (XOR, abs-difference,
near-equality; trained, seed 0). It **passes** if, on at least 2 of the 3, Claude Code's program
passes `check` and is clearly smaller than the dump, and on at least 1 of the 3 it recovers the
expected formula. If it fails, adjust the instruction once (and record that), then rerun. If it
still fails, stop, write it up as a result, and go back to the author before building more.

### Step 2: Claude Code refactoring + heuristics log

**The instruction** (`nndc/agent/instruction.md`, the author's wording, lightly cleaned). It is short
and names *kinds* of refactoring. It never hints at a specific answer: no task names, no `HTTP`,
no Game of Life.

> This code is a dump from a neural network. It roughly resembles some functionality.
> Refactor the code and optimize it. Snap coefficients and thresholds to nearby meaningful values
> (0, ±1, small integers, simple fractions, half-steps between integer inputs, one shared
> threshold for nearby thresholds). Detect constants and fixtures. Try to form formulas and to
> name variables. Remove redundant code and code that doesn't help. Check your version with
> `check`. Afterwards, write down what you learned and any heuristics you used that would apply
> to other networks.

| Refactoring kind | Example |
|---|---|
| Snap | `0.0013·a → 0`, `x > 0.47 → x > 0.5` on bytes, `0.98·a − 1.01·b → a − b` |
| Detect constants | a neuron chain that always outputs `2.0` → `2.0` |
| Detect fixtures | byte compares against 72, 84, 84, 80 → `b"HTTP"` |
| Form formulas | two ReLUs → `abs(a − b)`; nine weights ≈ 1 → `sum(cells)` |
| Name variables | `h3 → a_gt_b`, `h7 → neighbors` |
| Remove | repeated conditions, unreachable branches, unused neurons |
| Small weights judged together | eight weights of 0.02 on byte inputs can shift a neuron by 40, so they're dropped only if their combined effect `Σ|wᵢ|·max|xᵢ|` is negligible, otherwise kept as a sum or count |

**How it runs: the user runs Claude Code, and the tool prepares and collects.** The pipeline
does not launch Claude itself. The repo provides the instruction file and the folders. The person
running the experiment (the author, or any reader with Claude Code) runs Claude Code themselves,
as many times and on as many models as they choose. The number of runs per model is therefore
not fixed by the design. The report uses whatever runs exist, and shows the spread when a model
has more than one.

- **`nndc agent prepare <run>`** creates a **blind folder** with a neutral id (e.g. `m07/`) in
  the system temp directory, outside the repo. It contains only:
  - `dump.py`
  - the blind spec (neutral input names, no task name)
  - `INSTRUCTIONS.md` (copied from `nndc/agent/instruction.md`)
  - the net's weights as `weights.npz` (already visible in the dump, so this leaks nothing)
  - a self-contained `check.py` that uses those weights

  No run folder names, training code, or targets are inside. The command prints the folder path
  and a suggested command to run there, for example
  `claude "Follow INSTRUCTIONS.md"` interactively, or the equivalent `claude -p …` with
  restricted tools.
- **The user runs Claude Code in that folder.** Claude Code writes `agent.py`, `agent_notes.md`
  (changes, reasons, names, what it learned, heuristics), and appends its `check.py` runs to
  `agent_checks.log`.
- **`nndc agent collect <folder>`** copies those files back into the run folder as attempt
  `agent_<n>.*`, records the Claude model/version and date where it can find them, and re-runs
  `check` **independently**. That verdict is the one reported.
- Isolation is **best effort plus audit**, not a sandbox. `collect` scans the session transcript,
  if it can find it, for file access outside the blind folder. Runs that touched outside paths,
  or whose transcript can't be found, are marked as such in the report.
- **`nndc agent prepare --all`** prepares folders for a set of models in one go, for users who
  want to run many. A convenience script that loops `claude -p` over prepared folders may be
  added, but it is optional; the manual flow is the main path.
- Heuristics from each run's notes are appended to the shared **`HEURISTICS.md`**, tagged with
  the run id. Each run starts from the same instruction. Whether later runs should also see
  `HEURISTICS.md` is an experiment in itself, run as a separate, labeled condition (see 5.3).
- Claude Code's outputs are committed, so readers can reproduce every later step without Claude.
  Time and cost are logged per run.
- A **mock agent** writes canned outputs, including deliberately wrong ones, so the pipeline and
  the rejection path are tested without Claude.

### Step 3: AST heuristics tool

- **Held-out split:** heuristics are distilled only from the Claude Code runs on **seeds 0 and 1**
  of the step 1 models. Seed 2 and all step 6 models are never used for distillation. The
  reproduction score is reported separately for distillation models (in-sample) and held-out
  models, and only the held-out number is the headline.
- The builder distills by hand: read `HEURISTICS.md` (distillation runs only) and pick the
  heuristics that recur across runs.
- Implement each one as a Python-AST rewrite, one module per heuristic under `nndc/heuristics/`.
  Each module cites the `HEURISTICS.md` entry it came from and has its own tests. Expected early
  entries: constant folding, near-zero term removal with the combined-effect rule, snapping
  relative to input scale, `relu` pair → `abs`/`max`/`min`, dead-branch removal, and
  sum/count detection.
- `nndc ast <run>` writes `ast.py`, checked by the same checker.
- **Main measure: reproduction.** For each model, does `ast.py` pass the check, how close is it in
  size to `agent.py`, and did it recover the same formula? This gives "X of Y Claude Code
  refactorings reproduced mechanically", along with the list of things only Claude managed (for
  example naming, or recognizing a string).
- The AST tool is not independent of Claude Code (it was distilled from it), and the report says so.

**Optional condition (5.3):** rerun Claude Code with `HEURISTICS.md` available, and compare with
the blind runs.

### Step 4: literature search + comparison with existing methods

First, a literature search to pick the right baselines and check novelty. It should look
specifically for LLM-based decompilation or explanation of neural nets (likely recent and the most
important comparison), rule extraction for speed, and constant snapping.

Then run each method on the same nets and compare:

| Method | Type |
|---|---|
| Decision tree fit to the net's answers (scikit-learn) | TREPAN-style rule extraction |
| Exact ReLU region decomposition (`nndc/regions.py`) | symbolic execution; exact but large |
| Symbolic regression (PySR, or gplearn if Julia is a problem) | formula discovery |
| DeepRED, if a working implementation exists | layer-wise rule extraction |
| Anything the literature search turns up, if runnable | — |
| AST tool / Claude Code | this project |

Where possible, each output is converted into the program language so the same checker grades it.
Where it can't be, the method's own predictions are compared with the net through the same input
sets.

Compared on: fidelity to the net, accuracy on the true task (inside and outside the training
range), size (lines, conditions, nodes), whether the known algorithm was recovered (judged
against a per-model "expected formula", written down before the run), and cost (time, and dollars
for Claude).

### Step 5: C compilation, exact pruning, speed

- `cgen`: any program file → C99 → DLL. Each compiled version exports a **typed** batch entry
  (`uint8_t*` for bytes, ints/bits for booleans, `float*` for continuous inputs) and a **float32**
  batch entry that converts and calls it. Both are benchmarked.
- `prune`: exact cleanup using interval bounds over the input range. Delete neurons that are
  always off, fold always-on neurons into the next layer, drop exact zeros. Writes `pruned.py`;
  its label is "bounded-interval exact (real arithmetic)".
- `regions.py` (also used in step 4): activation-region decoding with a strict region/time
  budget and an LP feasibility check (SciPy/HiGHS); it falls back cleanly when the budget runs out.
- **Benchmarks:** PyTorch eager vs C of `dump.py` (the baseline), `pruned.py`, `ast.py`, and
  `agent.py`. Same machine, float32, `-O2`, no fast-math. Batch 1 and 1024; warm-up; timing inside
  native code; checksum against dead-code elimination; repeated runs with spread; a
  branch-unpredictable input set. Record CPU, compiler, flags, threads. Report operation count,
  source size, compiled size, and compile time too.
- "Fastest" is decided per whole version in v1. Choosing the cheapest form block by block inside
  one program (rule vs lookup vs matrix) is later work.

### Step 6: bigger models

Run steps 1–5 on:
- **Grid/bit logic:** Game of Life cell rule (9 bits; known rule `n == 3 or (alive and n == 2)`),
  tic-tac-toe win detector, box overlap, point in triangle.
- **Bytes and fixtures:** keyword/magic detector over a 4–8 byte window (`HTTP`, `GET `,
  `\x89PNG`), with near-miss negatives (one byte off) so shortcuts show up; character classes
  (digit/hex/letter); several keywords at once.
- **Should stay matrices:** a 3×3 edge-detection kernel, and `x*y` (not piecewise linear).
- **Reinforcement learning** (trained from reward with stable-baselines3, small ReLU MLP policies,
  never by imitating the heuristic): Blackjack vs casino basic strategy, CartPole (expected
  single linear rule), MountainCar vs "push with velocity", and LunarLander vs Gymnasium's built-in
  heuristic if Box2D installs. Atari-RAM is a later stretch goal.

Step 6 minimum: Game of Life, a single keyword, multiple keywords, character class, the edge
kernel, Blackjack, and CartPole.

## 6. Repository layout and commands

```
README.md                headline example: weights → program; prior work; limitations
REPORT.md                generated from all results.json files
HEURISTICS.md            what Claude Code learned, appended per run
pyproject.toml, uv.lock

nndc/                    the tool
  model.py               load MLP + spec
  lang.py                program language: ast allowlist, interpreter
  dump.py                net → program
  check.py               program vs net → verdict, label, counterexamples
  agent/
    instruction.md
    prepare.py           blind folder for the user to run Claude Code in
    collect.py           copy results back, audit, independent re-check
    mock.py
  heuristics/            AST rewrites, one module per heuristic (step 3)
  baselines/             decision tree, symbolic regression, ... (step 4)
  regions.py             region decoding (steps 4–5)
  prune.py, cgen.py, bench.py   (step 5)
  report.py

experiments/             model definitions: handcrafted nets, training tasks, RL
tests/

runs/<model>/seed<k>/    everything about one model in one place
  spec.json  net.pt  train_log.json
  dump.py
  agent.py  agent_notes.md  agent_checks.log
  ast.py
  tree.py / symreg.py / ...   baselines (step 4)
  pruned.py  c/              (step 5)
  counterexamples.json
  results.json
```

Commands (a single `nndc` entry point):

```
nndc run <model> --seed <k>       all automatic steps so far, for one model (no Claude)
nndc agent prepare <run> | --all  blind folder(s) for the user to run Claude Code in
nndc agent collect <folder>       bring Claude Code's results back and re-check them
nndc dump | ast | check | compare | compile | bench <run>
nndc report                       rebuild REPORT.md from runs/
pytest                            all tests, no Claude needed (uses the mock agent)
```

## 7. Deliverables

- Source with locked dependencies; one command for the tests; one command per step to reproduce.
- Committed `runs/`: weights, specs, seeds, every program version, Claude Code notes and check
  logs, counterexamples, `results.json`.
- `HEURISTICS.md`.
- `REPORT.md` with a table per step: model/seed, training quality, check verdict and label, max
  error, disagreements (inside/outside margin), true-task accuracy, size, recovered expected
  formula (yes/no), cost; and from step 5 on, runtimes.
- For at least three models, a full walk-through: raw weights → `dump.py` → `agent.py` →
  `ast.py`, with the evidence.
- A README that leads with the most striking example, credits prior work, and states limitations.

## 8. Left to the builder

- Internal module structure beyond the layout above, and file formats of `spec.json` and
  `results.json`.
- Network sizes and training hyperparameters within the stated ranges.
- The exact suggested `claude` command printed by `prepare` (tool limits, turn budget).
- How the C toolchain is found on each OS.
- Which heuristics make it into the AST tool, and in what order.
- Report styling and plots.

## 9. Calls made by Claude (not explicitly decided by the author)

- Discovery-first order (decompiler → Claude Code → AST tool → state of the art → C → bigger
  models) is the author's. Placing the spike gate after step 1 is my call.
- One program format for all versions; restricted Python parsed with `ast` against an allowlist,
  never `eval`'d.
- **Blind scratch folder** for Claude Code (neutral id, no task name or target reachable).
- The independent re-check after Claude Code, committed outputs, and the mock agent.
- The user runs Claude Code themselves (author's decision), as often as they choose. The
  `prepare`/`collect` split, attempt numbering, and reporting spread across attempts are my calls.
- **Snap acceptance rule:** a refactoring may disagree with the net only inside a declared
  low-confidence margin. This is the call that most affects the headline result.
- Claude Code also seeing `HEURISTICS.md` is treated as a separate, optional condition, not the
  default.
- Baseline picks for step 4 (decision tree, region decomposition, symbolic regression, DeepRED if
  available) are provisional until the literature search.
- "Expected formula" per model, written down before runs, for judging "recovered the algorithm".
- Python + PyTorch; later C99 with clang/MSVC on Windows; float32; `-O2`, no fast-math. Typed
  plus float32 entry points; "fastest" decided per whole version in v1.
- Z3 dropped; verification is exhaustive where the domain is finite, labeled empirical
  otherwise, and region-certified (LP, real arithmetic) once step 5 exists.
- Step 6 minimum set; RL trained from reward; LunarLander dropped if Box2D won't install; Atari
  deferred.
- Design document location: `docs/design/`.
- How the snap margin and the regression `snap_tol` are defined (4.3).
- Neutral input names for the refactorer, with real names substituted only in the report.
- `argmax` output kind, from the start.
- "Exhaustive" capped at ~10⁷ inputs in Python; 4-byte windows stay empirical (with structured
  near-miss cases) until compiled C exists.
- Spike-gate pass criterion.
- Held-out split for the AST tool's reproduction score (distill from seeds 0–1, headline on
  seed 2 and step 6).
- Blind folder in the system temp directory, plus a transcript audit, instead of a full sandbox.
