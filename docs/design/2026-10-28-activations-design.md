# Activation types: preparation for sigmoid nets

Status: draft, 2026-10-28.

## Why

The user wants the decompiler and the AST tool to handle sigmoid as well as ReLU. A sigmoid feeding a
decision reads as an `if`, because `sigmoid(z) > 0.5` is exactly `z > 0`. A throwaway spike (scratchpad,
not committed) trained sigmoid versions of xor, window, neareq, absdiff, tax and Iris, and tried
replacing each sigmoid unit with simpler code. It found:

- **Sharp sigmoids** (xor, window, neareq after longer training): every unit can become a step
  `1 if z > 0 else 0`, so the program is an if-chain.
- **Soft but nearly straight sigmoids** (Iris petal): the unit can become the line `0.5 + z/4`, which
  folds into the next layer, or that line clamped to 0..1.
- **Real curves** (absdiff, tax regression): some units must stay `sigmoid()`.
- **Rounded corners:** smooth nets round the corners of the intended formula. The sharp formula
  (`abs(x0 - x1)`, the tax brackets) fails the checker against the net, but only near the kinks.

The work is split in two projects, in this order:

- **A (this document):** the code learns that a network can have an activation other than ReLU.
  No new behaviour, no new results.
- **B (a later design):** sigmoid nets, the new AST moves (step, line, clamped line), and the report
  that shows a faithful program and a snapped program side by side.

Splitting them is the user's idea. It lets A be checked on its own terms: nothing that exists today
may change.

## Goals

1. **Every hidden layer of a network has an activation:** `relu` (default), `sigmoid` or `tanh`.
   Hidden layers of one net may differ; the output layer stays linear.
2. **The activation travels through the whole pipeline:**
   - training (`experiments/train.py`, a new optional parameter; default `relu`);
   - `from_torch` / `to_torch` (accept `nn.Sigmoid`, `nn.Tanh`);
   - `weights.npz` save/load;
   - the forward pass and pre-activations (`nndc/model.py`);
   - the dump (`sigmoid(...)` / `tanh(...)` in place of `relu(...)`);
   - the program language (new functions `sigmoid(x)`, `tanh(x)`);
   - the checker: forward pass, float32 rounding bound, boundary points;
   - the symbolic net: parse, evaluate, `to_program`.
3. **Pieces that only work for ReLU refuse non-ReLU nets clearly.** They raise a clear error, or
   their route is skipped. They never compute a wrong answer. See "ReLU-only pieces" below.
4. **Tests** cover a small sigmoid net and a small tanh net end to end:
   - save/load round trip;
   - the dump passes the checker exactly;
   - `from_dump` evaluates the same as the forward pass;
   - `to_program` round trip;
   - every refusal.

## Non-goals

- Training, saving or committing any sigmoid run. That is project B.
- New AST moves for sigmoid units (step, line, clamped line). That is project B.
- A policy for snapping smooth nets to sharp formulas. Project B shows both programs and the user
  decides from that.
- Sigmoid support in `compile_back`, the region decomposition baseline, or the ReLU motif and spline
  routes.
- A CLI flag for choosing the activation. Project B adds it, together with where its runs live.

## Anti-goals (a failure even if the code works)

- **Any change to an existing result.** Every `results.json`, `ast.py` and `ast_trace.json` must stay
  exactly the same.
- **A sigmoid unit silently treated as linear.** Today `Unit.relu == False` means "linear unit", and
  heuristics branch on it. If a sigmoid unit became `relu=False`, those heuristics would treat it as
  a straight line and produce wrong programs that might even pass on some inputs. The refactor must
  replace the boolean with an explicit activation, and check every branch on it.
- **Changing what an existing dump looks like.** ReLU dumps stay byte-identical.

## Constraints

- **Stdlib and NumPy only:** `nndc/model.py`, `nndc/lang.py`, `nndc/check.py` and `nndc/spec.py`
  are copied into blind folders. Sigmoid must be computed in a numerically stable way, with no
  overflow warnings for large `|z|`.
- **Experimental discipline:** the AST tool is frozen for the held-out runs. This refactor touches
  heuristics code, but it must not change behaviour on ReLU nets.
  - Equivalence is shown on the distill split only, in a temporary copy (see Acceptance).
  - Held-out models are not re-run.
  - If the equivalence check finds any difference, the refactor is wrong. A new freeze is not an
    acceptable way out.
- **Commits:** code only; no `data:` commit is expected, because no data changes.

## Decisions

### Where the activation is stored

- **`MLP`:** it gains `acts: list[str]`, one entry per hidden layer, defaulting to all `relu`. Code
  that builds an `MLP` from weights alone keeps working.
- **`weights.npz`:** an optional key `acts`, a string array. When it is missing, every hidden layer is
  `relu`, so all existing files load unchanged. When every layer is `relu`, `save_npz` does not write
  the key, so files written for ReLU nets stay byte-identical.
- **Not in `spec.json`:** the activation is a property of the network, not of the task. The blind spec
  keeps its current fields.

### The symbolic net

- `Unit.relu: bool` is replaced by `Unit.act: str`, one of `relu`, `linear`, `sigmoid`, `tanh`.
  `linear` is what `relu=False` meant before; the argmax output units in `from_dump` are an example.
- `from_dump` recognises `relu(...)`, `sigmoid(...)` and `tanh(...)` around an affine expression, and
  anything else stays an error as today.
- `evaluate`, `substitute`, `to_program` and `unit_bounds` handle all four. Intervals for sigmoid and
  tanh are `(f(lo), f(hi))`, since both are increasing.

### Every site that branches on `u.relu`

Each site is rewritten deliberately, as one of:

- **ReLU-specific logic:** it requires `act == "relu"`.
- **"Not ReLU, so identity":** it requires `act == "linear"`.
- **Activation-agnostic:** for example, interval bounds go through one shared helper.

Known sites:

| Site | What happens with sigmoid/tanh |
|---|---|
| `ast_tool.py` | Units are filtered with `u.relu`; this becomes `act == "relu"`. |
| `heuristics/bounds.py` | Dead/always-on pruning is ReLU-only. Sigmoid/tanh units only get interval bounds. |
| `heuristics/merge.py` | Positive-scaling normalisation is ReLU-only (positive homogeneity). Duplicate merging is safe for any activation when the pre-activations are identical. |
| `heuristics/snap.py` | Uses the shared interval helper. |
| `heuristics/motifs.py` (`as_relu_sum`) | Returns `None` when any unit is sigmoid/tanh, so the motif, spline and boundary routes skip. |
| `heuristics/decision_list.py` | Activation patterns are ReLU-only. The route is skipped for nets with sigmoid/tanh units. Project B adds a black-box reading. |
| `heuristics/fit.py` | Builds candidates from inputs, not units. Unchanged. |

The builder must grep for every other use of `relu` / `np.maximum` on units, and classify each one
the same way.

### Program language

- **New functions:** `sigmoid(x)` and `tanh(x)`, one argument each, vectorised. The `FUNCTIONS` set
  and the language description string (shown to agents) list them.
- **Size statistics:** the keys stay as they are (`lines`, `nodes`, `constants`, `relus`, `ifs`), so
  existing `results.json` files do not change. Sigmoid and tanh calls count as nodes. A separate count
  is left to project B.

### Checker

- **Forward pass:** `net_output` uses the net's own forward pass, so it needs no change beyond `MLP`.
- **Rounding bound:** `_rounding_bound` propagates the error through each activation by its Lipschitz
  constant (ReLU 1, sigmoid 1/4, tanh 1), plus a few ulps for evaluating `exp` / `tanh`. The ReLU path
  must give exactly the same numbers as today.
- **Boundary points:** they already bisect on the sign of the pre-activations. For sigmoid and tanh,
  `z = 0` is also the natural boundary (sigmoid crosses 0.5, tanh crosses 0), so no change is needed.

### ReLU-only pieces

These raise a clear error for a net or program they cannot handle; nothing else changes in them.

- `compile_back`: a program that calls `sigmoid` / `tanh`.
- `baselines/regions.py`: a net with non-ReLU layers. The tree and PySR baselines are black-box and
  work unchanged.
- `repair/weights.py`:
  - fine-tuning and retraining go through `to_torch` and work with any activation;
  - the last-layer LP must take hidden features from the net's own forward pass, not from a
    hard-coded ReLU;
  - the LP builds its repaired net as `MLP(mlp.weights[:-1] + [W], ...)` (around line 125). That
    call must pass `acts` through. Otherwise the `relu` default silently turns a repaired sigmoid net
    into a ReLU net, the same trap as the anti-goal. Any other place that builds an `MLP` from
    another net's layers must do the same; `compile_back` only ever emits ReLU nets, so it is fine.

## Acceptance

1. `uv run pytest -q` passes, including the new tests.
2. **Results unchanged:** before the refactor, re-evaluate every run under `runs/` and `runs_bugs/`
   and record any diff; it should be none. After the refactor, re-evaluate again. The `git diff` of
   `runs/` and `runs_bugs/` must be empty.
3. **AST tool unchanged on ReLU nets:** copy runs that are not held out to a temporary folder and run
   the AST tool there. Every `ast.py` and `ast_trace.json` must be identical to the committed ones.
   The runs:
   - the distill split of the formula runs;
   - Iris and Iris-petal seeds 0–2, which are not held out for tool v2. They are the only argmax
     nets, so they cover the decision-list route and the argmax paths of `from_dump` and
     `to_program`. `--split distill` leaves them out, so they need a direct call or `--split all`
     on a copy that holds only these runs.

   Held-out models (formula seed 2, Iris seeds 3–5) are not run, not even in a throwaway copy, and
   neither are the planted-bug nets in `runs_bugs/`. The runs above cover every code path the
   refactor touches.
4. **Repair outputs unchanged:**
   - Run `nndc repair baselines` and `nndc repair evaluate` into a copy of `runs_bugs/` and the
     natural-bug runs, then diff `repair/results.json` and the `*_net.npz` files against the
     committed ones.
   - First run the same thing twice on the current code. If the fine-tune arms come out differently
     between those two runs (`finetune` does not call `torch.set_num_threads(1)`), they are left out
     of the comparison. The LP arm and the compile-back nets must match exactly.
5. **Dumps unchanged:** re-dumping every existing net gives byte-identical `dump.py` files.
6. **How "identical" is judged:** use `git diff`, or compare with CRLF stripped. On Windows,
   `write_text` writes CRLF, and `nndc ast` rewrites the `ast:` line of `run_git.txt`; both are
   expected in a temporary copy.
7. **A sigmoid smoke test:** train a tiny sigmoid net and a tiny tanh net in a test (seconds, CPU).
   Dump each one and check that the dump passes the checker exactly. Run the AST tool on each and
   check that it returns a passing program, even if that is only the pruned dump.

## Left to the builder

- Names, module layout, how the shared interval helper is organised, test file names.
- How `from_torch` validates layer order (it must still reject two Linear layers in a row, or an
  activation first).
- How re-evaluation in acceptance steps 2–3 is scripted. `evaluate_run` over all runs, plus a temporary
  `--root`, is enough.
