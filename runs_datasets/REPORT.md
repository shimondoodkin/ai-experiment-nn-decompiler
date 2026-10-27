# On Iris, every net reduces to a short decision list of linear tests

This is a side run on a real dataset instead of a known formula. Small ReLU nets were trained to classify the 150 Iris flowers, then decompiled and refactored in two ways: by the deterministic `nndc ast` tool and by blind Claude Code agents.

- **Claude agents:** all 6 nets (2 tasks × 3 seeds) came down to an if-chain of 2–3 linear tests in 32–76 nodes, against 364–484 for the dump. All 6 pass the checker.
- **`ast` tool:** a decision-list route (P10) was added after seed 0, distilled from the two seed-0 agents' notes and frozen before seeds 1–2. It reproduced both seed-0 programs in-sample. On the 4 held-out nets it produced a passing list once (iris seed 2). On the other three it came within 6, 54 and 591 confident disagreements but did not pass, so the tool fell back to the pruned net.
- **Why the tool missed:** the class boundaries on those nets have a kink, or sit near a second activation pattern. The agents wrote a kink as two half-planes joined by `or`, or placed the threshold in the middle of the range the confident points allow. P10 does neither; it folds one activation pattern into one straight line.
- **Not the textbook rule:** the recovered boundaries are slanted lines, not the textbook petal thresholds. The textbook rule agrees with the nets on 95–97% of the flowers but on only 58–70% of the full input box, because the nets extend their boundaries through regions where there are no flowers.

## Setup

- **Data:** the 150-row Iris table, exported from scikit-learn to `experiments/data/iris.csv`. Classes: 0 setosa, 1 versicolor, 2 virginica.
- **Tasks** (`experiments/datasets.py`). Both are kept out of `TASKS`, so `nndc run all`, `REPORT.md` and the step-1 tests do not see them.
  - `iris`: 4 inputs (sepal length, sepal width, petal length, petal width) → 8 hidden → 3 outputs.
  - `iris_petal`: 2 inputs (petal length, petal width) → 8 hidden → 3 outputs.
- **Input ranges** are the table's min and max per column. The checker samples the whole box, not only the flowers.
- **Training** (`experiments/train.py` supports argmax): cross-entropy loss, Adam with lr 1e-2, 3000 steps, batches of 512 rows drawn from the table with replacement, seeds 0–2.
- **Target off the table:** grading needs a label for every point in the box, so the target is the label of the nearest flower (1-NN). That makes the "recovered" column meaningless for these tasks. The numbers that matter are the checker result against the net and the accuracy on the 150 flowers.
- **Blind refactors:** `nndc agent prepare`, then one fresh general-purpose subagent per model, then `nndc agent collect --transcript`. One attempt per model.

## Results

The `ast` and agent columns give the checker verdict and the program size in nodes. "in" means in-sample (seed 0, which P10 was distilled from); "held-out" means seeds 1–2, run once with the frozen tool.

| Run | Net acc. | Dump | `ast` (route) | Agent | Program acc. (ast / agent) |
|---|---|---|---|---|---|
| iris seed0 (in) | 98.0% | exact 484 | snap 62 (decision-list-snap) | snap 76 | 98.7% / 98.7% |
| iris seed1 (held-out) | 98.0% | exact 482 | exact 396 (structural-exact) | **snap 70** | – / 98.0% |
| iris seed2 (held-out) | 98.0% | exact 480 | **snap 62 (decision-list-snap)** | snap 67 | 97.3% / 97.3% |
| iris_petal seed0 (in) | 95.3% | exact 374 | exact 32 (decision-list) | exact 32 | 95.3% / 95.3% |
| iris_petal seed1 (held-out) | 95.3% | exact 366 | exact 322 (structural-exact) | **snap 43** | – / 95.3% |
| iris_petal seed2 (held-out) | 96.0% | exact 364 | snap 285 (structural-snap) | **snap 49** | – / 96.0% |

The accuracy columns are measured on the 150 flowers. iris_petal seed2's net is just below the 0.95 "learned" bar on the training log's 20,000-row quality sample (94.9%), but it is 96.0% on the table itself.

**Held-out score: tool 1/4, Claude 4/4.** This counts a short decision list that passes the checker. The step-3 tool scored 5/6 on the step-1 held-out seeds, so argmax nets are clearly harder for it.

### Seed 0: tool and agent agree (in-sample)

Petal only (x0 = petal length, x1 = petal width). The tool and the agent produced the same exact program:

```python
def net(x0, x1):
    if x0 + 2.05163779247935 * x1 < 4.096047795169456:
        return 0
    if x0 + 1.1003193515404486 * x1 < 6.803922393002628:
        return 1
    return 2
```

The rounded form `x0 + 2*x1 < 4.1 → 0; x0 + 1.1*x1 > 6.8 → 2` passes snap. Exact wins the tie on size, so the tool keeps the long constants, and so did the agent.

All 4 features (x0..x3 = sepal length, sepal width, petal length, petal width). The tool landed on exactly the agent's rounded constants:

```python
def net(x0, x1, x2, x3):
    if x0 + 4 * x1 - 5 * x2 - 4.5 * x3 > 0:
        return 0
    if x0 + 2 * x1 - 2.75 * x2 - 5 * x3 > -10:
        return 1
    return 2
```

### Held-out: what the agents found and the tool did not

- **iris_petal seed1. The boundary has a kink.** Class 0 is the union of two half-planes:

  ```python
  if x0 + 4 * x1 / 3 < 3.5 or x0 + x1 < 3:
      return 0
  if x0 + x1 > 6.65:
      return 2
  return 1
  ```

  The tool's single line `x0 + 1.32*x1 < 3.50` was right on one side of the kink only: 54 confident disagreements.
- **iris_petal seed2. A kink on the other boundary.** Here class 2 is `x0 + 1.1*x1 > 6.75 or x0 > 6.3`. The tool's single line left 591 confident disagreements.
- **iris seed1. No kink, but the threshold was in the wrong place.** The agent labelled 3M points with the net and kept only the confident ones. It then picked simple coefficients and put each threshold in the middle of the range those points allow (0.64–0.75 and 10.09–10.41). The tool folded the most common activation pattern on the boundary, and a second unit (rarely on) tilts the boundary in a corner. The unsnapped fold was 6 confident disagreements short; the snapped versions were worse.

  ```python
  petal_size_0 = x2 + 1.1 * x3 - 0.2 * x0 - 0.6 * x1
  if petal_size_0 < 0.7:
      return 0
  petal_size_2 = 2.75 * x2 + 5 * x3 - x0 - 2 * x1
  if petal_size_2 > 10.25:
      return 2
  return 1
  ```
- **iris seed2. Both passed with lists of similar size.** The tool's list has 62 nodes, the agent's 67, with different but equivalent rounding. The agent first chose threshold 1.35, which left one confident error in a corner, and moved it to 1.3.

## The decision-list route (P10)

The route lives in `nndc/heuristics/decision_list.py`. It was distilled from the iris_petal/seed0 and iris/seed0 notes in `HEURISTICS.md`, plus two step-1 notes:

- xor/seed0: "For binary/argmax outputs only the sign/ordering matters".
- window/seed1: "solve output = 0 piecewise per linear region".

The steps:

1. **Sample and label.** Use the checker's own input set: uniform points, corners, and points on unit and class boundaries. Label each point with the net.
2. **Try every class order.** For each of the 3! orders, build `if test_a: return a; elif test_b: return b; else c`.
3. **Build each test as a binary problem.** "Class a wins" becomes `o_a − o_b > 0`, where b is the class that most often comes second where a wins. Candidate tests come from:
   - fold: `o_a − o_b` folded through the top 3 activation patterns near that boundary;
   - the existing boundary rule (P6) and boundary scan (P6b) on that difference;
   - on bool inputs, the truth-table library (P7).

   The test that best matches the net's labels wins.
4. **Pick the order** whose program matches the net best on the sample.
5. **Write and round.** Each fold test is written with the first input's coefficient 1. The tool also offers 3 rounded versions, from coarse to fine. The checker picks among them with the existing rule: smallest passing program first, exactness breaks ties.

Freeze and reproducibility:

- Unit tests on synthetic nets only: `tests/test_heuristics_decision_list.py`.
- Rerunning the tool on all 21 step-1 runs left every `ast.py` and trace byte-identical. The step-1 tasks have no argmax output, so the route never runs there.
- The tool was frozen at `e62d74b` (`runs_datasets/ast_freeze.txt`) before seeds 1–2 were run.

**Ideas for the next version** (post-freeze, not applied):

- **Kinked boundaries.** When one boundary needs two activation patterns, emit `test1 or test2` (or `and`). Both kinks above were two half-planes.
- **Threshold placement.** After folding, move the threshold to the middle of the range that the confident points allow, as the iris/seed1 agent did, instead of keeping the fold's own constant.
- **Cosmetics.** A coefficient of `0.9999999999999999` is printed instead of 1 after dividing by the leading coefficient; snap it to 1.

## Comparison with the textbook rule

The classic CART tree is `petal_length < 2.45 → setosa; petal_width < 1.75 → versicolor; else virginica`. It is right on 96.0% of the flowers.

| Run | Net agrees with rule (150 flowers) | Net agrees with rule (uniform box) |
|---|---|---|
| iris seed0 / 1 / 2 | 96.7% / 95.3% / 95.3% | 70.4% / 69.9% / 65.7% |
| iris_petal seed0 / 1 / 2 | 95.3% / 95.3% / 97.3% | 59.1% / 62.0% / 57.5% |

On the data, the nets and the textbook rule are nearly the same classifier. Across the box they differ a lot. Iris occupies a thin diagonal band of the box, and the nets draw slanted boundaries that run through the empty corners differently from the axis-aligned tree. The checker compares against the net over the whole box, so the faithful program is the slanted one. Every seed gives different slopes and thresholds: seed 0's petal boundary is `x0 + 2.05*x1 < 4.1`, seed 2's is `x0 + 2.25*x1 < 4.4`. This connects to the "Clock and Pizza" warning in the literature report: equivalence to the net, not a match with an expected rule, is the right success test.

## Caveats

- **One attempt per model.** Each net got one agent attempt and one tool run.
- **Snap, not exact.** Only petal seed 0 reached PASS exact. The other agent programs pass snap, with 0 confident disagreements and all remaining disagreements in the unsure zone (5–6.5% of inputs). The iris seed1 and seed2 agents also reran the checker with sampling seeds 1–5 and still had 0 confident disagreements.
- **Audit flags.**
  - Seed-0 runs: the 4-feature agent read only its own background-command output file; the rest are false positives (`/len`, `/print`). The petal agent wrote helper scripts to the session scratchpad. In its tool calls, the only files read outside its blind folder were those scripts. That scratchpad also held a baseline script naming Iris, and the petal transcript does not mention it.
  - Seeds 1–2: the agents were told to keep scratch files inside their blind folder. Their only flags are false positives (`/len`, `/w[2]`).
  - All agents guessed "Iris" from the input ranges alone.
- **Meaningless "recovered" column.** Grading uses a 1-NN target off the table, so `recovered` is always false, and the in-range "accuracy" (about 79%) measures agreement with nearest-flower labels across the box.
- **Small held-out set.** There are 4 held-out nets from one dataset. The 1/4 result says the route covers the straight-boundary case, not how often that case occurs in general.

## Reproduce

```bash
uv run nndc run iris --seed 0 --root runs_datasets          # also iris_petal, seeds 0-2
uv run nndc ast --all --split all --root runs_datasets
uv run nndc agent prepare runs_datasets/iris_petal/seed0    # then a blind Claude Code run in the printed folder
uv run nndc agent collect <blind folder> --transcript <subagent jsonl>
```
