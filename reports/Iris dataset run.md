# On Iris, an 8-unit net reduces to two straight-line tests

This is a side run on a real dataset instead of a known formula. Small ReLU nets were trained to classify the 150 Iris flowers, decompiled, and refactored, first by the deterministic `nndc ast` tool and then by a blind Claude Code agent. On seed 0 the agent reduced both nets to **two linear inequalities in a decision list**, and both programs pass the checker against the net. The petal-only program passes exact at 32 nodes, down from the 374-node dump. The 4-feature program passes snap at 76 nodes, down from 484. The `ast` tool has no route for 3-class (argmax) outputs, so it only pruned and snapped. The recovered boundaries are slanted lines, not the textbook petal thresholds. The textbook rule agrees with the nets on 95–97% of the flowers but on only 58–70% of the full input box, because the nets extend their boundaries through regions where there are no flowers.

## Setup

- **Data:** the 150-row Iris table, exported from scikit-learn to `experiments/data/iris.csv`. Classes: 0 setosa, 1 versicolor, 2 virginica.
- **Tasks** (`experiments/datasets.py`). Both are kept out of `TASKS`, so `nndc run all`, `REPORT.md` and the step-1 tests do not see them.
  - `iris`: 4 inputs (sepal length, sepal width, petal length, petal width) → 8 hidden → 3 outputs.
  - `iris_petal`: 2 inputs (petal length, petal width) → 8 hidden → 3 outputs.
- **Input ranges** are the table's min and max per column. The checker samples the whole box, not only the flowers.
- **Training** (`experiments/train.py` now supports argmax): cross-entropy loss, Adam with lr 1e-2, 3000 steps, batches of 512 rows drawn from the table with replacement, seeds 0–2.
- **Target off the table:** grading needs a label for every point in the box, so the target is the label of the nearest flower (1-NN). That makes the "recovered" column meaningless for these tasks. The numbers that matter are the checker result against the net and the accuracy on the 150 flowers.
- **Runs:** `runs_datasets/<task>/seed<k>/`, built with `nndc run <task> --seed k --root runs_datasets`.
- **Blind refactors:** `nndc agent prepare` followed by one fresh general-purpose subagent per model, then `nndc agent collect --transcript`. Only seed 0 of each task was refactored.

## Results

| Run | Net accuracy (table) | Dump | `ast` | Claude agent |
|---|---|---|---|---|
| iris seed0 | 98.0% | exact, 484 nodes | exact, 252 nodes (structural-exact) | **snap, 76 nodes** |
| iris seed1 | 98.0% | exact, 482 | exact, 396 (structural-exact) | not run |
| iris seed2 | 98.0% | exact, 480 | exact, 308 (structural-exact) | not run |
| iris_petal seed0 | 96.6% | exact, 374 | exact, 240 (structural-exact) | **exact, 32 nodes** |
| iris_petal seed1 | 96.6% | exact, 366 | exact, 322 (structural-exact) | not run |
| iris_petal seed2 | 94.9% (below the 0.95 "learned" bar) | exact, 364 | snap, 285 (structural-snap) | not run |

The accuracies come from the training log's quality sample, which is 20,000 rows drawn from the table.

### Petal-only program (seed 0, PASS exact)

x0 is petal length and x1 is petal width. The agent saw only the neutral names.

```python
def net(x0, x1):
    if x0 + 2.0516378 * x1 <= 4.0960478:
        return 0
    if x0 + 1.1003194 * x1 > 6.8039224:
        return 2
    return 1
```

The snapped form `x0 + 2*x1 < 4.1 → 0; x0 + 1.1*x1 > 6.8 → 2` passes snap. Rounding 1.1 to 1 fails, with 374 confident errors near x1 = 2.5. This program is right on 95.3% of the 150 flowers, the same as the net.

### 4-feature program (seed 0, PASS snap)

x0..x3 are sepal length, sepal width, petal length and petal width.

```python
def net(x0, x1, x2, x3):
    small_petal = x0 + 4 * x1 - 5 * x2 - 4.5 * x3        # > 0 -> class 0
    if small_petal > 0:
        return 0
    narrow_petal = 4 * x0 + 8 * x1 - 11 * x2 - 20 * x3 + 40   # >= 0 -> class 1, else class 2
    if narrow_petal >= 0:
        return 1
    return 2
```

This program is right on **98.7%** of the 150 flowers, slightly better than the net's 98.0%. The unsnapped boundaries were `x0 + 4.11x1 − 5.10x2 − 4.68x3 − 0.33` and `x0 + 1.98x1 − 2.75x2 − 5.06x3 + 10.03`. The second needed quarter steps, because half steps failed the checker.

## How the agents got there

The full notes are in each run's `agent_1_notes.md`, and the agents' heuristics were appended to `HEURISTICS.md`.

1. **Prune.** Some hidden units never fire inside the box: 3 of 8 in the petal net, 4 of 8 in the 4-feature net. Some are always on (or nearly always, about 97% of the box in the 4-feature net) and become linear.
2. **Read argmax as a decision list.** Classes 0 and 2 never touch, so there are only two boundaries: class 0 against the rest, then class 1 against class 2.
3. **Remove the remaining ReLU.** In the 4-feature net, class 0 only occurs where the one surviving ReLU is on, and class 2 only where it is off. Each boundary therefore lies in a region where the net is purely linear and becomes one inequality.
4. **Fit and snap.** For the petal net, the agent bisected the boundary along one axis and fitted a line, which had zero residual. Then came a grid search over simple coefficients, keeping the simplest version that the checker accepted.

Both agents guessed from the input ranges alone that the data was Iris.

## Comparison with the textbook rule

The classic CART tree is `petal_length < 2.45 → setosa; petal_width < 1.75 → versicolor; else virginica`. It is right on 96.0% of the flowers.

| Run | Net agrees with rule (150 flowers) | Net agrees with rule (uniform box) |
|---|---|---|
| iris seed0 / 1 / 2 | 96.7% / 95.3% / 95.3% | 70.4% / 69.9% / 65.7% |
| iris_petal seed0 / 1 / 2 | 95.3% / 95.3% / 97.3% | 59.1% / 62.0% / 57.5% |

On the data, the nets and the textbook rule are nearly the same classifier. Across the box they differ a lot. Iris occupies a thin diagonal band of the (length, width) box, and the nets draw slanted boundaries that run through the empty corners differently from the axis-aligned tree. The checker compares against the net over the whole box, so the faithful program is the slanted one. Restating the textbook rule would have failed. This connects to the "Clock and Pizza" warning in the literature report: equivalence to the net, not a match with an expected rule, is the right success test.

## Caveats

- **One refactor per task.** Only seed 0 was refactored by an agent. Seeds 1 and 2 have no agent attempt, so it is unknown whether every seed collapses to two lines.
- **Snap, not exact, for 4 features.** The 4-feature program passes snap, not exact: 7,092 disagreements, all in the unsure zone. Checker reruns with sampling seeds 1–3 also found 0 confident disagreements.
- **Audit flags.** The transcript audit flagged paths outside the blind folders: helper scripts the petal agent wrote to the session scratchpad, a background-command output file of the 4-feature agent, and two false positives (`/print`, `/len`). In the petal agent's tool calls, the only files read outside its blind folder were its own scratch scripts. Both agents ran as subagents of a session whose scratchpad held a baseline script naming Iris, and the petal transcript does not mention it.
- **Meaningless "recovered" column.** Grading uses a 1-NN target off the table, so `recovered` is always false and the in-range "accuracy" (about 79%) measures agreement with nearest-flower labels across the box.
- **Undertrained petal net.** iris_petal seed2 is just below the 0.95 accuracy bar, so `learned=False`.
- **`ast` limits.** The `ast` tool is the step-3 frozen version, so this is a fair held-out test for it. It has no argmax-specific route; spline, boundary, truth-table and fit apply only to regression and binary outputs.

## Reproduce

```bash
uv run nndc run iris --seed 0 --root runs_datasets          # also iris_petal, seeds 0-2
uv run nndc ast --all --split all --root runs_datasets
uv run nndc agent prepare runs_datasets/iris_petal/seed0    # then a blind Claude Code run in the printed folder
uv run nndc agent collect <blind folder> --transcript <subagent jsonl>
```
