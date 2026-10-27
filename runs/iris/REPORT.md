# On Iris, every net reduces to a short decision list of linear tests

This is a side run on a real dataset instead of a known formula. Small ReLU nets were trained to classify the 150 Iris flowers, then decompiled and refactored in two ways: by the deterministic `nndc ast` tool and by blind Claude Code agents.

- **Claude agents:** all 6 nets of seeds 0–2 (2 tasks × 3 seeds) came down to an if-chain of 2–3 linear tests in 32–76 nodes, against 364–484 for the dump. All 6 pass the checker.
- **`ast` tool, v1:** a decision-list route (P10) was added after seed 0, distilled from the two seed-0 agents' notes and frozen before seeds 1–2. It reproduced both seed-0 programs in-sample. On the 4 held-out nets it produced a passing list once (iris seed 2). On the other three it came within 6, 54 and 591 confident disagreements but did not pass.
- **Why v1 missed:** the class boundaries on those nets have a kink, or sit near a second activation pattern. The agents wrote a kink as two half-planes joined by `or`, or placed the threshold in the middle of the range the confident points allow. v1 did neither; it folded one activation pattern into one straight line.
- **`ast` tool, v2:** it adds both agent moves: two half-planes joined by `or`/`and`, and thresholds centred in the confident range or snapped to the simplest value inside it. Seeds 0–2 are in-sample for v2, and it passes on all 6, including the 3 that v1 missed. It was frozen before a fresh held-out set (seeds 3–5) was trained. **On those 6 new nets it produced a passing 2-test decision list every time (6/6)**, in 29–64 nodes against 368–484 for the dump. All 6 also pass the checker with 4 other sampling seeds.
- **Tool v2 against Claude on seeds 3–5:** blind Claude agents also passed all 6 nets.
  - **Size:** the tool's programs are the same size or smaller on 5 of 6 (29–64 nodes, against 29–115 for Claude).
  - **Robustness:** the tool passes the extra checker seeds on all 6. One Claude program fails one point at one extra seed, as its agent noted.
  - **Readability:** Claude's constants are rounder. The tool kept long decimals on 2 of 6 nets.
  - **Time:** about 7–12 seconds per net for the tool, against 2–10 minutes for an agent.
- **Not the textbook rule:** the recovered boundaries are slanted lines, not the textbook petal thresholds. The textbook rule agrees with the nets on 95–97% of the flowers but on only 58–70% of the full input box, because the nets extend their boundaries through regions where there are no flowers.

## Setup

- **Data:** the 150-row Iris table, exported from scikit-learn to `experiments/data/iris.csv`. Classes: 0 setosa, 1 versicolor, 2 virginica.
- **Tasks** (`experiments/datasets.py`). Both are kept out of `TASKS`, so the step-1 tests and the step-3 reproduction numbers do not see them. Their runs are `runs/iris/seed0..5` and `runs/iris_petal/seed0..5`, and this report covers both.
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
- The tool was frozen at `e62d74b` (freeze commit `84aa4ab`) before seeds 1–2 were run. The v1 `ast.py` files were later overwritten by v2, so the runs' `run_git.txt` now show v2; v1's outputs are in commit `a0d15d3`.

## P10 v2 and a second held-out set (seeds 3–5)

v2 adds three things the v1 held-out run showed were missing. The agents' seed 1–2 notes are its source, so seeds 0–2 are in-sample for v2.

1. **Kinked boundaries.** Up to 4 distinct fold half-planes are kept per test. Pairs of the first 3 are also tried as `h1 or h2` and `h1 and h2`.
2. **Threshold placement.**
   - The tool takes its own sample: the checker's kind of input set, but with seed 1, while the checker uses seed 0.
   - On the confident points of that sample, it computes the range of thresholds that classifies every point right. For an `or`, each point of the class is assigned to the half-plane it lies deeper inside.
   - The "centred" variant uses the middle of that range. The rounded variants first round the coefficients, then take the simplest number inside the recomputed range.
   - This is what the iris/seed1 agent did: 0.7 inside 0.64–0.75, and 10.25 inside 10.09–10.41.
3. **Leading coefficient exactly 1**, instead of `0.9999999999999999`.

Within a step, candidates are ranked by agreement on the confident sample points (thresholds centred), then by node count. v1 used character count, which penalised long decimals and lost petal seed 0's exact program in a first try. The checker still decides among the finished programs with the unchanged rule. v2 was frozen at `363b8c9` (freeze commit `b400702`; the held-out runs' `run_git.txt` record `ast: b400702`, which has the same tool code), and the step-1 runs were again byte-identical.

**In-sample (seeds 0–2), v2:** 6/6 pass.

| Run | Nodes | Verdict |
|---|---|---|
| iris seed 0 | 62 | snap |
| iris seed 1 | 64 | snap |
| iris seed 2 | 62 | snap |
| iris_petal seed 0 | 32 | exact |
| iris_petal seed 1 | 32 | snap |
| iris_petal seed 2 | 46 | snap |

- Petal seed 1 now passes with one centred line, `x0 + 1.3*x1 < 3.52`, where v1 missed by 54. The agent had written an `or` of two lines and reported that slopes 1.2, 1.25 and 4/3 failed.
- Petal seed 2 needs the `or`, `x0 + 1.09*x1 > 6.76 or x0 + 0.37*x1 > 6.37`. That is the same two half-planes the agent found (`x0 + 1.1*x1 > 6.75 or x0 > 6.3`).

**Held-out (seeds 3–5), frozen v2:** fresh nets trained after the freeze; learned accuracy 96.6–98.6%.

| Run | Net acc. (table) | Dump | `ast` v2 | Program acc. (table) | Confident disagreements, checker seeds 0/2/3/4/5 |
|---|---|---|---|---|---|
| iris seed3 | 98.0% | 484 | **exact 62** | 98.0% | 0/0/0/0/0 |
| iris seed4 | 98.7% | 476 | **snap 64** | 98.7% | 0/0/0/0/0 |
| iris seed5 | 98.0% | 478 | **snap 62** | 98.0% | 0/0/0/0/0 |
| iris_petal seed3 | 95.3% | 372 | **snap 32** | 96.7% | 0/0/0/0/0 |
| iris_petal seed4 | 95.3% | 370 | **snap 29** | 96.0% | 0/0/0/0/0 |
| iris_petal seed5 | 95.3% | 368 | **snap 32** | 95.3% | 0/0/0/0/0 |

Examples of the held-out programs:

```python
# iris_petal seed4 (x0 = petal length, x1 = petal width)
if x0 + x1 < 3.25:
    return 0
if x0 + 1.125 * x1 < 6.8:
    return 1
return 2

# iris seed4 (x0..x3 = sepal length, sepal width, petal length, petal width)
if x0 + 9 * x1 - 19 * x2 - 30 * x3 > -40:
    return 0
if x0 + 1.8 * x1 - 2.5 * x2 - 4.5 * x3 > -8.5:
    return 1
return 2
```

What the second held-out set does and does not show:

- **The improvement came from centring.** Every held-out winner is a single line per boundary with a centred or rounded threshold. No held-out net needed the `or`/`and` test, so the kink handling is tested only in-sample (petal seed 2) and on a synthetic test net.
- **Rounding is uneven.** iris seed 3 kept its long fold constants: it passes exact, and its rounded variants failed. Petal seed 3 and iris seed 5's first test kept unrounded or lightly rounded coefficients. The agents' programs are rounder.
- **Programs are sometimes right on more flowers than the net**: 96.7% against 95.3% on petal seed 3, and 96.0% against 95.3% on petal seed 4. They agree with the net wherever it is confident; the difference is in the unsure zone.
### Tool v2 against blind Claude agents on seeds 3–5

After the tool run, one fresh blind subagent per net refactored the same 6 nets, one attempt each. The agents never saw the tool's output.

Column notes:
- **Verdict:** "ex" is PASS exact, "sn" is PASS snap.
- **Long consts:** constants with more than 4 significant digits.
- **Extra seeds:** confident disagreements when the checker is rerun with sampling seeds 2/3/4/5. Seed 0 is the checker's default and seed 1 is the tool's own sample, so neither is independent.

| Run | Tool: verdict, nodes | Tool: long consts | Tool: extra seeds | Claude: verdict, nodes | Claude: long consts | Claude: extra seeds | Flower acc. (tool / Claude) | Claude time |
|---|---|---|---|---|---|---|---|---|
| iris seed3 | ex, **62** | 8 | 0/0/0/0 | sn, 72 | 0 | 0/0/0/0 | 98.0% / 98.7% | 3.3 min |
| iris seed4 | sn, **64** | 0 | 0/0/0/0 | sn, 72 | 0 | **1**/0/0/0 | 98.7% / 98.7% | 9.9 min |
| iris seed5 | sn, **62** | 0 | 0/0/0/0 | sn, 115 | 0 | 0/0/0/0 | 98.0% / 98.0% | 8.8 min |
| iris_petal seed3 | sn, **32** | 4 | 0/0/0/0 | sn, 56 | 0 | 0/0/0/0 | 96.7% / 95.3% | 2.6 min |
| iris_petal seed4 | sn, 29 | 0 | 0/0/0/0 | sn, 29 | 0 | 0/0/0/0 | 96.0% / 96.7% | 1.7 min |
| iris_petal seed5 | sn, 32 | 0 | 0/0/0/0 | sn, **29** | 0 | 0/0/0/0 | 95.3% / 95.3% | 2.3 min |

The tool took 7–12 seconds per net on this laptop.

**Reading:**
- **Both pass everything.** Both solved 6/6 against the checker's default seed. This is the first held-out set where the tool matches Claude's success rate; on seeds 1–2, v1 managed 1/4.
- **The tool is smaller or equal on 5 of 6.**
  - It sticks to one straight line per boundary when a centred threshold is enough.
  - The agents sometimes model a small bend that the unsure zone already absorbs:
    - petal seed3: `x0 + 1.1*x1 > 6.8 or x0 + 0.7*x1 > 6.6`;
    - iris seed5: an `and` of two lines for class 2, plus an extra `0.05*x0` term.
  - These additions are not wrong. The iris seed5 agent kept its extra term because the simpler test failed one of its own pooled seeds. But they cost nodes.
- **Claude's constants are always round.** The tool kept long fold constants on iris seed3 (8 of them, but PASS exact) and partly on petal seed3. The agents searched coefficients and threshold together, for example the iris seed4 agent used a linear program to find each coefficient's allowed range. That gives programs like `8*x2 + 14*x3 - 3*x0 - 6*x1 > 27`. v2 rounds coefficients on a fixed ladder first and only then fits the threshold, so when rounding fails it falls back to raw constants.
- **Robustness.** The iris seed4 Claude program fails one far-corner point at checker seed 2. Its agent reported this itself: its threshold window was only 26.7–27.1, and it kept 27. None of the tool's held-out programs fail an extra seed. Centring the threshold on its own independent sample seems to buy some margin.
- **Same structure in both.** Every pair has the same class order and the same two boundaries, and their slopes are close. On petal seed4, for example, the tool has `x0 + x1 < 3.25; x0 + 1.125*x1 < 6.8` and Claude has `x0 + x1 < 3.3; x0 + 1.1*x1 < 6.75`.
- **Audit.** No agent read outside its blind folder. The flags are false positives (`/len`, `/x2coef`, `/np.linalg.norm`, `/nrm`) and one agent's `../spec.json`-style paths from a `work/` subfolder inside its own blind folder.

**Remaining idea:** search simple coefficients jointly with the threshold range, as the agents' grid searches and linear program do, so that more programs end up with round constants. That is the one area where Claude still clearly leads.

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
- **Small held-out sets.** Both held-out sets come from one dataset: 4 nets for v1 and 6 for v2. They show that the route works on these Iris nets, not how it does on other data.

## Step 4: comparison and fix-and-reinsert

The full tables are in the "Comparison (step 4)" section of the top-level `REPORT.md`.

**Baselines on the 12 Iris nets:**
- **Pass the checker:**
  - exact region decomposition: 12 of 12, with 337–2,499 nodes;
  - AST tool v2: 12 of 12, with 29–64 nodes;
  - Claude from input/output samples only: 11 valid runs, all passing (the iris/seed0 run is excluded, see below);
  - PySR: 2 of 12;
  - a decision tree: 0 of 12, even at 512 leaves.
- **Snapping on PySR:** PySR+snap passed none; snapping its long constants moved the slanted boundaries too far.
- **Claude from input/output only:**
  - It did no worse than Claude reading the dump: every valid run passes.
  - It called the checker a median of 14 times per net, against 4 for the dump agents.
  - Several agents first fitted a small ReLU net to the samples, or used RANSAC to find the linear pieces, before writing the decision list.
  - The iris/seed0 agent ran `ls _checker` with the output discarded. That counts as forbidden-file access, so the run is excluded.

**Fix and reinsert (experiment R).** A blind agent got the decompiled program and 100 labelled flowers (a stratified split, seed 0), was asked to fix the program, and was then judged on the other 50.
- **Repair set:** it fixed 60–100% of the net's failures there, and every fixed program compiled back into a ReLU net that matches it exactly.
- **Held-out flowers:** at most one held-out flower is a failure of the net, and no arm fixed it. On the repeated 2-fold splits, the non-agent arms fix 0–33% of held-out failures (median over nets): tree 0.33, fine-tune 0.25, last-layer LP 0.15.
- **Overfitting:**
  - The four-feature fixes that reached 100% of the repair set did it with a small zone rule around 3 overlapping flowers. On iris/seed4 that rule broke one held-out flower (drawdown 0.02).
  - The petal-only agents instead named the duplicated flower with conflicting labels as noise and stopped at 97–98%.
  - The special-case flag misses these zone rules, because it scores each comparison alone, and each one has 8 or more supporting flowers.
- **Prior knowledge:** one agent (iris/seed3) said it checked its rule against Iris flowers it recalled from the real dataset.
- **Reading:** on Iris the remaining errors are class overlap, not a bug in the net. The fix experiment's clean results are on the planted-bug formula nets, where the agent recovered the true formula every time.

## Reproduce

```bash
uv run nndc run iris --seed 0                    # also iris_petal, seeds 0-5 (or `nndc run all`)
uv run nndc ast runs/iris/seed0                  # or `nndc ast --all --split all` for every run
uv run nndc agent prepare runs/iris_petal/seed0  # then a blind Claude Code run in the printed folder
uv run nndc agent collect <blind folder> --transcript <subagent jsonl>
```
