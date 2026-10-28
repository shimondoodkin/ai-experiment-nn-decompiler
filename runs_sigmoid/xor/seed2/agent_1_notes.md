# Notes

## What the network does

Two boolean inputs, binary output (class 1 if score > 0). The domain has only 4 points, so I
evaluated the dump on all of them:

| x0 | x1 | score  | class |
|----|----|--------|-------|
| 0  | 0  | -11.76 | 0     |
| 0  | 1  |  +7.93 | 1     |
| 1  | 0  |  +8.51 | 1     |
| 1  | 1  |  -7.68 | 0     |

This is XOR: class 1 exactly when the inputs differ. Every score is far from the threshold (|score| > 7),
so the decision does not depend on the exact sigmoid weights.

## What I changed

- Replaced the 4 sigmoid units and the output layer with `x0 != x1`.
- Returned +1 / -1 rather than the raw comparison (1/0), so the score stays symmetric around the
  threshold 0 and is never exactly 0.
- Hidden units for the record: h1_2 and h1_3 are near-duplicates (both about sigmoid(1 - 7*(x0+x1)),
  "neither input is on", i.e. NOR), and their output weights are about -3.5 and -3.7, so they act as one
  unit with weight about -7.2. h1_0 and h1_1 are opposing x0-vs-x1 detectors with output weights of
  about -16 and +16. None of this structure needs to survive once the truth table is known.

Result: `PASS (exact)` on all 4 inputs.

## Heuristics

- With all-boolean inputs and a small number of them, enumerate the full truth table first. The
  table is the specification; then name the boolean function (XOR, AND, majority, ...) and write it
  directly.
- For binary outputs, look at the margin |score| at each point. If every point is far from the
  threshold, the exact weights don't matter, and a clean boolean expression will match exactly.
- Return a symmetric score (+1/-1) rather than 0/1 when the threshold is 0 and the decision is
  `score > 0`. It is clearer and avoids a tie at the threshold.
- Hidden units with near-identical incoming weights (within about 10%) are duplicates. Merge them
  by summing their outgoing weights. Here two "NOR" units merged into one.
- Sigmoid units with large weights (|w| > 6) on 0/1 inputs are effectively step functions. Read
  each one as a threshold on a linear combination of the inputs: AND, OR, NOR, or "x0 and not x1".
