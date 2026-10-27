# Notes

## What I changed and why

- The dump is a 2 -> 8 -> 3 ReLU MLP with an argmax output. With only 2 inputs I skipped
  algebra on the 8 hidden units and worked from the decision map instead. I printed the class
  on a 0.1 grid, then bisected along x0 for each x1 row to find where the class changes.
- The map has three bands ordered along x0 + k*x1 (0 at small values, 1 in the middle, 2 at
  large values). Given the ranges (x0 in [1, 6.9], x1 in [0.1, 2.5]), this looks like iris
  petal length and width, so I named them `length` and `width`.
- Boundary 0/1: x0 + 3.12*x1 = 4.89 for x1 < 0.65 and x0 + 3.25*x1 = 4.97 above that. The kink
  is small, so one line, `length + 3.2*width < 4.9`, covers it.
- Boundary 1/2 has a real kink near x1 = 0.55: x0 + 1.09*x1 = 6.78 above it and
  x0 + 0.70*x1 = 6.58 below it. One line with x0 + x1 or 1.1 failed where the net is confident,
  so I kept two pieces. The class-2 region is convex here, meaning it is the union of the two
  half-planes, so they join with `or`. I first tried `and`, which produced thousands of errors.
- Snapped values: 3.2 / 4.9, 1.1 / 6.8 and 0.7 / 6.6. I also tried simpler snaps.
  `3*x1 < 4.8` failed (12 confident errors) and `x1 > 6.7` for the upper piece failed. For the
  lower piece, 0.5 / 6.55 and 0.75 / 6.6 also passed, so its exact slope barely matters.
- Result: `PASS (snap)`, with 0 confident disagreements and about 4k disagreements inside the
  unsure zone. An exact pass would mean rebuilding the ReLU logits, which defeats the
  simplification.

## Heuristics

- For low-dimensional argmax nets, read the decision boundaries rather than the weights.
  Scan a grid, bisect each row, and fit `x0 + k*x1 = c` per segment (look at how x0 + k*x1
  drifts to find k).
- Watch for slope changes along a boundary. A small change can be merged into one line. A
  large one (here 0.7 vs 1.1) needs two pieces, joined with `or` when the region is convex
  toward it (a union) and with `and` when it is concave (an intersection). Check which by
  comparing each piece's prediction against the measured boundary at both ends.
- Order the tests by class band: return the extreme classes first and fall through to the
  middle class.
- Try coarser snaps one at a time with the checker. Keep the simplest one that has 0
  disagreements where the net is confident. The size of the unsure zone (5% of the 95th
  percentile logit gap) sets how much a boundary may move.
- Input ranges often identify a known dataset (here iris petals), which gives meaningful
  variable names.
