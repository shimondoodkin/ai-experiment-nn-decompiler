# Notes

## What the function is
Four inputs whose ranges match the Iris dataset (sepal length, sepal width, petal length,
petal width) and three outputs with argmax decision: 0 = setosa, 1 = versicolor, 2 = virginica.
Class counts in the samples: 1156 / 482 / 362 for classes 1 / 2 / 0.

## How I got there
1. A global linear fit of each output leaves residuals of ~5, so the outputs are not linear.
   A ReLU network is piecewise linear, so I ran RANSAC over all three outputs together to find
   linear regions. Five regions explain 1991 of 2000 samples (777, 592, 563, 37, 22 points).
2. Only one region contains both class 1 and class 2, and in that region `out1 - out2` is
   exactly linear (residual 0): 3.0276 x0 + 5.9333 x1 - 8.4756 x2 - 15.4841 x3 + 30.6979.
   This is the versicolor/virginica boundary. Class 2 never appears in the setosa regions, so
   checking this plane first is safe.
3. Class 0 vs class 1 is split across two linear regions (a hidden unit switches between them),
   giving two planes for `out0 - out1`. Setosa = both planes positive, i.e. `min(a, b) > 0`.
   This reproduced all 2000 samples and gives `PASS (exact)` (kept as `agent_exact.py`).
4. Snapping: I searched over scale factors and rounded the coefficients to grids of 1, 0.5,
   0.25 and 0.1, scored each candidate on the samples, and then ran the best ones through
   `check.py`.
   - Versicolor/virginica: `3 x0 + 6 x1 - 8.5 x2 - 15.5 x3 + 30.75` (the intercept 31 also passes).
   - Setosa: one plane, `-x0 + 4.2 x1 - 4.3 x2 - 6.7 x3 + 9.3`, which replaces the min of two.
   - Coarser versions failed: integer coefficients (`x0 + 2x1 - 3x2 - 5x3 + 10`) and
     `-x0 + 4x1 - 4x2 - 6.5x3 + 9` both disagree where the function is confident.
5. `agent.py` (the snapped version, with named variables) gives `PASS (snap)`: 0 disagreements
   where the function is confident and 758 in the unsure zone. Those are all thin-petal points
   near the setosa/versicolor boundary.

## Heuristics
- First fit one global linear model per output. A large residual means the function is
  piecewise linear (ReLU), and then you should look for its linear regions.
- Use RANSAC on all outputs jointly to split the samples into linear regions. A few large
  regions mean only a few hidden units switch inside the data range.
- For argmax outputs, model the pairwise differences `out_i - out_j` rather than the raw
  outputs, and only where both classes actually compete. Near the boundary those differences
  are often exactly linear, even when the outputs as a whole are not.
- For each pair of classes, check which classes can co-occur. If two classes never meet
  (setosa vs virginica here), the program can be a short if-chain.
- When a boundary is bent by a ReLU switch, write it as `min` of two planes (or `max`,
  depending on whether the region is convex). Try both against the samples.
- Snap by trying many scale factors with a rounding grid, not by rounding the raw
  coefficients: a plane can be rescaled freely, so pick a scale where all coefficients land
  near a grid point. Rank candidates on the samples first (cheap), then confirm with the checker.
- Keep the exact (unsnapped) version as a fallback, and report both the exact and the
  snapped results.
- Check which side the snapped program errs on: here the errors cluster where the function's
  margin is small (thin petals), which is acceptable for `PASS (snap)`.
