# Notes

## What I did
- Read `spec.json`: two float inputs in [0, 1], one regression output.
- Eyeballed the first rows of `samples.csv`: when x0 > x1 the output equalled 2*x0 (e.g. 0.2698 -> 0.5396);
  when x0 < x1 it equalled x0 + x1 (e.g. 0.637 + 0.977 = 1.614).
- Unified both cases as `x0 + max(x0, x1)` (equivalently `x0 + x1 + relu(x0 - x1)`).
- Max abs error over all 2000 samples: 1.8e-7 (float32 rounding). `check.py`: PASS (exact),
  max |program - function| = 1.79e-7.

## Final program
`out = x0 + max(x0, x1)`: the first input plus the larger of the two inputs.

## Heuristics
- Before fitting anything, look at a few rows by hand and try simple integer combinations
  (sums, doubles, differences) of the inputs; with float32 output, matches to ~1e-7 are exact.
- If one region of the input looks like one linear formula and another region looks like a different one,
  look for the switching condition among simple comparisons (x0 > x1, x > 0.5, ...), then try to merge the
  pieces with max/min/abs/relu, since a ReLU net is piecewise linear and these are its natural pieces.
- Piecewise-linear pieces that agree on the boundary (here 2*x0 = x0 + x1 at x0 = x1) are a strong hint that
  the whole thing is one max/min/relu expression, not an if/else.
- Check the hypothesis against every sample (max abs error) before running the checker, and print the worst
  row to see where a formula breaks.
