# Notes

## What I did
- Spec: 3 float inputs in [-3, 3], one regression output.
- A global linear fit gave coefficients ~0.40 / 0.41 / 0.39 on each input (symmetric, each ~1/3 plus
  something) with large max error (1.67), so the function is symmetric but non-linear.
- Correlation with max(x) and min(x) were both ~0.56 (equal), suggesting an order statistic in the
  middle: the median. `out - median(x0,x1,x2)` had max |err| 0.055, mean 0.017.
- A fit `out ~ a*median + b` gave a = 0.998, b = -0.016: snapped to a = 1, b = 0.
- Program: `median = x0 + x1 + x2 - min(...) - max(...)` (expressible without sorting).
- `check.py` -> PASS (snap), max |program - net| = 0.109 (tolerance 0.120), the largest gaps at
  corners of the box (|x| near 3) where the net under-shoots; this is approximation error of the
  net, not structure, so I did not model it.

## Heuristics
- Start with a global linear least-squares fit: equal coefficients on all inputs signal a symmetric
  function (sum/mean, or a symmetric order statistic like min/max/median).
- Correlate the output with candidate symmetric features (max, min, median, sum, sum of |x|);
  equal correlation with max and min points to the median.
- Median of three without sorting: `a + b + c - min(a,b,c) - max(a,b,c)`
  (or `max(min(a,b), min(max(a,b), c))`).
- After finding the candidate, regress the output on it (`a*f + b`) and snap slope/offset to 1/0
  when they are within a few percent; residuals concentrated at the box corners are training
  artifacts of the net, not extra terms.
