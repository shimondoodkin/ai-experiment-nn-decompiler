# Notes

## What I did
- spec: 3 float inputs in [-3, 3], one regression output.
- A linear least-squares fit gave ~0.4 on each input (symmetric, no intercept) with max error 1.7:
  the function is symmetric in its inputs but not linear. Adding |x| or relu(x) terms did not help.
- The output range matched the input range ([-2.92, 2.91]) and, looking at rows, `out` was always
  close to one of the inputs. Tried order statistics: the median of (x0, x1, x2) matched within 0.04
  on all 2000 samples (within 0.01 on 91%); mean, clipped sum etc. were far off.
- Wrote the median with min/max only: `max(min(x0,x1), min(max(x0,x1), x2))`.
- `check.py`: PASS (snap). Max deviation 0.05, located where two inputs are nearly tied near the
  edge of the box (e.g. x0 ~ x2 ~ -2.9): the function is a smooth-ish approximation of the median
  there. Those residuals are approximation noise, not structure, so I did not model them.

## Heuristics
- Start with a linear fit; equal coefficients on all inputs plus a large residual suggests a
  symmetric non-linear function (sort/order statistic, max/min, norm).
- If the output range equals the input range and each output is close to one of the inputs,
  test order statistics (min, max, median, k-th largest) before fitting piecewise-linear models.
- Measure "fraction of samples within a small tolerance" as well as max error: a candidate that
  hits most samples nearly exactly is right, with the outliers being approximation error.
- Median of three in a min/max-only language: max(min(a,b), min(max(a,b), c)).
- Residual errors clustered where inputs are tied or near the box edges are the network's
  approximation imperfections; do not add terms to chase them once the snap check passes.
