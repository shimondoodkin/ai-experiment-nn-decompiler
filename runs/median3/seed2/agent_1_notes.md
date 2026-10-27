# Notes

## What the network computes
The 3 -> 16 -> 16 -> 1 ReLU net is an approximation of the **median of the three inputs**.
`agent.py` replaces all 32 hidden neurons with

    median(x0, x1, x2) = max(min(x0, x1), min(max(x0, x1), x2))

Check: `PASS (snap)`, max |program - net| = 0.052 against a snap tolerance of 0.12. An exact pass
(1e-5) is not reachable with a clean formula: the residual is the net's own approximation
error (largest near the corners of the domain, e.g. two inputs near -3, where the net is off by ~0.05),
not a missing term.

## How it was found
1. Sampled 200k uniform points in [-3,3]^3 and evaluated the reference float32 forward pass.
2. Linear fit: coefficients ~0.4 on each input, equal -> output is a symmetric function of the inputs,
   but residual was large (1.77), so it is piecewise.
3. Fit on a library of symmetric piecewise-linear features (|xi|, |xi-xj|, pairwise max, max/min of
   all three, products, squares). Products/squares/|xi| got ~0 weight; pairwise |xi-xj|, pairwise
   max, global max and min carried everything -> an order statistic.
4. Tested median directly: slope 0.998, intercept -0.001, max error 0.05 -> median, coefficients
   snapped to 1 and 0.

## Heuristics
- Start with a least-squares fit on raw inputs; equal coefficients on all inputs signal a symmetric
  function (sorting / order statistics, sum, mean, norm).
- Build a feature library of candidate piecewise-linear primitives (abs, pairwise max/min, |xi-xj|,
  global max/min) plus a few nonlinear ones (products, squares). Features that get ~0 weight can be
  eliminated; the surviving set names the function family.
- Order statistics of n inputs: linear coefficient ~ 1/n-ish on each, and the fit concentrates on
  max/min/|xi-xj| features. Median of 3 = sum - max - min = max(min(a,b), min(max(a,b),c)).
- Once a candidate is guessed, regress the net's output on it (slope, intercept): slope ~1 and
  intercept ~0 confirm it and give the snapped coefficients.
- Look at the worst residuals: if they sit at the domain corners / edges (where training data is sparse)
  and are small relative to the output range, they are approximation error of the net, not structure
  worth reproducing. Don't chase exact pass by encoding that error.
- Read the checker's tolerances first (here snap tol = 2% of output range) to know how much
  approximation error is acceptable.
