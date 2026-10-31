# Agent notes

## What I did

- The dump is a 1-input, 8-sigmoid, 1-output regression net on x0 in [-2, 2].
- I evaluated it on a grid. The values matched x0² closely: 4.00→3.95 at x0=-2, 1.01 at -1, 0.004 at 0,
  1.00 at 1, 3.96 at 2. The curve is symmetric, while exp and cosh did not match.
- A least-squares quadratic fit gives 1.0004·x0² + 0.0022·x0 + 0.0027, and the higher-order terms of a
  quartic fit are about 1e-3. The net is a quadratic with coefficients that snap to (1, 0, 0).
- `agent.py` is `return x0 * x0`. The checker says PASS (snap): the max |program − net| is 0.047, the
  tolerance is 0.079, and the largest deviations are at the domain edges (x0 = ±2), where the sigmoid
  sum flattens a little.
- I did not look for an exact version. Reproducing the net within 1e-5 would need all eight sigmoids again,
  which defeats the refactoring.

## Heuristics

- For a 1-input regression net, first tabulate the output on a coarse grid. Compare it with candidates
  (x², |x|, exp, cosh, sin, sqrt) and check its symmetry (f(-x) vs f(x)) before reading any weights.
- Fit low-degree polynomials by least squares. If the coefficients sit near small integers and higher
  terms are tiny, snap them and check.
- A sum of sigmoids that approximates a convex function loses accuracy at the edges of the training box,
  usually by under-shooting. Expect the largest snap residual there, not in the middle.
- Prefer the clean formula when it passes snap with a comfortable margin (here about 60% of the
  tolerance). Unless the net is itself piecewise simple, an exact match costs as many terms as the net.
