# Notes

## What the network does
`dump.py` is a 2 -> 16 -> 1 ReLU net on x0, x1 in [-3, 3], regression output.
Sampling it on a 7x7 grid showed values that are symmetric under x0 -> -x0, x1 -> -x1 and swapping
x0 and x1, about 3 on the axes at distance 3 and about 4.24 (= 3*sqrt(2)) at the corners. That is the
Euclidean norm. A least-squares fit `a*hypot(x0, x1) + b` gave a = 0.9988, b = 0.003, so the net is
almost exactly `sqrt(x0^2 + x1^2)`.

The 16 ReLU units are a piecewise-linear approximation of the cone: each one is a plane cut at a
different direction around the origin. No unit is worth keeping on its own.

## Versions tried
- `sqrt(x0*x0 + x1*x1)`: FAIL. The worst error, 0.10, is at the origin, where the net does not reach
  0 (net(0,0) = 0.102). A piecewise-linear net cannot form the sharp tip of the cone, so it rounds it off.
- `max(sqrt(x0*x0 + x1*x1), 0.1)`: PASS (snap), max error 0.072.
- `sqrt(x0*x0 + x1*x1 + 0.01)` (final, `agent.py`): PASS (snap), max error 0.047. This is a smoothed
  norm, sqrt(r^2 + eps^2) with eps = 0.1 = net(0,0). The remaining error is the net's own ripple
  (largest near the corner region, about 0.047), not a shape mismatch.

## Heuristics
- Before reading the units, evaluate the net on a coarse grid and look for symmetry (sign flips, swapped
  inputs) and for known values at the edges and corners (3 and 3*sqrt(2) here). That identifies norms,
  products and similar formulas faster than going through the weights.
- Fit candidate closed forms (L2 norm, L1 norm, max-norm, product, ...) as `a*f + b` with least squares
  and compare max errors; the right one has a near 1 and b near 0.
- Smooth or non-linear targets learned by a ReLU net are wrong at their kinks or tips: the net rounds
  off a cone's apex. When the clean formula fails only there, read the net's value at that point
  (here net(0,0) = 0.1) and use it as a smoothing constant, e.g. `sqrt(r^2 + c^2)`, instead of a
  `max`/clamp or a patch with extra cases.
- When several candidates pass, keep the one with the smaller max error if it is no bigger.
