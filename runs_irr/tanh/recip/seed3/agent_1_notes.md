# Notes

## What the network computes

The dump is 1 input -> 8 tanh units -> 1 linear output, on x0 in [0.5, 4], regression.
Tabulating it (scratch/look.py) showed a smooth, decreasing, convex curve with
net(1) = 0.994, net(2) = 0.502, net(4) = 0.253: this is the reciprocal 1/x.
The residual net - 1/x is below 0.01 everywhere except at the lower edge, where the net
droops: net(0.5) = 1.964 vs 2.0 (residual -0.036), net(0.51) residual -0.028,
net(0.55) residual -0.005.

## What I changed

- Replaced all 8 tanh units and the output layer with `1 / x0`.
- Plain `1 / x0` failed just barely (max error 0.0357 at x0 = 0.5, snap tolerance 0.0342,
  113 inputs within ~0.005 of the edge).
- Added a cap `min(1 / x0, 1.95)` to follow the edge droop: the net cannot reach the steep
  top of 1/x with saturating units. Result: `PASS (snap)`, max error 0.0256 (near x0 = 0.513,
  where the cap starts). Exact pass is not achievable with a clean formula; the remaining
  difference is the net's own fitting ripple (about ±0.01).
- Alternative I considered: `0.99 / x0` (would likely also pass), rejected because the
  uniform scale is not what the residual shows; the error is local to the edge.

## Heuristics

- For 1-input regression nets, tabulate the output on a grid and compare side by side with
  candidate elementary functions (1/x, log x, sqrt x, exp(-x), x^2); reading off f(1), f(2),
  f(4) often identifies the function immediately (here f(2) ~ f(1)/2, f(4) ~ f(1)/4).
- Compute the residual net - candidate across the domain. If it is small everywhere except at
  one edge, the formula is right and the net is just failing to follow a steep or curved region
  near the domain boundary (saturating tanh/sigmoid units cannot follow 1/x's blow-up).
- Fix edge droop with a clamp (`min`/`max` against a round constant near the net's edge value)
  rather than by distorting the whole formula with fitted coefficients.
- Sum-of-tanh nets fitting a smooth target show a small oscillating residual (ripple) of
  roughly constant size; do not try to reproduce it, it is fit error, not structure.
- Pick the clamp value from the tolerance budget: the error peaks where the clamp starts
  (residual of the net at that point), so choose the constant where the net's residual is
  comfortably inside the tolerance.
