# Notes

## What I did
- The dump is 1 input -> 8 tanh units -> 1 linear output. Six of the eight units have tiny slopes
  (|w| ~ 0.07-0.12) with biases near ±0.85, so on [-1, 3] they sit in the gentle part of tanh and
  together act as a smooth curve; units 4 and 7 have larger slopes and bend the curve near the left end.
- Sampling the net on [-1, 3] showed a positive, monotonically decreasing, convex curve
  (4.92 at -1, 3.00 at 0, 1.11 at 2, 0.66 at 3), i.e. it looks like exponential decay.
- A straight-line fit of log(y) against x gave slope -0.4997 and intercept exp(1.0987) = 3.0004, so
  the snapped formula is y = 3 * exp(-x0 / 2).
- `agent.py`: `return 3 * exp(-x0 / 2)`. Result: PASS (snap), max |program - net| = 0.027 against a
  snap tolerance of 0.085. The largest deviation is at the domain edge x0 = -1 (net 4.920 vs 4.946),
  where the net under-shoots the exponential. An exact pass would need the full tanh sum; a free fit
  a*exp(-k x)+c only lowers the max error to 0.015 with no nice coefficients, so the clean formula
  was kept.

## Heuristics
- For 1-input regression nets, sample the net on a grid first and look at the shape (monotone?
  convex? sign?) before touching individual units; the sum of many small-slope tanh/sigmoid units is
  usually just a smooth textbook function.
- If the output is positive and convex-decreasing (or increasing), fit a line to log(y): a straight
  line means a*exp(b*x); snap a and b (here 3.0004 -> 3, -0.4997 -> -1/2).
- Try other linearising transforms the same way: log(y) vs log(x) for powers, y vs 1/x for
  reciprocals, y^2 vs x for square roots. A near-perfect straight line with nice coefficients is the
  formula.
- Tanh units with |slope| * input-range well below 1 are nearly linear over the box; many such units
  are a sign that the net is approximating a smooth nonlinear function, not a piecewise one.
- Largest residuals of a snapped formula tend to sit at the domain edges, where the net extrapolates
  worst; compare max error to the snap tolerance (2% of output range) rather than chasing exactness.
