# Notes

## What the network computes

A 2 -> 16 -> 1 ReLU regression net on x0, x1 in [-2, 2]. On a 9x9 grid the output is
almost exactly `x0 * x1`. For example net(1, 1) = 1.00, net(1.5, 1.5) = 2.28, net(-1, 1) = -1.03,
and it is 0 along both axes. The 16 hidden units are a piecewise-linear approximation of the
saddle `x0*x1 = ((x0+x1)^2 - (x0-x1)^2) / 4`. Most of their weight vectors point along (1, 1) or (1, -1).

## What I changed

- Replaced the 16 ReLU units with the formula `p = x0 * x1`.
- Plain `x0 * x1` fails the check: max error 0.238 against a tolerance of 0.155. All the large
  errors are in the four corners, where |x0*x1| is near 4 and the net undershoots (it gives
  about 3.76 to 3.89 instead of 4). Elsewhere the error is a ripple of about ±0.08 from the
  piecewise-linear fit.
- Tried two ways of modelling the corner shortfall:
  - `0.97 * x0 * x1` passes, but only just (0.1527 vs 0.1549).
  - `p - p^3/500` (an odd soft saturation) passes with more room, max error 0.112. Scanning
    the divisor over 256 to 800 put the best value near 500, a round number.
- Final `agent.py`: `p = x0*x1; return p - p*p*p/500`. Result: PASS (snap). PASS (exact) is not
  possible without the 16 kinks themselves.

## Heuristics

- Before reading the weights, evaluate the dump on a coarse grid. A table that is zero on both
  axes and has sign pattern +,-,-,+ by quadrant is a product `x0*x1`. Units in pairs along (1, 1)
  and (1, -1) are the ReLU form of `((x0+x1)^2 - (x0-x1)^2)/4`.
- Fit the clean formula first, then look at where the residual exceeds tolerance. If the
  residual is concentrated where |target| is largest (box corners), the net is saturating. Model
  that with an odd shrink term (`p - p^3/k`, or a scale just below 1), not with clamps or extra
  kinks.
- Prefer the candidate with margin over the one that passes by a hair. The checker's input set
  can grow with counterexamples.
- A ripple of a few percent left over from a piecewise-linear fit of a smooth function is noise.
  Do not try to reproduce individual kinks when the snap tolerance already covers them.
