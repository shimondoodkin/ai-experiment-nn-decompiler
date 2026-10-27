# Notes

## What I changed and why
- The dump is 1 -> 3 -> 1: three ReLUs `relu(x0 - k)` with knots k = 2, 3, 4 and output weights
  +1, -2, +1. That is a second difference of a ramp, i.e. a piecewise-linear hat (triangle) function:
  slope 0 for x0 < 2, +1 on [2, 3], -1 on [3, 4], 0 again for x0 > 4 (slopes sum 1 - 2 + 1 = 0).
- Rewrote it as `relu(1 - abs(x0 - 3))`: a triangular bump centred at 3, half-width 1, height 1.
- No snapping was needed; the weights were already integers. Check: PASS (exact).

## Heuristics
- For a 1-D input, read each hidden ReLU as a knot at -bias/weight and the output as a sum of slope
  changes; accumulate slopes left to right to get the piecewise-linear shape.
- Output weights of the pattern (+a, -2a, +a) on equally spaced knots k-d, k, k+d form a hat:
  `a*d*relu(1 - abs(x - k)/d)`. Patterns (+a, -a) on two knots form a clamp/saturating ramp
  (`a*min(max(x - k1, 0), k2 - k1)`).
- If the slopes sum to zero the function is constant beyond the last knot - check the tail value to
  decide whether a bump (returns to 0) or a step (plateau) is meant.
- Prefer abs/min/max formulations over raw ReLU sums: they make centre, width and height explicit.
