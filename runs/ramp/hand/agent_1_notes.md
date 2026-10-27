# Notes

The dump is `relu(x0) - relu(x0 - 1)` with all weights already exactly ±1 and bias -1.
That difference of two shifted ReLUs with equal slopes is a saturating ramp:
0 for x0 <= 0, x0 on [0, 1], 1 for x0 >= 1. So the whole net is `clamp(x0, 0, 1)`,
written as `min(max(x0, 0), 1)`. No snapping was needed. Check: PASS (exact).

## Heuristics

- `relu(x - a) - relu(x - b)` with a < b and equal unit slopes is `clamp(x, a, b) - a`,
  i.e. a ramp that saturates; replace with `min(max(x, a), b) - a`.
- A sum of ReLUs of the same 1-D input is a piecewise-linear function; locate the kinks
  (where each pre-activation is 0) and the slope change at each to read off the shape.
- If weights are already clean integers, look for a closed form (clamp, abs, max/min)
  rather than just simplifying arithmetic: `relu(x) + relu(-x) = abs(x)`,
  `relu(x) - relu(-x) = x`, `a + relu(b - a) = max(a, b)`.
- Check the input bounds: kinks outside the input range can be dropped.
