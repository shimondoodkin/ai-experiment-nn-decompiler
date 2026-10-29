# Notes

## What the network is

1 input `x0` in [0, 200], 8 tanh units, regression output. Every output weight is about +-7, so
each unit is a smooth step of height about 14.

Normalising each unit to `w * tanh(k * (x0 - c))` with `k > 0` (using `tanh(-z) = -tanh(z)`) shows
that all units rise, and gives their centres:

| unit | centre c | slope k | on [0, 200] |
|------|----------|---------|-------------|
| 6 | 0.19 | 0.018 | soft rising step |
| 0 | 72.3 | 0.024 | soft rising step |
| 1 | 108.3 | 0.043 | soft rising step |
| 3 | 149.0 | 0.042 | soft rising step |
| 7 | 183.6 | 0.049 | soft rising step |
| 2 | -31.2 | 0.117 | saturated at +1 (tanh >= 0.997) |
| 4 | -1.67 | 2.31 | saturated (tanh <= -0.999) |
| 5 | -8.2 | 0.48 | saturated at +1 (tanh >= 0.999) |

Steps 1: fold the three saturated units into the constant (27.65). This 5-tanh version
(`scratch/tanh5.py`) passes snap with max error 0.02.

Step 2: the steps are wide (widths 40-100) and overlap, so their sum is close to a ramp.
The numerical derivative is about 0.16-0.19 for x0 < 70 and about 0.39-0.41 for x0 > 100
(with a drop near 200 from the last step's edge). A least-squares fit with one hinge put the
knot at 81. A grid search over rounded coefficients gave

    net(x0) = max(0.19*x0 - 1, 0.39*x0 - 17) = 0.19*x0 - 1 + 0.2*relu(x0 - 80)

with max |program - net| = 0.64 (tolerance 1.22, which is 2% of the output range of 61). Final
`agent.py` is this one-hinge formula. It passes at the snap level, not exact: 8 tanh units became
one hinge. Rounder variants did worse: `max(0.16x, 0.4x-20)` had error 2.4,
`0.2x - 1 + 0.2 relu(x-80)` had 2.4, and `max(0.2x-1.5, 0.4x-19)` had 1.09, which is too close to
the tolerance.

Interpretation: the output rises at about 0.19 per unit up to x0 = 80, then at about 0.39 per
unit. It looks like a two-bracket rate schedule.

## Heuristics

- Normalise every tanh/sigmoid unit to `w * act(k*(x - c))` with `k > 0`, using
  `tanh(-z) = -tanh(z)`. Then the centre `c = -b/a` and the sign of `w` show which way each
  step goes. Read it as a table of steps.
- A unit whose centre is outside the input box by more than about 3/k is saturated. Fold it into
  the output constant (`w * tanh(+-inf) = +-w`). Check how far from +-1 it is at the nearest edge.
- When every output weight has about the same size (here +-7), the net is building a function
  from equal-height steps. If the step widths (about 2/k) are larger than the gaps between
  centres, the sum is a smooth ramp, not a staircase. Look at the numerical derivative and fit
  piecewise-linear pieces, not thresholds.
- For 1-input regression: tabulate the net, take the gradient, find the slope plateaus, fit
  hinges by least squares (`[1, x, relu(x-k)...]`), then grid-search rounded coefficients and
  keep the simplest that stays well under the tolerance (aim for about half of it, because the
  checker samples densely, including boundary points).
- Edge effects: a step centred near the domain edge bends the curve there (here the slope drops
  at x0 = 200). A hinge fit should tolerate this rather than add a piece for it.
- Keep a faithful intermediate (saturated units removed, exact rewrites) and check it first, to
  separate "my algebra is right" from "my approximation is good enough".
