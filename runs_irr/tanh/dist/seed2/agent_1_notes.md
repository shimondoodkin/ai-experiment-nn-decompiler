# Notes

## What the net does

The output is a bowl. It is 0.60 at the origin, about 1, 2 and 3 at radius 1, 2 and 3,
and 3.9 to 4.06 at the corners. It looks like the distance from the origin,
sqrt(x0^2 + x1^2), smoothed near 0 and flattened at the corners. It is not close enough to any
simple formula: the best k*sqrt(x0^2+x1^2+c) + linear has a max error of 0.18 against the
net, and the best radial polynomial f(r) has 0.14. The tolerance is 0.069. The net has a ring-shaped
wobble of about +-0.15 and a tilt (on the x0 axis it gives 2.92 at -3 and 3.12 at +3), so the
program stays a small tanh net.

## What I changed

1. **Eleven nearly linear units removed.** Units 0, 1, 2, 4, 6, 7, 8, 10, 11, 12 and 14 all share one direction,
   (0.8, -0.6)/|.|, and have small weights. Their pre-activations stay within +-0.6 of a bias near
   +-1. Together they come to 6.82 + 0.356 v - 0.025 v^2, where v = 0.8 x0 - 0.6 x1, with an error of 0.028.
   After the output weights of the other units were refitted, the quadratic part was no longer needed.
   What is left of them is the constant and the linear term 0.375 (x0 - x1).
2. **Five real units kept**, written with tanh(-z) = -tanh(z) so that each has a positive leading weight.
3. **Mirror symmetry.** The net is close to symmetric under (x0, x1) -> (-x1, -x0). For example, the
   x0 axis at -3 gives 2.922 and the x1 axis at +3 gives 2.920. The units form mirror pairs:
   tanh(0.6x0-0.1x1+0.98) with tanh(0.1x0-0.59x1+0.97), tanh(0.47x0+0.26x1+0.89) with
   -tanh(0.28x0+0.51x1-0.93), and tanh(0.28x0-0.27x1-0.78), which maps to itself. I tied each pair
   to one set of weights and one output coefficient, and fitted 8 hidden parameters plus the linear readout.
4. **Snapping.** Hidden weights are snapped to tenths (0.6/0.1/1, 0.5/0.3/0.9, 0.3/0.8). The output
   coefficients come from a minimax LP fit on a grid with the snapped hidden weights, rounded to two decimals.
   Rounding the outputs to one decimal as well failed (0.064 to 0.11 on the grid, too close to or over the tolerance).
5. Result: 5 tanh units instead of 16, and 4 output coefficients plus a constant instead of 17.
   `PASS (snap)`, max |program - net| = 0.043, tolerance 0.069.

An earlier version without the symmetry tie (5 units with 2-decimal weights, plus 6.72 + 0.39 v - 0.02 v^2
for the near-linear group) also passed, with an error of 0.034.

Dropping any one of the units or mirror pairs, even with all weights refitted, gives errors of 0.24 or more.
All five are needed.

## Heuristics

- **Find groups of units that share a direction and are nearly linear.** Normalize each unit's input weight
  vector. Units with the same direction and a small pre-activation range (|w|*input range << 1)
  together form a smooth 1-D function of one projection. Replace the group with a low-degree polynomial in
  that projection (here a quadratic, error 0.03). Often only the constant and the linear term survive once
  the other units' readout is refitted.
- **Refit the readout after each structural change.** It is a linear least squares (or a minimax LP via
  `linprog` for the L-infinity tolerance the checker uses), which is cheap and exact. Snapping the hidden
  weights first and then refitting the outputs works much better than rounding everything at once.
- **Look for symmetries of the output** by comparing the net on axes and diagonals (f(x0, x1) against f(-x1, -x0),
  f(x1, x0), f(-x0, -x1)). If units map onto each other under the symmetry (use tanh oddness to
  match signs), tie their weights. This halves the parameters and makes the structure readable.
- **Test the obvious target formula against the net first, not against the task.** Here hypot is clearly
  the intended shape, but the net differs from it by 0.18 to 0.6. That decides early that a pure formula cannot
  pass, and the work should go into compressing the net.
- **Snap to a coarse grid (tenths) for hidden weights.** tanh units tolerate it, because the readout refit
  absorbs the change. Output coefficients are more sensitive (a 0.05 change times an output range of 2 is
  already 0.1), so keep two decimals there unless a grid search over nice values passes.
- **Before keeping a unit, check whether it can be dropped:** refit with that unit removed and all other
  weights free. If the error stays far over tolerance, the unit is real.
