# Notes

## What the network does

A 1-input ReLU net (32 hidden units) on x0 in [0, 4], with regression output. Only 18 of the 32 units
have their kink inside [0, 4]. The other 14 are either always zero there or always linear, so together
they only add to the bias and the slope.

I worked out the curve as a piecewise-linear function (`scratch/knots.py`): kinks at x0 =
0.027, 0.107, 0.135, 0.247, 0.483, 0.487, 0.791, 1.124, 1.395, ..., 3.431.
- The slope only goes down, from 4.93 near 0 to 0.26 at the top, so the curve is concave.
- Kinks are dense near 0 and sparse further out.
- f(0.25) ≈ 0.50, f(1) ≈ 1.0, f(4) ≈ 2.0, and the slope near 4 (0.26) is close to 1/(2·sqrt(4)) = 0.25.

This is sqrt(x0). The largest gap between net and sqrt over [0, 4] is 0.0383, at x0 = 0, where the net
outputs 0.038 instead of 0. A finite ReLU net cannot follow sqrt's infinite slope at 0. The snap
tolerance is 0.0393, so the margin is small, but the check passes.

## Result

`agent.py`: `return sqrt(x0)`. `check.py` gives PASS (snap): 259098 inputs, none outside the tolerance.
An exact version would mean copying the 18 kinks with their unsnapped coefficients. That is just the
network again, so I kept the formula.

## Heuristics

- **1-input nets: compute the piecewise-linear form first.** Kink positions are -b/w. Drop units whose
  kink is outside the input box, since there they are constant or linear. Evaluate the net at each kink
  and print the slope of each segment. A list of 15–20 segment slopes is much easier to recognise than
  32 units.
- **Look at the slope sequence.** If the slopes only decrease or only increase, the function is concave
  or convex, which points to a smooth function (sqrt, log, x², exp), not a logic or step function. Kinks
  crowded at one end of the range show where the curvature is largest. For sqrt that is next to 0.
- **Test candidate formulas at landmark inputs:** 0, 1, the endpoints, and perfect squares or powers of 2.
  f(1) = 1, f(4) = 2 and f(0.25) = 0.5 together pin down sqrt.
- **Compare the end slope with the derivative** of the candidate (here 1/(2·sqrt(x)) at x = 4 gives 0.25,
  against 0.26 measured). That confirms the guess cheaply.
- **The largest error is usually where the target is not smooth or has an infinite derivative** (here x =
  0 for sqrt). Check that point against the snap tolerance before trying to fix the middle of the range.
- **Most of the units are padding.** About half the hidden units have no kink in the domain, and several
  in-domain kinks change the slope by less than 0.01. Expect the real structure to be much smaller than
  the unit count.
