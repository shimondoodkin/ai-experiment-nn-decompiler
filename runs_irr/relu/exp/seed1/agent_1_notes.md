# Notes

## What the network is
A 1 -> 8 -> 1 ReLU net on x0 in [-2, 2], regression. Each hidden unit is a hinge, so the net is a
piecewise-linear function of x0.

- Unit 6 (kink at -4.49, active only for x0 < -4.49) never fires on the domain. Removed.
- Unit 1 (kink at 2.015, active for x0 < 2.015) is always on in the domain, so it is just a linear
  term. Folded into the constant and slope.
- The other 6 units are all increasing hinges, with kinks at -1.584, -0.571, 0.158, 0.738, 1.229 and 1.645.

Segment slopes from left to right: 0.045, 0.33, 0.77, 1.55, 2.74, 4.05, 5.96. They only increase, so the
function is convex. Each slope is roughly the value of the function at that point: f(0) = 0.997,
f(1) = 2.745, f(2) = 7.32, f(-1) = 0.378, f(-2) = 0.154. That is a piecewise-linear fit of **exp(x0)**.

## Result
`agent.py` is `return exp(x0)`. Checker: **PASS (snap)**. The max |exp - net| is 0.070 at x0 = 2 (the
net's chord falls short of the curve at the end of the range). The snap tolerance is 0.143.

An exact pruned version (6 relus plus one folded linear term) is in `scratch/exact.py`. It gives
**PASS (exact)** (max diff 1.5e-6), but it is only the network with dead and linear units removed.
I kept the formula as the answer because it is the readable function the net approximates.

## Heuristics
- For 1-input ReLU nets, compute each unit's kink -b/w and which side it is active on. Drop units whose
  active side is entirely outside the domain. Fold units that are active over the whole domain into a
  linear term.
- List the segment slopes in kink order. If the slopes only increase (or only decrease), the function is
  convex (or concave), so it is a smooth curve fitted by chords. Then try exp, x^2, |x|, log and sqrt.
- Exp check: slope ≈ value at each point, f(0) ≈ 1 and f(1) ≈ e. For x^2 the slopes grow linearly
  instead.
- A smooth formula's largest error against a chord fit is usually at the domain ends, where the net has
  the fewest kinks. Measure the max error against the snap tolerance (2% of the output range) before
  you trust the formula.
- Keep an exact pruned version as a fallback when the formula's error is close to the tolerance.
