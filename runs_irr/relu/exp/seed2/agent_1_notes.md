# Notes

## What the network does

One input x0 in [-2, 2], 8 ReLU units, regression output. Seen as a 1-D function it is piecewise
linear. Three units have their kink outside [-2, 2] (at -3.72, -2.08, 2.00 / 3.03), so inside the box
they are plain linear terms (or zero) and fold into the base line. Four kinks are left, at about
-0.56, 0.35, 1.03 and 1.56. The slope goes up at each one: 0.30, 0.92, 2.02, 3.67, 5.97, and the
values are f(-2)=0.086, f(-1)=0.38, f(0)=1.03, f(1)=2.66, f(2)=7.30.
That matches exp(x0) (0.135, 0.368, 1, 2.718, 7.389): it is convex and grows by about e for each unit
step, and each slope is close to the value at the middle of its segment.

## Versions

- `agent.py`: `return exp(x0)`. PASS (snap). The largest gap is 0.091 at x0 = 2, where the net's last
  segment falls below the curve. The snap tolerance is 0.144 (2% of the output range).
- `scratch/exact.py`: the net written exactly as a base line plus 4 hinges,
  `0.6756 + 0.29505*x0 + 0.62726*relu(x0+0.56006) + 1.09304*relu(x0-0.35104) + 1.65484*relu(x0-1.02514) + 2.30394*relu(x0-1.56144)`.
  PASS (exact), max error 1.1e-5. Five numbers per hinge would need to be kept for exact
  equality; none of them snap to anything simple, which is what you would expect from a
  piecewise-linear fit to a curve. I kept the formula as the main version because it explains the
  net and still passes.

## Heuristics

- For a 1-input ReLU net, compute each unit's kink at -b/w first. Units whose kink lies outside the
  input box are either dead or linear across the whole box; fold them into the base line.
- Then sample the slopes between the remaining kinks. If the slopes keep growing (convex) or keep
  shrinking (concave) and the hinge positions look irregular, the net is probably fitting a smooth
  curve, not a function made of hinges. Compare the values at integer points with exp, x^2, |x|,
  sqrt, log and sigmoid before trying to snap the hinges.
- Ratios between values one unit apart (here about e) and slope ≈ value (f' = f) are quick tests for
  exp.
- Check the smooth formula against the snap tolerance (2% of the output range). The largest error
  usually sits at the ends of the box, where the last linear segment cuts across the curve.
- Keep an exact hinge version as a backup: the base line on the first segment plus one
  `c*relu(x - k)` per inner kink, with c the slope change. It needs about 5 significant digits to
  pass as exact.
