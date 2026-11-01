# Notes

## What I did

- The net is 1 input -> 32 ReLU -> 1, regression on x0 in [-1.5, 4]. With one input every hidden unit
  is just a kink at x = -b/w with a slope change w*v (v = output weight). I listed all kinks and the
  piecewise slopes between the in-range ones (scratch/look.py).
- 15 of the 32 kinks fall outside [-1.5, 4]; those units are plain linear terms (or zero) inside the box.
  Several in-range units have slope changes below 1e-3 (units 0, 3, 15, 27, 30, 31) and do nothing.
- The remaining curve is concave and monotone increasing: slopes fall from 1.66 at x=-1.5 to 0.18 at x=4,
  roughly halving each time x+2 doubles, i.e. slope ~ 1/(x+2).
- Sampled values matched a log: f(-1) = 0.008 ~ log 1, f(0) = 0.693 ~ log 2, f(4) = 1.796 ~ log 6,
  f(-1.5) = -0.683 ~ log 0.5. So the program is `log(x0 + 2)`.
- Check: PASS (snap), max |program - net| = 0.0106 against a snap tolerance of 0.0496. The largest
  deviation is at the left edge x0 = -1.5, where log is steepest and a piecewise-linear net with few
  kinks there cuts the corner. PASS (exact) would require reproducing all ~17 in-range kinks, i.e. the
  dump itself, so I kept the formula.

## Heuristics

- For a 1-input ReLU net, compute each unit's kink -b/w and slope change w*v, drop kinks outside the
  input box (they are linear there) and units with negligible slope change, then read the piecewise
  slopes. The slope sequence often identifies the function: constant -> linear, slopes ~ 1/(x+c) -> log,
  slopes growing linearly -> square, slopes proportional to the value -> exp.
- Test a candidate closed form at a few "nice" points (where the formula gives 0, log 2, integers, ...)
  before running the checker; a zero crossing at an integer input (here f(-1) ~ 0) pins the shift.
- The net's error against a smooth target is largest at the edges of the input box, especially where the
  target is steepest; expect the snap residual to concentrate there, not in the middle.
- When many units are near-zero (weights ~1e-3 or 1e-9), the net is over-parameterised; the effective
  number of kinks is much smaller than the width, and a smooth formula is a better summary than a
  pruned piecewise-linear program.
