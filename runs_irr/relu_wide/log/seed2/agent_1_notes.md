# Notes

## What I changed and why

- The dump is a 1 -> 32 -> 1 ReLU net on x0 in [0.1, 5], so it is a piecewise-linear function of one
  variable. I evaluated it on a grid: it is increasing and concave, about 0 at x0 = 1, -2.28 at 0.1,
  0.694 at 2 and 1.616 at 5. These are the values of the natural log (ln 0.1 = -2.303, ln 2 = 0.693,
  ln 5 = 1.609).
- A least-squares fit `a*log(x0) + b` gave a = 1.0006, b = 0.0017, so it snaps to a = 1, b = 0.
- The 32 units are only the kinks of a piecewise-linear interpolation of log. About 20 of the kink
  positions fall inside the domain, packed most tightly near the low end where log curves most. The rest are
  dead or linear over the whole domain. None of them is meaningful on its own, so the whole network is
  replaced by the formula.
- `agent.py` is `return log(x0)`. Check: PASS (snap), max |program - net| = 0.020 at x0 = 0.1 (the
  chord error of the first linear piece, where log curves most), tolerance 0.078. An exact
  program would have to reproduce the ~20 linear pieces. That gives a long table and less insight, so I kept the formula.

## Heuristics

- For a 1-input regression net, evaluate it on a dense grid first and compare it with common
  functions (x, x^2, sqrt, log, exp, 1/x, sin). Concave and increasing, with a zero at 1, points to log.
- Fit `a*f(x) + b` for the candidate f. If a ≈ 1 and b ≈ 0, snap them and drop the network entirely.
- Kink positions -b/w that crowd where the curvature is highest point to a ReLU approximation of a smooth
  curve, not a piecewise rule. A smooth formula is the right refactoring then.
- The largest error against the smooth formula sits at the domain edge with the highest curvature (here
  x0 = 0.1). Check that point first when estimating whether snap tolerance holds.
- Units whose kink is outside the input box are linear (or zero) over the domain. Fold them into an
  affine term rather than reading meaning into them.
