# Notes

## What I changed and why

- `dump.py` is a 1 -> 32 -> 1 ReLU net on x0 in [0, 4]. Its only kinks are where a hidden unit
  crosses zero, at -b/w. 14 of them fall inside [0, 4]. The other 18 units are dead or purely
  linear on the box.
- I sampled the net on a grid. Values at x0 = 0, 1, 2, 3, 4 were 0.056, 0.999, 1.418, 1.732,
  2.002, and the slope keeps falling, which matches sqrt(x0) (1, 1.414, 1.732, 2).
- Plain `sqrt(x0)` fails only near x0 = 0. The net there is 0.0556, an error of 0.0556 against a
  snap tolerance of 0.039. A ReLU net cannot copy sqrt's vertical slope at 0, so it starts with a
  positive offset and a steep first segment (kinks at 0.00003, 0.045, 0.048, 0.18, ...).
- Final program: `max(sqrt(x0), 0.05)`. PASS (snap). Max |program - net| is 0.015, from the net's
  own piecewise-linear wobble around sqrt in the middle of the range, not from the floor.
  `sqrt(x0 + 0.003)` also passes (max error 0.021), but the floor fits the net better and is
  easier to read.
- No exact version is practical: an exact one would be the 14-segment spline itself.

## Heuristics

- For a 1-input regression net, list the kinks (-b/w inside the box) and evaluate the net at
  integer inputs. Values like 1, 1.414, 1.732, 2 identify sqrt; check the other common shapes too
  (x^2, 1/x, log, exp) against those points before fitting anything.
- When a clean formula fails only at a domain edge where it has an infinite or very large slope
  (sqrt at 0, log at 0, 1/x near 0), the net has smoothed that singularity. Add a clamp
  (`max(f, c)` with c about the net's value at the edge) or a small input shift (`f(x + eps)`),
  and pick whichever has the smaller max error.
- Look at where the remaining max error sits. If it is in the middle of the range, it is the net's
  own approximation error, and tuning the edge constant further will not help, so round the
  constant to a simple value.
- Many hidden units in a wide 1-D ReLU net are dead (kink outside the box, always negative) or
  always on (they add up to one linear term). Only the kinks inside the box carry shape.
