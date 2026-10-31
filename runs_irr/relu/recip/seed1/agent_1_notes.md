# Notes

## What the network does

One input x0 in [0.5, 4], 8 ReLU units, regression output.

- Dead units on the domain: h1_1, h1_6 (kink at negative x, negative slope) and h1_4 (kink at 0.388,
  negative slope). They are 0 for every x0 >= 0.5, so they are removed.
- Always-on units: h1_0 and h1_3 (kinks at -2.27 and -1.73), and h1_5 and h1_7 (kinks at 0.497 and
  0.496, just below the lower bound 0.5). On the domain they are plain linear terms, so they fold
  together with the output bias into one line: 0.87504 - 0.17217 x.
- One real kink: h1_2 = relu(0.99684 - 0.78867 x), kink at x = 1.2640, weight 1.61685. It adds a
  steeper segment on the left.
- `c + s*x + a*relu(k - x)` with a > 0 is exactly `max(left line, right line)`. Left line:
  2.48677 - 1.44732 x; right line: 0.87504 - 0.17217 x.

So the net is a convex, decreasing two-piece linear function. Values: 1.763 at 0.5, 1.039 at 1,
0.53 at 2, 0.186 at 4 - it looks like a two-segment approximation of something like 1/x (a guess,
not checked against any target).

## Versions

- `agent.py` (written): `max(2.48677 - 1.44732 x, 0.87504 - 0.17217 x)`. PASS (exact), max error 1.1e-5.
  Five significant digits are the fewest that keep exact; 7 digits give 3.5e-7.
- Snapped alternative: `max(2.5 - 1.45 x, 0.175 * (5 - x))`. PASS (snap), max error 0.0119 (tolerance
  0.0315). Rounder values failed: `0.875 - x/6 + 1.25 relu(1.25 - x)` misses by 0.034 at x = 0.5,
  and a left slope of 1.5 misses by about 0.039 at x = 1. I kept the exact version because the checker ranks
  exact above snap and the formula is just as readable.

## Heuristics

- Evaluate every first-layer kink `-b/w` against the input bounds first. A unit whose kink is outside
  the box is either dead (remove it) or always on (fold it into a linear term). Kinks just outside
  the box (here 0.496, 0.497 vs lo = 0.5) count as always-on. Check the direction, not just the location.
- After folding, a 1-input net with one surviving kink is `line + a*relu(+-(x - k))`; rewrite it as
  `max(line1, line2)` when a > 0 (convex) or `min(...)` when a < 0 (concave). That is exact and
  easier to read than the relu form.
- Express each segment by its own slope and intercept (compute the left segment as
  intercept + a*b, slope + a*w), then round digits until the exact check breaks. With atol 1e-5,
  about 5 significant digits are needed on slopes when |x| reaches 4.
- For snapping, grid-search small sets of nice values for each segment's intercept and slope, using
  max error on a dense grid against the folded exact formula. Then confirm with check.py. The error
  is largest at the domain ends and at the kink, so check those points first.
- Kink position and segment slopes interact: snapping the kink alone (1.264 -> 1.25) while keeping
  the relu form can fail where moving to a max-of-lines form with snapped lines passes, because the
  intercepts absorb the shift.
