# Notes

## What the network is
One input, 8 ReLU units, one linear output. On the input box [-1, 3]:

- Units 0, 4, 5, 6 are dead: their kinks (x0 = -11.85, -2.48, 5.02, -20.8) lie outside the box on the
  side where the unit is off, so they are always 0. Removed.
- Unit 7 has its kink at 3.087, just outside the box on the "on" side, so it is always active: a plain
  linear term folded into the constant and slope.
- Units 1, 2, 3 are live, with kinks at x0 = 2.51, -0.089, 1.031. All three point left (negative
  input weight), so the output is a convex, decreasing piecewise-linear curve with 4 segments.

Exact reduced form (PASS exact, max error 6e-7) kept in `scratch/exact.py`.

## Snapping
Rewrote as `tail + sum(k_i * relu(t_i - x0))` and refit:

- kinks snapped to 2.5, 1, 0 (from 2.51, 1.031, -0.089);
- kink weights 0.445 / 0.564 / 0.792 snapped to 0.4 / 0.6 / 0.8, tail slope 0.184 to 0.2, so the
  segment slopes are -0.2, -0.6, -1.2, -2.0;
- constant chosen as the minimax offset for those snapped slopes (0.729 -> 0.73), giving
  `1.33 - 0.2*x0` as the tail.

Result: PASS (snap), max |program - net| = 0.047 against a tolerance of 0.083. The largest error is at
x0 = -1 (the snapped kink at 0 instead of -0.089 accumulates there). Snapping only the weights but
keeping the kink at -0.1 gives 0.033, so the kink at 0 costs about 0.013 extra; I took the rounder kink.

## Heuristics
- For a 1-input ReLU net, compute each unit's kink t = -b/w first. Units whose kink is outside the box
  are either dead (box on the off side: delete) or always on (box on the on side: fold into the
  linear part). Only units with kinks inside the box shape the function.
- Rewrite each live unit as `|c*w| * relu(t - x)` or `relu(x - t)` so its weight and its kink are
  separated; then the coefficients are directly "slope change at the kink", which are easy to snap.
- If all live units point the same way and their output weights have the same sign, the function is
  monotone and convex/concave: describe it by its segment slopes, which often turn out to be a clean
  sequence (here -0.2, -0.6, -1.2, -2.0).
- Snap kinks and slope changes first, then pick the constant as the minimax offset
  (-(max err + min err)/2) for the snapped shape rather than rounding the original constant; this
  recovers most of the error budget.
- Fit with least squares on a dense grid for each candidate kink set before rounding; it shows quickly
  which kink snap is affordable (here moving -0.089 to 0 cost about 0.04 of 0.083 tolerance).
- The error of a moved kink grows linearly with distance from the kink toward the box edge, so check
  the box corners: they are where a snapped program fails first.
