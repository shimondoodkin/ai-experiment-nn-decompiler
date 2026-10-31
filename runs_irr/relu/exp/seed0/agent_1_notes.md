# Notes

Network: 1 float input on [-2, 2], 8 ReLU units, regression output (range about 0.04 to 7.07).

## What I changed

1. **Dead units removed.** For each unit I evaluated its pre-activation at both ends of the box
   (it is linear in x0, so the max is at an endpoint). Units 0, 4, 5 and 6 are negative on the whole
   box, so they are always 0 and were dropped. Removing them alone gives `PASS (exact)`
   (kept as `scratch/exact.py`, max error 1e-6).
2. **Read off the piecewise-linear shape.** The four live units have kinks at x0 = 0.090, 0.093,
   0.096 (units 3, 2, 1) and 1.222 (unit 7). The slopes increase left to right
   (0.417, then 1.980, then 5.055), so the function is convex and equals the upper envelope
   (max) of its linear pieces:
   `max(0.4165*x0 + 0.870, 1.980*x0 + 0.718, 5.055*x0 - 3.040)` (max error 0.0015).
   The three kinks near 0.09 merge into one kink (they are within 0.006 of each other).
3. **Snapped coefficients** to simple values: slopes 0.4, 2, 5 and intercepts 0.9, 0.7, -3.
   The kinks become x0 = 0.125 and x0 = 1.233. Max error 0.070 at x0 = 2, against a snap tolerance
   of 0.141: `PASS (snap)`.

Final program (`agent.py`):

    low = 0.4 * x0 + 0.9
    mid = 2 * x0 + 0.7
    high = 5 * x0 - 3
    return max(low, mid, high)

The shape is roughly exponential-like growth (convex, about 0.9 near 0, 7 at 2), approximated by
three lines; I kept the piecewise-linear form because that is what the network computes.

## Heuristics

- For a 1-input ReLU net, compute each unit's pre-activation at the two box ends: if both are
  negative the unit is dead; if both are positive it is linear and folds into the bias.
- Kink positions are -b/a per unit. Kinks closer together than a few percent of the input range can
  usually be merged into one kink without leaving the snap tolerance.
- Compute the slope of every segment. If the slopes only increase, the function is convex and can
  be written as `max(line1, line2, ...)` (decreasing slopes: concave, use `min`). This is far more
  readable than a sum of ReLUs and needs no thresholds.
- Snap slopes and intercepts of the envelope lines, not the raw unit weights: errors of the lines
  are directly the output error, so the effect of rounding is easy to bound.
- Check the endpoint values of the box first; the worst error of a snapped piecewise-linear
  program is at a kink or a box end.
- Keep an exact version (dead units removed only) as a fallback before snapping.
