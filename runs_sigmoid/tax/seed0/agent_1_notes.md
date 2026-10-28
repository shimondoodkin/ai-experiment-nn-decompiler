# Notes

## What the network does

One float input x0 in [0, 200], 8 sigmoid hidden units, regression output (range about -0.14 .. 60.6).
Sigmoid centres (-b/w): 89.7, 134.2, 92.5 (falling), 122.1 (falling), -5.7, 181.1, 38.6 (falling),
168.1. Their slopes are small (0.03-0.14), so every unit is a soft ramp several tens of units wide;
the sum is a smooth, monotone increasing, convex-ish curve.

Numerical derivative of the net: about 0.12 near 0, about 0.2 over 30-60, about 0.4 over 90-180, then
it drops to about 0.2 at the far edge (x0 > 185, an edge effect of the last sigmoids saturating).

## What I changed

- Replaced all 8 sigmoids with a single convex piecewise-linear function:
  `max(0.18*x0 - 0.7, 0.39*x0 - 16.45)`, i.e. slope 0.18 below a knee at x0 = 75 and slope 0.39
  above it (equivalently `0.18*x0 - 0.7 + 0.21*relu(x0 - 75)`).
- Found it by a minimax search over one-knot ramps with snapped slopes (multiples of 0.01/0.025) and
  knots on multiples of 5, intercept chosen as the mid-range of the residual. Max error 0.914 vs a
  snap tolerance of 1.216 (2% of the output range). The worst point is x0 = 200 where the net bends
  down.
- Tried round slopes 0.2 / 0.4: `max(0.2*x0 - 1, 0.4*x0 - 18)` misses by 1.36 at x0 = 200 (fails).
  Adding a third line for the top-end flattening (`min(max(0.18x-0.75, 0.4x-18), 0.2x+20)`) only
  brings the error to 0.76, so it was not worth the extra node. Two knots (75 and 195) reduce it to
  0.77; three knots (35, 80, 190) to 0.47 - none are needed for a pass.
- Result: `PASS (snap)`, max |program - net| = 0.914. An exact pass is not reachable without keeping
  the sigmoids.

## Heuristics

- For 1-input sigmoid nets, compute each unit's centre `-b/w` and width `1/|w|`; if widths are large
  relative to the gaps between centres, the net is a smooth curve and no single unit is meaningful -
  fit the whole curve instead of simplifying unit by unit.
- Plot / tabulate the numerical derivative on a grid. Plateaus of the derivative are linear pieces;
  transitions mark knots. Here: plateaus near 0.2 and 0.4 gave a 2-piece ramp.
- Fit piecewise-linear forms with knots on a nice grid (multiples of 5), solve slopes by least
  squares, then snap slopes and search intercept by minimax (intercept = mid-range of residual).
  Compare the resulting max error with the snap tolerance (2% of the output range).
- Write a convex piecewise-linear function as `max(line1, line2, ...)` (concave as `min`); it is the
  shortest form and makes the knot implicit.
- The error is usually largest at the domain edges, where saturating sigmoids bend the curve; check
  the endpoints first and add an edge piece only if the tolerance requires it.
- Prefer the fewest pieces that pass with a comfortable margin over rounder coefficients that fail
  (0.39 passed, the rounder 0.4 did not because of the edge).
