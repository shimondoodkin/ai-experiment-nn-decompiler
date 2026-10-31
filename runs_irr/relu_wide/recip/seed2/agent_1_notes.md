# Notes

## What the network is
One float input x0 in [0.5, 4], one regression output, 32 ReLU hidden units, a single layer.

## What I did
- Loaded `weights.npz` and evaluated the net on a grid. The values were 1.99 at 0.5, 1.00 at 1, 0.50 at 2,
  0.334 at 3 and 0.2525 at 4, which is 1/x0 to two or three digits.
- Looked at the kinks inside the domain (-b/a for each unit). There are 16, packed most densely at the left
  end (0.58, 0.69, 0.81, 0.96, 1.15, 1.38, ...). Every slope change there is negative, so the curve is decreasing
  and convex, the way 1/x is. The units with kinks outside [0.5, 4] are constant or linear there, and the
  rest have contributions near zero (|a*c| < 0.001). They are fitting noise.
- Replaced the whole net with `return 1 / x0`.
- `check.py`: PASS (snap), max |program - net| = 0.0105 (at x0 = 0.5, where the piecewise-linear net
  cuts the steepest part of the curve), and the snap tolerance is 0.0347. An exact pass would mean
  reproducing the 16-kink polyline, which is not readable, so I kept the formula.

## Heuristics
- For a 1-input regression net, first tabulate the output at simple points (0.5, 1, 2, 3, 4) and
  compare it with candidate elementary functions (1/x, sqrt, log, exp, x^2). Values like 2, 1, 0.5, 0.333 at
  x = 0.5, 1, 2, 3 identify 1/x right away.
- Kink density tells you the curvature: the kinks of a ReLU net cluster where the target bends most, and
  the sign of each slope change (a_i * c_i, adjusted for which side is active) gives convexity or concavity. Dense,
  same-sign kinks at one end of the domain point to 1/x or log (convex/concave near small x).
- Units with |a*c| far below the output tolerance, or with kinks outside the domain, carry no structure.
  Ignore them when reading the shape.
- When a smooth elementary function passes snap, prefer it over a piecewise-linear refit. The leftover
  error is the net's own approximation error, and it is largest where the curvature is largest
  (here at the left end).
- Check the error budget early: the snap tolerance is 2% of the output range. Compute max |f - net| on a
  dense grid before running the checker, to see how much room a candidate formula has.
