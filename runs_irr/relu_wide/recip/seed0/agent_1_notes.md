# Notes

## What the network computes

One float input x0 in [0.5, 4], regression output, one hidden layer of 32 ReLUs.
Evaluating the dump on a grid gave 1.989, 1.333, 1.000, 0.799, 0.664, ..., 0.499 (x0=2), ..., 0.248 (x0=4),
which is 1/x0 (x0 * net(x0) is about 1 everywhere). Comparing against candidates sqrt(x0), log(x0) and 1/x0
on the same grid made the match obvious.

## What I changed

- Replaced the whole network with `return 1 / x0`.
- Check: `PASS (snap)`, empirical, 252k inputs; max |program - net| = 0.0107 against a snap tolerance of 0.0348
  (2% of the output range 1.74). The largest deviation is at the left edge x0 = 0.5 (net 1.989, formula 2.0),
  where the curve is steepest and the net's piecewise-linear chords undershoot the most.
- Structure of the dump (for reference, not used in the final program): 11 of the 32 units are dead on
  [0.5, 4] (knot outside the box and pointing away from it: units 0, 2, 3, 4, 11, 12, 13, 21, 25, 27, 30);
  unit 5 is always on (pure linear term); the remaining ~20 units have knots spread over 0.51..3.53, densest
  near the left end (0.51-1.1), where 1/x curves most. The large negative slope contributions sit at small x
  and shrink toward x = 4, i.e. a convex decreasing spline — the shape of a reciprocal.
- I did not aim for PASS (exact): an exact program would have to keep ~20 ReLU knots (a spline of the fit
  error), which is less readable than the formula and carries no extra meaning.

## Heuristics

- For a 1-input regression net, first tabulate the output on ~15 grid points and compare it with a small
  library of candidate functions (x, x^2, sqrt, 1/x, log, exp, abs, sin) and simple affine variants
  (a*f(x)+b). Products like x*y(x) ~ const or ratios y(2x)/y(x) ~ const quickly identify 1/x and powers.
- Range of the input domain is a hint: a box starting at 0.5 (not 0) suggests a function singular at 0
  (1/x, log x, sqrt behaviour near 0).
- List each unit's knot -b/w and classify it as dead / always-on / kinked inside the box. Dead units can be
  dropped, always-on units fold into a linear term, and the density of interior knots shows where the target
  curves most (dense knots where |f''| is large).
- The sign pattern of slope contributions (w_in * w_out) per knot tells convexity: all one sign of change in
  slope across knots = convex/concave function; decreasing and convex over a positive domain is the
  signature of 1/x or exp(-x).
- Expect the formula's largest disagreement with the net at domain edges and where curvature is highest;
  check that the max error there is well under the snap tolerance before accepting the formula.
- Prefer the clean closed form at PASS (snap) over an exact spline reproduction when the error is a small
  fraction of the tolerance (here about 30%).
