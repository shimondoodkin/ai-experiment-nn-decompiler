# Notes

## What I found
- 1 float input on [0.5, 4], 32 ReLU units, one regression output.
- I evaluated the net on a grid. It gives 1.99 at 0.5, 1.00 at 1, 0.496 at 2, 0.333 at 3 and 0.2475 at 4.
  The function is decreasing and convex, and x*f(x) stays close to 1. So it is f(x) = 1/x.
- Only about 12 of the 32 units have a kink inside [0.5, 4] with a slope change that matters. The kinks
  are dense near 0.5 (0.60, 0.72, 0.88, 1.08, 1.32, 1.63, 2.02) and get sparser as x grows. The slope
  changes are all negative and shrink with x: -1.05, -0.73, -0.52, -0.37, -0.24, -0.16, -0.11, ...
  That is the shape of a spline fit of a convex curve whose curvature falls off quickly, which fits 1/x
  (curvature 2/x^3).
- The other units are either always off in the box (kink outside it, inactive side), or always on, so they
  only add to the line a + b*x. A few have tiny slope changes (|dv| < 0.005) and are noise.

## What I changed
- I replaced the whole net with `1 / x0`.
- Check: PASS (snap), max |program - net| = 0.0097 against a snap tolerance of 0.035. The largest gap is
  near the first strong kink at x = 0.60, where the spline cuts the corner of the curve.
- An exact version would have to keep the piecewise-linear spline, about 12 kinks. The formula is far
  more readable and stays well inside tolerance, so I kept the formula.

## Heuristics
- For a 1-input net, evaluate it on a grid first and look at the values at round inputs (0.5, 1, 2, 3, 4).
  Recognisable values (2, 1, 1/2, 1/3, 1/4) give the formula away faster than reading weights.
- List each unit's kink position -b/w and its slope change w*v, and keep only kinks inside the input box.
  Units with kinks outside the box are constants or straight lines; drop or fold them in.
- When all slope changes have the same sign, the function is convex (positive) or concave (negative).
  When they shrink steadily as x grows, the curve flattens: think 1/x, log x, sqrt x or exp(-x). Test each
  with products and differences: x*f(x) constant means 1/x, f(2x) - f(x) constant means log, f(x)^2
  linear means sqrt.
- Kinks that crowd where the curvature is high match where a fitted spline needs them. Their density tells
  you where the target curve bends most.
- Try the plain formula first, with no fitted scale or offset. A ReLU spline of a convex curve sits
  slightly off the curve between knots, and that error is far below the 2% snap tolerance. Fitting a*f+b
  only adds noise coefficients.
- Expect PASS (snap), not exact, when replacing a spline with a smooth formula. The largest gap sits at the
  first, strongest kink.
