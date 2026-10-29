# Notes

## What the net does
2 inputs (x0 in [1, 6.9], x1 in [0.1, 2.5]), 8 tanh units, argmax over 3 classes.
Printing the class on a 60x25 grid showed three bands separated by two nearly straight lines;
class 0 and class 2 never touch.

## What I changed
- I bisected the class change along x0 for each x1 from 0.1 to 2.5 (step 0.1):
  - 0/1 boundary: x0 = 4.61 at x1 = 0.1 down to 1.23 at x1 = 1.1. Slope about -3.3 (slightly
    steeper at the low end), and it leaves the box near x1 = 1.15.
  - 1/2 boundary: x0 = 6.69 at x1 = 0.1 down to 4.09 at x1 = 2.5. Slope about -1.1, with a little
    curvature of about ±0.03 in x0.
- I fitted each boundary with a line and snapped it to one decimal: `x0 + 3.3*x1 < 4.9` -> class 0,
  `x0 + 1.1*x1 > 6.8` -> class 2, otherwise class 1.
- Before snapping, I measured the checker's unsure zone (5% of the p95 top-2 margin, about 0.56).
  It is about ±0.08–0.1 wide in x0 around each boundary, so any line that stays within about 0.08
  of the true boundary passes. I checked candidate constants against the boundary table at
  x1 = 0.1, 0.7, 1.1 and 2.5 before running the checker.
- All 8 tanh units are gone. Result: PASS (snap), 0 confident disagreements. All disagreements are
  in the unsure zone, which is 4.3% of inputs. Exact is not reachable without keeping the tanh
  curvature.

## Heuristics
- For a 2-input argmax net, first print the class map on a coarse grid. It shows at once how many
  boundaries there are and which classes touch.
- Bisect the boundary along one input for a sweep of values of the other input. A table of crossing
  points shows whether the boundary is straight; check the first differences.
- Compute the checker's unsure-zone width in input units (score margin / gradient, or measure it
  directly along a line). It tells you how much you can round coefficients. Pick the simplest
  1-decimal line that stays inside the zone at both ends and in the middle of the boundary.
- When two classes never touch, the argmax is a chain of one-sided tests, e.g. `if A: 0 elif B: 2
  else 1`. The order of the tests does not matter where their regions do not overlap.
- tanh nets with smooth boundaries usually reduce to linear rules for a snap pass. Mild curvature
  (well under the zone half-width) can be ignored.
