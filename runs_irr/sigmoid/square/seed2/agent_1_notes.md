# Notes

## What I did

- The dump is a 1-input net with 8 sigmoid units, a regression output and x0 in [-2, 2].
- I tabulated the net at steps of 0.1. Its values are 0.004 at 0, 1.005 at ±1 and 3.96 / 3.97 at ±2.
  The curve is symmetric and grows as a parabola, so x0².
- The sigmoid units do not mean anything one by one. They are a smooth basis that the net fits into
  a parabola (two large units at the edges, h1_1 on the left and h1_2 on the right, give the steep
  ends, and the smaller ones shape the middle). So I replaced the whole sum with the formula and did
  not snap it unit by unit.
- On a dense grid, max |net - x0²| = 0.041, at x0 = -2, where the net falls short of 4 by 0.04. The
  mean error is about -5e-5. The snap tolerance is 0.079, so the error is about half of it.
- `agent.py` returns `x0 * x0`. Check: PASS (snap), empirical, 227493 inputs, max error 0.0410.
  PASS (exact) would need the full 8-sigmoid sum, which brings back the whole dump, so I did not try
  for it.

## Heuristics

- For a 1-input net, tabulate the output on a fine grid before reading the weights. Values at the
  integers and half-integers (0, ±1, ±2) often show the formula right away (here 0, 1, 4 means x²).
- Check symmetry: f(-x) ≈ f(x) points to an even function (x², |x|, cos), f(-x) ≈ -f(x) to an odd one.
- A sigmoid net with many units of mixed sign that adds up to a smooth curve is a basis expansion.
  Fit candidate closed forms to the whole output, not to single units. Snapping the units one by one
  keeps all the clutter.
- Fit errors are largest at the edges of the domain, where the net has few units to bend the curve.
  Read the max error's location: if it is at a bound and the mean error is near 0, the formula is
  right and the net is the inexact one.
- Compare the max error with the snap tolerance (2% of the output range). A clean formula that uses
  about half of it is a better result than a fitted polynomial that reaches exact.
