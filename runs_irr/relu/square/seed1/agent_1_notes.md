# Notes

## What the network does
One input x0 in [-2, 2], 8 ReLU units, regression output.

- Four units face left (negative weight on x0), with kinks at x0 = -0.22, -0.66, -1.10, -1.55.
  Four face right, with kinks at x0 = 0.22, 0.67, 1.11, 1.56.
- The slopes add up as |x0| grows. The kinks are spaced about 0.44 apart and placed symmetrically
  around 0. The output bias is about 0.016. Together they make a piecewise-linear parabola.
- Sampled values match x0^2 to within about 0.04 everywhere: f(±1) = 1.00, f(±1.5) = 2.24,
  f(±2) = 3.96 to 3.97, f(0) = 0.016.

## What I changed
I replaced the whole network with `x0 * x0`. The language has no `**`, so I wrote the square as a
product.

Check: `PASS (snap)`, empirical, 230k inputs. The max |program - net| is 0.037, against a snap
tolerance of 0.079. The largest gaps are at the domain ends (x0 = ±2), where the net undershoots
4 by about 0.04, and near 0, where the net's flat bottom sits at 0.016 and its first kinks are at
±0.22.

An exact pass would need the eight kinks spelled out again. That is the dump itself, so I chose the
readable formula.

## Heuristics
- For a 1-input ReLU net, compute every kink position (x = -b/w), which side each unit faces
  (the sign of w) and its slope contribution (w * v). If the kinks are symmetric around a point
  and the slopes grow steadily outward, the net is a convex polynomial, most often x^2 or |x|.
  Kinks evenly spaced, with the slope increasing by about the same amount at each one, point to a
  quadratic.
- Tabulate the net on a coarse grid next to candidate formulas (x, x^2, |x|, x^3) before fitting
  anything. Recognising a familiar function is quicker than snapping weights one by one.
- Expect a net that approximates a smooth function to miss it most at the domain edges (after the
  last kink, it continues in a straight line) and at the vertex (the flat segment and the small
  bias). Those points tell you how far you are from the snap tolerance.
- When the snap tolerance is roughly twice the approximation error, take the clean formula over a
  kink-by-kink exact rewrite, which only restates the dump.
- `**` is not allowed in this language, so write x*x for a square.
