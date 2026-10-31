# Notes

## What the network does

One input `x0` in [0.5, 4], eight sigmoid hidden units, one regression output. The output falls
from 1.966 at x0 = 0.5 to 0.254 at x0 = 4, convex and monotone: 0.995 at 1, 0.503 at 2, 0.332 at 3,
0.254 at 4. That is `1/x0`.

## What I did

- Sampled the net on a dense grid (10k points) and fitted `a*f(x0) + b` by least squares for
  candidate shapes: 1/x, 1/x^2, 1/sqrt(x), sqrt(x), log(x), exp(-x). Only 1/x fitted well, with
  a = 0.9994 and b = 0.0009, which snap to a = 1, b = 0.
- Replaced the whole net with `1 / x0`. All eight units and their weights are gone; none of them
  has a separate meaning, they are a sigmoid basis fitted to the curve.
- Residual net - 1/x0 is a small wiggle (|r| < 0.01) across the range except at the left edge,
  where the net undershoots: -0.034 at x0 = 0.5 (the curve is steepest there and the sigmoids
  can't follow it). That is the max error, just inside the snap tolerance (0.0341 vs 0.0342).

Result: `PASS (snap)`, max |program - net| = 0.0341, all on the edge x0 -> 0.5.

I kept the clean formula rather than adding an edge clamp (for example `min(1/x0, 1.97)`) to win
margin: the clamp would only copy the network's fitting error at the boundary.

## Heuristics

- For a 1-input regression net, look at the curve first (values at a handful of points), then
  fit `a*f(x) + b` for a small library of shapes (1/x, 1/x^2, sqrt, 1/sqrt, log, exp, x^2) and
  pick the one with the smallest max error. Snap a and b to nice values and re-check.
- A sigmoid layer with many units of mixed sign and no obvious structure is usually just a basis
  for a smooth curve; don't try to interpret units one by one when a whole-net formula fits.
- Domains that start at a positive lower bound (like [0.5, 4]) hint at functions that blow up or
  are undefined at 0: 1/x, log, sqrt.
- The largest disagreement between a smooth formula and a sigmoid net is typically at the domain
  edge where the target is steepest; the net underfits there. Check the residual at the edges
  before concluding the formula is wrong.
- Read the residual (net - formula): a small oscillation around zero means the formula is right
  and the rest is fitting noise; a systematic trend means a missing term or a wrong shape.
