# Notes

## What the network computes

The dump is 2 -> 16 ReLU -> 1, a regression over x0, x1 in [-3, 3]. The 16 hidden units have
weight vectors pointing in many directions around the circle, with biases close to 0, and all but
one output weight is positive. That is a piecewise-linear approximation of a convex, radially
symmetric cone.

On a 7x7 grid the net's outputs match sqrt(x0^2 + x1^2) to about 0.02 (for example (3,3) gives
4.236 against 4.243, and (1,0) gives 0.998 against 1). The one place they differ is the origin:
there the net gives 0.093 and the radius is 0. A ReLU sum cannot form the sharp tip of a cone,
so the tip is rounded off.

## What I wrote (agent.py)

    radius = sqrt(x0*x0 + x1*x1 + 0.01)     # = hypot(x0, x1, 0.1)

- The plain `sqrt(x0*x0 + x1*x1)` FAILs, but only near the origin: it is off by 0.093 there, and
  the snap tolerance is 0.083.
- Adding a softening constant of 0.1^2 = 0.01 lifts the tip to 0.1. That brings the worst error
  down to 0.042 (half the tolerance), and the check gives `PASS (snap)` on 317,815 inputs.
- I also tried other constants on 400k random points: 0.005 gave a max error of 0.044, 0.02 gave
  0.051, and max(r, 0.09) gave 0.068. 0.01 was the best and is the simplest round value.
- The remaining error is the ReLU net's own faceting: about ±0.03 at large radius, worst near
  (-0.15, 3).

I left all 16 units out. None of them carries separate meaning: they are tangent planes of the
cone.

## Heuristics

- Before reading individual units, evaluate the net on a coarse grid and compare it with
  candidate closed forms (|x|, x*y, x^2, hypot, min/max). A grid that looks the same along a row
  and a column and grows with distance from the centre points to a radial function.
- If a 1-hidden-layer ReLU net has its hidden weight vectors spread around the circle, biases
  near 0 and positive output weights, it is a convex piecewise-linear surface. For an output of 0
  at the origin that grows linearly, try the Euclidean norm.
- ReLU nets round off non-smooth points such as cone tips and kinks: the output there sits a bit
  above the true value. If the clean formula fails only at such a point, add a small smoothing
  term (sqrt(r^2 + eps^2), or sqrt(x^2 + eps^2) in place of |x|). Choose eps close to the net's
  value at the tip and round it to a nice number. A floor such as max(f, c) is usually worse.
- Bin the residual (net minus formula) by the natural variable, here the radius. A flat mean
  residual away from the special point shows the formula is right, and that what is left is
  facet noise, not a missing term.
- When the error is set by facet noise spread over the whole domain, stop at PASS (snap):
  `PASS (exact)` would need the piecewise-linear net itself.
