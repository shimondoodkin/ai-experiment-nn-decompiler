# Notes

## What I found

The dump is a 2 -> 64 -> 1 ReLU net. Every output weight that matters is positive (the four negative
ones are below 0.013), so the net is a sum of convex ReLU ramps plus a constant: a convex bowl.
Three units (12, 45, 63) have weights near 1e-7 to 1e-2 and are numerically dead.

Sampling a 9x9 grid over [-2, 2]^2 showed a function that is symmetric in x0 and x1 and in their
signs, depending only on the radius:

- f(0, 0) = 0.996 ~ 1
- f(1, 0) = 1.413 ~ sqrt(2)
- f(2, 0) = 2.231 ~ sqrt(5)
- f(2, 2) = 2.992 ~ sqrt(9) = 3

So `out = sqrt(1 + x0^2 + x1^2)`, a hypotenuse with a fixed third side of 1.

## Result

`agent.py`: `sqrt(1 + x0*x0 + x1*x1)`. `check.py` gives PASS (snap), max |program - net| = 0.0166
against a snap tolerance of 0.040. Over 200k uniform points the residual (net - formula) has mean
-0.005 and std 0.003. Its largest values sit at the apex near the origin, where the piecewise-linear net
cannot follow the curve, and the bias grows slowly with the radius. No exact pass is possible without
copying the 64 ramps, and the formula is the readable program.

## Heuristics

- Check the signs of the output weights first. If they are all positive (or all negative) in a single
  hidden ReLU layer, the net is convex (or concave). Then look for a smooth convex formula
  (norms, x^2, exp, sqrt(c + r^2)) rather than a piecewise rule.
- Before reading 64 units, evaluate the dump on a small grid (9x9 over the box) and look for
  symmetries: x0 <-> x1, x -> -x, and radial dependence. A radial bowl points to a function of
  x0^2 + x1^2.
- Read off values at simple points (origin, axis points at integers, the corner) and compare them with
  square roots of integers, squares and exponentials. Values like 1, 1.414, 2.236 and 3 at r^2 = 0, 1,
  4, 8 identify sqrt(1 + r^2) directly.
- Units whose weights are all below about 1e-2 and whose output weight is tiny are dead. Ignore them
  when reading the structure.
- A many-unit ReLU net that approximates a smooth function will never pass exact against the
  formula. Measure the residual (mean, max, where it is largest); if it is well inside the snap
  tolerance and has no structure beyond a small bias and the apex error, the formula is the answer.
