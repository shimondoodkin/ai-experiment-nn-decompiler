# Notes

## What the net does

`net(x0, x1) = sqrt(x0^2 + x1^2)`: the Euclidean distance of the point from the origin.

## How I found it

- Read the dump. About 20 of the 64 hidden units have weights of 1e-5 or less (dead units). About 30 of the
  live units have biases near 0 (|b| < 0.05) and point in many different directions. Every live unit's output
  weight is positive, and the output bias is near 0. A sum of `relu(w.x)` through the origin with positive
  weights is convex and positively homogeneous, so it is a norm-like function: |x0|+|x1|, max(|x0|,|x1|) or the
  Euclidean norm.
- Ran the net on circles of radius 0.5, 1, 2 and 3. The output equalled the radius within about 0.008 in every
  direction, and on the diagonal it gave 1.414 at (1, 1), so the function is the Euclidean norm.
- The units with larger biases (for example -0.8 and -0.55) have tiny output weights (about 0.001) and only
  correct the piecewise-linear approximation slightly.

## Result

`agent.py` is `sqrt(x0 * x0 + x1 * x1)`, which gives PASS (snap). The max |program - net| is 0.034 at the
origin, where the net has a small positive offset (the polygon approximating the cone does not reach the tip).
Elsewhere the error is under about 0.01. An exact match would need the net's 40-odd linear pieces, which add no
meaning.

## Heuristics

- Look at the output layer first. If every significant output weight is positive and most hidden biases are near
  0, the function is convex and close to positively homogeneous (f(t*x) = t*f(x)), so try norms: L1, L-infinity
  and L2.
- Tell the norms apart by the value at a unit diagonal point: L1 gives 2, L2 gives 1.414, L-infinity gives 1. Then
  check that the output is constant on circles (L2) or on squares (L-infinity).
- Many live first-layer units with directions spread around the circle, rather than aligned to the axes, point to
  a rotation-invariant function (L2). Axis-aligned units point to L1 or L-infinity.
- Units whose input weights or output weights are about 1e-5 or less are dead and can be dropped at once.
  Units with large biases and output weights of about 1e-3 are fitting corrections.
- Expect the largest error at the kink of a cone (here the origin). A ReLU net cannot reproduce the sharp tip, so
  a small offset there is approximation error, not part of the formula.
