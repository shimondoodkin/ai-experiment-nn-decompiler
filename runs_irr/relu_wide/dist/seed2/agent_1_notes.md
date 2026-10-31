# Notes

## What I did

- Evaluated the 2 -> 64 -> 1 ReLU net on the integer grid of [-3, 3]^2 (scratch/grid.py).
  The values matched sqrt(x0^2 + x1^2) to about 0.01 everywhere (e.g. (1,1) -> 1.41, (1,2) -> 2.23,
  (3,3) -> 4.23), symmetric under sign flips and under swapping x0 and x1.
- The hidden units are mostly bias-free ReLUs of directions spread around the circle
  (relu(w . x) with |b| ~ 0), with roughly equal output weights. A sum of relu(u_k . x) over directions
  u_k gives a polygon approximation of the norm. The remaining units have small coefficients or
  near-zero weights (h1_17, h1_50 are ~1e-9, dead). They only fine-tune the polygon and the offset.
- Replaced the whole net with `radius = sqrt(x0*x0 + x1*x1)`.
- Check: PASS (snap), max |program - net| = 0.035 (at the origin, where the net has a +0.035 bias
  offset; tolerance is 0.084). PASS (exact) is not reachable with a smooth formula, because the net is
  piecewise linear. Copying its 64 kinks would undo the refactoring.

## Heuristics

- Before reading the weights, evaluate the net on the integer grid of the input box and compare it
  with a few candidate formulas (|a|+|b|, max(|a|,|b|), sqrt(a^2+b^2), a*b, a^2+b^2). A table of
  values usually identifies the function faster than the units do.
- Many bias-free ReLUs whose directions spread around the circle, all with positive output weights of
  similar size, point to a convex positively homogeneous function: a norm. Equal weights at all angles
  mean the Euclidean norm. Weights only on axis directions mean L1, weights only on diagonal directions mean L-inf.
- Units with weights ~1e-9 are dead. Units with tiny output weights (<0.005), or units whose kink lies
  far out with a small slope, are fine-tuning and can go once a formula is found.
- A ReLU net cannot be exactly zero at a norm's apex. Expect the largest error at the
  non-smooth point (here the origin), and compare it with the snap tolerance instead of
  adding correction terms.
