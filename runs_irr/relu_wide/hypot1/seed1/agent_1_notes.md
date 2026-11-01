# Agent notes

## What I did

- The dump is one hidden layer of 64 ReLUs on 2 inputs in [-2, 2], one regression output.
  Eight units have weights around 1e-7 or smaller (h1_0, 7, 30, 33, 47, 48 and a few others with
  tiny output weights, such as h1_8 and h1_20): they are dead or negligible.
- Every output weight that matters is positive and the hidden units' directions point all around
  the circle. A sum of ReLUs with positive weights is convex, so the function is a convex bowl.
- Tabulating the net on a 9x9 grid showed a symmetric bowl: about 1.0 at the origin, about 2.24 at
  (±2, 0) and (0, ±2), about 3.0 at the corners (±2, ±2). Those are sqrt(1 + x0² + x1²)
  (1, sqrt 5, 3).
- Least-squares fits of `a*f + b` on a 201x201 grid: sqrt(1 + x0² + x1²) gave a = 0.9988,
  b = 0.0018, max error 0.019. Plain radius, squared radius, |x0| + |x1| and a Gaussian were all
  far worse (max error 0.28 or more).
- Snapped to a = 1, b = 0: `return sqrt(1 + x0*x0 + x1*x1)`.
- `check.py`: PASS (snap), max |program - net| = 0.0197, tolerance 0.040. The largest errors are
  at the edges of the box (for example (-2, -2): net 2.981, formula 3.0). The piecewise-linear net
  undershoots the curved surface there.
- I did not aim for PASS (exact): that would mean keeping about 55 live ReLU units, which is
  the same net with the dead units removed and tells you nothing more.

## Heuristics

- One-layer ReLU net with all (significant) output weights positive and hidden directions spread
  over the circle: the function is convex. Try radial/norm-like formulas first.
- Tabulate the net on a coarse grid before reading the weights. Symmetry (x -> -x, x0 <-> x1)
  and values at the origin, edge midpoints and corners usually identify the formula quickly
  (here 1, sqrt 5, 3 at radius² 0, 4, 8 gives sqrt(1 + r²)).
- Fit `a*f(x) + b` for a handful of candidate features (r, r², sqrt(1 + r²), L1 norm,
  exp(-r²)) by least squares, and compare max errors. A fit with a ≈ 1, b ≈ 0 means the
  formula is right as written; snap it.
- Units with all weights below about 1e-4 are dead or negligible; drop them before any analysis.
- A ReLU net approximating a smooth curved surface has its largest error at the corners of the
  input box, where the training data is sparsest. Expect snap rather than exact there, and judge
  the formula by the fit coefficients, not by those corner errors.
