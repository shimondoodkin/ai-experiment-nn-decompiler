# Notes

## What I did
- Evaluated the net on a 9x9 grid over [-2, 2]^2 (scratch/grid.py). The table was exactly
  antisymmetric in each input and bilinear: f(-2,-2) ~ 3.96, f(-2,2) ~ -4.01, f(1,1) ~ 0.98,
  f(0, x1) ~ 0 and f(x0, 0) ~ 0. That is the product x0 * x1.
- The hidden layer agrees: almost all 64 units have weight directions near (1, 1)/(-1,-1) or
  (1,-1)/(-1, 1), with spread-out biases. They are piecewise-linear approximations of
  (x0 + x1)^2 and (x0 - x1)^2, and x0*x1 = ((x0+x1)^2 - (x0-x1)^2) / 4.
- Units 7, 20, 26, 50, 52 and 61 have tiny weights (dead or near-zero contribution) and are dropped.
- Final program: `return x0 * x1`. Check: PASS (snap), max |program - net| = 0.06 (tolerance 0.16),
  largest errors at the corners (2, 2), where the ReLU approximation of the parabola saturates.
  Exact pass is not possible: the net is a piecewise-linear chord approximation of a smooth surface.

## Heuristics
- Before reading 64 units one by one, tabulate the net on a coarse grid (integer and half-integer
  points). Symmetries (f(-x0, x1) = -f(x0, x1), zero along both axes) identify products and similar
  formulas at once.
- Many first-layer units sharing two directions, (1, 1) and (1, -1), with biases spread over the
  range, are a ReLU spline of a function of x0 + x1 and of x0 - x1. Opposite signs of output weights
  on the two families plus convexity suggest the polarisation identity x*y = ((x+y)^2 - (x-y)^2)/4.
- Units whose weights and output weight are ~1e-3 or below are dead; delete them first.
- For a smooth target fitted by ReLUs, the error is largest at the box corners and edges (fewest kinks
  there); expect PASS (snap), not exact, and check that the max error sits well below the snap
  tolerance rather than trying to fit the kinks.
- Try the clean formula before any fitted scale factor: here 1.0 * x0 * x1 already passed with
  a 2.7x margin.
