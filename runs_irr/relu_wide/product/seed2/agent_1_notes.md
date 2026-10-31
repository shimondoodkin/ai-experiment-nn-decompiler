# Notes

## What the network computes

The dump is a 2 -> 64 -> 1 ReLU regression net on x0, x1 in [-2, 2]. Evaluating it on a 9x9 grid
showed a saddle: zero along both axes, about +4 at (-2,-2) and (2,2), about -4 at (-2,2) and (2,-2),
and values that scale linearly in each input when the other is held fixed (e.g. x0=-1 row:
0.99, 0.75, 0.49, 0.25, 0, -0.26, ...). That is the product `x0 * x1`.

## What I changed

- Replaced all 64 hidden units with `return x0 * x1`.
- Units 22, 37, 50, 54 (weights ~1e-12 .. 1e-3) are dead or negligible; the rest are pairs of
  ReLU hinges along the directions x0+x1 (e.g. units 26, 53, 15, 14) and x0-x1 (e.g. units 61, 62,
  8, 9): a piecewise-linear spline for ((x0+x1)^2 - (x0-x1)^2) / 4, the polarization identity for
  the product.
- `check.py`: PASS (snap), max |program - net| = 0.062 against a snap tolerance of 0.159, worst at
  the corners (net 3.94 vs 4.0), where the spline undershoots the parabola.
- PASS (exact) is not reachable with a compact program: a ReLU net is piecewise linear and the
  product is not, so only the full spline would match exactly.

## Heuristics

- Evaluate the net on a coarse grid (for example 9x9 over the box) before reading weights. Many
  64-unit nets are smooth textbook functions, and the grid shows the shape at once.
- Signs of a product: zero on both axes, odd in each input, rows that are linear in the other
  input, corner values equal to +-lo*hi.
- Hidden units with weights in mirrored pairs along the directions (1,1) and (1,-1), with
  output weights of opposite sign between the two families, point to x*y built as
  ((x+y)^2 - (x-y)^2)/4 from two ReLU splines of a square.
- Units whose weights are all below ~1e-3 can be dropped first; they rarely matter.
- For a smooth nonlinear target approximated by ReLUs, aim for PASS (snap) with the clean formula;
  the residual is spline error, largest at the domain corners, and it is not worth encoding.
