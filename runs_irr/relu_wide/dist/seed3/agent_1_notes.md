# Notes

## What the network computes

The net is 2 -> 64 ReLU -> 1, regression, inputs x0, x1 in [-3, 3].
Final program: `sqrt(x0*x0 + x1*x1)`, the Euclidean distance from the origin. Check: `PASS (snap)`,
max |program - net| = 0.030 (at the origin, where the net gives 0.03 instead of 0), snap tolerance 0.084.

## How I got there

1. Ranked the hidden units by |output weight| * |input weight vector|. About 30 units matter. They all
   have bias near 0 and positive output weights of about 0.12-0.25, and their weight directions point
   all around the circle. The remaining ~34 units are dead or nearly dead: weights ~1e-4 or smaller,
   or a large negative bias that keeps them almost always off, with output weights ~1e-3.
2. A sum of `c_i * relu(w_i . x)` with zero bias, positive c_i and directions spread over every angle
   is a piecewise-linear approximation of a convex, positively homogeneous function: a norm.
3. Evaluated the net on a 7x7 grid and compared it with sqrt(x0^2+x1^2). They match to about 0.01
   everywhere, apart from the constant offset of ~0.03 at the origin (the output bias 0.0126 plus a few
   units with small positive biases).
4. Wrote the formula and checked it. Every unit, coefficient and threshold is replaced by the formula,
   so there is nothing to snap. An exact pass would need the 30-piece polygonal approximation, which
   is less readable and only matches the net's training error.

## Heuristics

- Sort units by |v_i| * ||w_i|| (output weight times input weight norm). The units with tiny products
  are dead and can be dropped together.
- Many ReLU units with bias ~ 0, output weights of the same sign and directions spread around the full
  circle (or sphere) mean a convex, homogeneous function of degree 1: a norm. Check the L2 norm
  `sqrt(sum x_i^2)` first. If the directions cluster on the axes or diagonals, check L1 (`abs` sums)
  or Linf (`max(abs(..))`) instead.
- Before analysing units one by one, evaluate the net on a coarse grid and print it. Symmetric
  patterns (equal values at (±a, ±b) and at (a, b) vs (b, a)) and familiar numbers (1.414, 2.236,
  3.162, 4.243) point straight at the formula.
- A small constant offset at a kink (here the net gives 0.03 at the origin) is approximation error of
  the net. Don't add it to the formula. The snap tolerance absorbs it.
