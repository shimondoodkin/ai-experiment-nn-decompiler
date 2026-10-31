# Notes

## What the network computes

The output is approximately `x0 * x1` on [-2, 2]^2.

- On a 9x9 grid the dump's output matched `x0*x1` to within about 0.1, apart from the corners.
- The hidden units split into two families:
  - 4 units are nearly aligned with `x0 + x1` (units 10, 11, 13, 15). Their output weights are positive,
    so they build a convex function of the sum.
  - Most of the rest are nearly aligned with `x0 - x1` (units 0, 1, 2, 3, 5, 8, 12, 14). Their output
    weights are negative, so they build a concave function of the difference.
  - This is the polarisation identity `x0*x1 = ((x0+x1)^2 - (x0-x1)^2) / 4`, with each square
    approximated by a few ReLU kinks.
- Units 4, 6 and 7 have tiny slopes (|out weight x |w|| < 0.02) and act as constants.
  Unit 9 is mixed and small. None of them are needed.

## What I changed

- I replaced all 16 units with `x0 * x1`. On its own it fails, with a max error of 0.278 against a
  tolerance of 0.154. All the failures were in the far corners: the net gives 3.72 at (2,2), 3.75 at
  (-2,-2) and -3.82 at (2,-2), where the product gives ±4. These corners are rarely seen in training.
- I added two corner corrections with snapped values: `-0.2*relu(|x0+x1| - 3)` for the (2,2) and
  (-2,-2) corners, and `+0.1*relu(x0-x1 - 3)` for the (2,-2) corner. The (-2,2) corner is already
  within tolerance.
  - I searched knot ∈ {2.5, 3, 3.5} and coefficients in steps of 0.05.
  - Knot 3 with 0.2 / 0.1 is one of the best choices and has round numbers.
- Result: `PASS (snap)`, max |program - net| = 0.1415 against a tolerance of 0.1538. What is left is
  the net's ripple of about ±0.1 from its piecewise-linear pieces, which no smooth formula can follow.
  I did not aim for an exact pass, because that would mean keeping the kinks themselves.
- Things that did not pass:
  - A pure additive piecewise-linear fit F(x0+x1) + G(x0-x1) with integer knots
    (error 0.176 to 0.27). The net is not exactly additive in the two diagonals.
  - A plain clamp of the product.

## Heuristics

- Evaluate the dump on a coarse grid of the input box first. A table of 9x9 values is often enough to
  recognise a classic function (product, abs, max, sums).
- Rewrite each unit's direction in rotated coordinates (`x0+x1`, `x0-x1`, or more generally in the
  candidate basis). Units grouped by direction, with output weights of one sign per group, reveal
  convex/concave 1-D pieces. Convex in the sum and concave in the difference means a product
  (polarisation identity).
- A unit whose `|output weight| * |input weight vector|` is tiny compared with the others is a
  constant or noise over the box. Drop it.
- When the ideal formula fails only at the corners or edges of the domain, the net is extrapolating
  badly where the training data was thin. Model that with small `relu(|direction| - k)` corrections at
  a round knot, rather than giving up the formula.
- Find the residual (net - formula) on a grid. Fix only the cells beyond tolerance, and check that
  the remaining ripple is below tolerance before adding more terms.
- Grid-search snapped corrections (round knots, coefficients in steps of 0.05) against a dense grid
  before calling the checker. Keep the simplest one that passes with some margin.
