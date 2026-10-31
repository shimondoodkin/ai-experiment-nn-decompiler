# Agent notes

## What the network does

`net(x0, x1) ≈ x0 * x1` on the box [-2, 2]², clipped near the corners.

## What I changed and why

- Evaluated the dump on a 9×9 grid: the output is zero along both axes, odd in each input and
  grows like the product (values 1, 2, 3, ~3.85 along the diagonal of the box). A least-squares
  fit of `c·x0·x1 + d` gave c = 0.99, d = 0.00, with max grid error 0.12; the other candidates tried
  (|x0±x1|, x0²±x1²) did not fit at all.
- The 16 ReLU units are a piecewise-linear approximation of the product (it has no exact snapped
  form), so all of them were replaced by the formula.
- Plain `x0 * x1` failed by a hair: max error 0.156 against a snap tolerance of 0.155, only at
  the corners (±2, ±2), where the net gives ±3.84–3.89 instead of ±4. Every error above 0.1 sits
  in the corners, so the net saturates there rather than being scaled down overall.
- I clipped the product at ±3.8: PASS (snap), max error 0.122. Other passing variants: clip at
  ±3.9 (max error 0.137), or `0.99 * x0 * x1` (max error 0.116, fewer nodes, but it hides that the
  shortfall is only in the corners).
- PASS (exact) is not reachable with a short program: the net's kinks are not at meaningful
  positions, they only approximate a smooth surface.

## Heuristics

- Before reading weights, print the output on a coarse grid. Zero rows/columns along both axes
  plus sign flips by quadrant point to a product `x0*x1`; fit `c·f + d` for a few candidate
  formulas (product, |x0±x1|, squares, sums) and keep the one with the smallest max error.
- A wide ReLU layer (16 units for 2 inputs) whose weights have no clean pattern, all of similar
  size with mixed signs, is usually a piecewise-linear approximation of a smooth function, not
  a sum of meaningful kinks; look for the smooth formula rather than snapping each unit.
- When the ideal formula fails only at the edges or corners of the input box, it is usually
  because the net under-fits the extreme values (fewer training points there). Add a clip at
  the net's own extreme value, or a small scale factor, rather than giving up the formula.
  Look at where the large errors are to choose between the two: errors only at the corners mean
  saturation (clip); errors growing with magnitude everywhere mean a scale.
- The snap tolerance is 2% of the output range, so a smooth formula that is right in shape can
  still miss by a few percent at the extremes; compute the tolerance first and compare the max
  error with it.
