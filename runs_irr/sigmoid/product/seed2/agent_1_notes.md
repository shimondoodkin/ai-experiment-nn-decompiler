# Notes

## Result

`agent.py` is `return x0 * x1`. `check.py`: PASS (snap), empirical over 252,970 inputs, max |program - net| = 0.154
against a snap tolerance of 0.156.

## What I did

1. **Read the weights.** All 16 sigmoid units have |w0| ≈ |w1|. Nine lie along u = x0 + x1 (same sign on both
   weights) and seven along v = x0 - x1 (opposite signs). So the net is f(u) + g(v): a sum of two 1-D curves on
   the diagonals.
2. **Printed the net on a 9×9 grid.** The table was antisymmetric with value ≈ x0·x1 (for example (-2,-1.5) → 3.01,
   (1,1) → 1.00, (-1,0.5) → -0.49). This matches the diagonal structure: x0·x1 = (u² - v²)/4, so the u units build
   +u²/4 and the v units build -v²/4.
3. **Checked x0·x1.** It passes. The error is 0.012 or less for |xi| ≤ 1, 0.026 or less for |xi| ≤ 1.8, and mean
   0.0066 over the box. The only large errors are at the four corners (|u| or |v| near 4, net ≈ ±3.85 where the
   formula gives ±4). The net flattens there because its sigmoid sum can't keep a parabola going to the edge of
   its range.
4. **Did not fit the corners.** A clamp or extra correction terms would follow the training artifact, not the
   function. The checker counts the corner points as inside the unsure zone.

All 16 units and the bias are gone. No constants or fixtures were found beyond the product.

## Heuristics

- **Look for shared weight directions first.** If every unit's weight vector is parallel to one of a few
  directions (here (1,1) and (1,-1)), the net is a sum of 1-D functions of those projections. Rewrite it in
  u = x0+x1, v = x0-x1, and so on.
- **Diagonal pair → product.** Units on (1,1) and (1,-1) that build even curves of opposite sign mean
  x0·x1 = ((x0+x1)² - (x0-x1)²)/4. The same pattern with matching signs would be x0² + x1².
- **Print a coarse grid before simplifying unit by unit.** On a 2-input net a 9×9 table often shows the formula
  outright (sign pattern, symmetry, values at integer points), and is faster than merging 16 sigmoids.
- **Expect edge saturation in sigmoid nets that fit polynomials.** Errors that grow only at the corners of the box
  are a sign the formula is right. Don't add terms to chase them.
- **Check the error by region.** Max error on the inner box against the full box tells a correct formula with edge
  artifacts apart from a wrong formula.
