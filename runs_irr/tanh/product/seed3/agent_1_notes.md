# Notes

## What the network computes

`x0 * x1` on [-2, 2]^2.

- Every hidden unit's weights are about (a, a) or (-a, a), so the 16 tanh units fall into two groups:
  9 read s = x0 + x1, and 7 read d = x1 - x0. The net is f(s) + g(d) with no cross terms.
- A 9x9 grid of the net matched x0*x1 to within about 0.06. The largest gaps are at the corners
  (net -3.94 where the product is -4.00), where the tanh units start to saturate.
- That fits the identity x0*x1 = ((x0+x1)^2 - (x1-x0)^2) / 4. The s group builds a parabola
  in s, and the d group builds a negative parabola in d.

## Result

`agent.py` is `return x0 * x1`. It passes `PASS (snap)`: max |program - net| = 0.065, with tolerance
0.158 (2% of the output range). An exact pass would need the net's own tanh curve, including its
corner saturation and a constant offset of about +0.005. No simple formula reproduces that, so I
kept the clean formula.

## Heuristics

- Group hidden units by the direction of their weight vector, ignoring scale and sign. If the
  directions fall into a few groups such as (1,1) and (1,-1), the net is a sum of 1-D functions
  of those projections. Examine each one separately.
- If the groups are the two diagonals and the 1-D pieces look like even (U-shaped) functions,
  suspect x0*x1 = ((x0+x1)^2 - (x0-x1)^2)/4. If only one diagonal is present, suspect
  (x0 ± x1)^2. If there is no cross direction, suspect x0^2 or x1^2.
- Before snapping individual coefficients, print the net on a coarse grid of round input values.
  Products, sums and abs values are often obvious from the table, and checking one candidate
  formula is much faster than simplifying 16 units.
- The largest errors of a smooth (tanh/sigmoid) fit sit at the domain corners, where units
  saturate. Judge a candidate formula by its interior fit, and expect a corner error of a few
  percent of the range.
