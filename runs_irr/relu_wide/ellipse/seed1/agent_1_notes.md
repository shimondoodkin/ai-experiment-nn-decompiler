# Notes

## What the net does
The dump is a 2 -> 64 -> 1 ReLU net with a binary output (class 1 when the score > 0) on x0, x1 in [-3, 3].
An ASCII plot of the decision on a 31x31 grid showed one convex blob of class 1, wider along x0 and
shifted toward positive x0. Bisecting the boundary along 720 rays from the origin and fitting a general
conic `x^2 + B y^2 + C xy + D x + E y + F = 0` by least squares gave

    x0^2 + 3.958 x1^2 + 0.018 x0 x1 - 1.010 x0 - 0.021 x1 - 3.727 = 0

That is (x0 - 0.505)^2 + 3.96 x1^2 = 3.98. Snapping it (no cross term, centre 0.5, B = 4, right side 4)
gives the ellipse (x0 - 0.5)^2 / 4 + x1^2 = 1. The boundary's extent (x0 from -1.49 to 2.50, x1 from
-1.00 to 1.00) agrees with it.

## Program
`agent.py` returns `1 - ((x0 - 0.5)/2)^2 - x1^2`: positive inside the ellipse centred at (0.5, 0) with
semi-axis 2 along x0 and 1 along x1. All 64 hidden units are gone. Most of them are the net's
piecewise-linear approximation of the curved boundary. A few are dead or near-constant (for example
unit 48, which has weights around 1e-6) or have large positive bias. Those set the confident score
inside the blob.

`check.py`: PASS (snap), 0 disagreements where the net is confident. There are 3813 disagreements in
the unsure zone, all within about 0.004 of the ellipse boundary. An exact pass is not possible with a
curved formula, because the net's boundary is a polygon.

## Heuristics
- For a binary net on 2 inputs, first print an ASCII map of the decision over the box. The shape class
  (half-plane, strip, blob, ring) is usually visible at once.
- For a closed blob, bisect the decision boundary along many rays from a point inside it, then fit a
  general conic by least squares (normalise the x^2 coefficient to 1). A small cross term and near-integer
  ratios mean an axis-aligned ellipse or circle. Complete the square to read off the centre and the
  semi-axes, then snap them to simple values (halves, small integers).
- Cross-check the snapped conic against the boundary's min/max extent along each axis: centre =
  (min+max)/2, semi-axis = (max-min)/2.
- Many units in a wide ReLU net that draws a curved boundary are just facets of the polygon. Do not try to
  simplify them one by one. Replace the whole net with the smooth formula, and expect PASS (snap), not
  exact.
- Return a signed score (1 - quadratic form) instead of a 0/1 comparison. The decision `raw > 0` is the
  same, and near the boundary the checker sees small magnitudes, which fall in the unsure zone.
