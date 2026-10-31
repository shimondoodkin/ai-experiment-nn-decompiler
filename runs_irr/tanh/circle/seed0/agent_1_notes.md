# Notes

## What the network does
Class 1 exactly when the point lies inside the disk of radius 2 around the origin:
`x0^2 + x1^2 < 4`. `agent.py` returns `4 - (x0^2 + x1^2)` and gets PASS (snap). It disagrees
with the net on 2.4% of inputs, all of them in the unsure zone near the circle.

## How I got there
1. **Constant units.** Seven of the 16 tanh units (2, 3, 4, 5, 7, 12, 14) have input weights
   around 0.003 and biases around ±3.5 to 3.9, so they are saturated at about ±0.998 over the
   whole box. Folding them with the output bias gives a constant of -42.07295. That version
   (9 tanh units plus that constant, kept as `scratch/exact_try.py`) gets PASS (exact).
2. **Geometry of the other 9 units.** Each one is a steep line (|w| about 2.6, |b| about 4.4) at
   distance 1.66 to 1.72 from the origin. Their angles are spread around the circle
   (-175, -155, -35, 45, 66, 85, 104, 126, 166 degrees). Their output signs are set so that each
   one adds about +7 on the origin side of its line. That makes a soft polygon around the origin
   that the constant -42 cuts down.
3. **Boundary scan.** Along 24 rays from the origin, the net's decision boundary is at radius
   1.97 to 2.03, mean about 2.007. So it is a circle of radius 2, and I snapped it to exactly 2.
4. The disk formula passes as snap, so I kept it as the answer over the exact 9-unit version,
   because it is the readable one.

## Heuristics
- Units with near-zero input weights and |bias| > 3 are saturated constants. Fold
  `w_out * tanh(bias)` into the output bias. This alone is often exact.
- For a 2-input binary net whose active units are lines at about the same distance from one
  point, with output signs that all favour that point's side, suspect a disk (or a convex
  polygon). Compute each line's distance `-b/|w|` and its angle.
- Find the decision boundary by scanning rays from the suspected centre. A nearly constant
  boundary radius means a circle: snap r to a nice value and return `r^2 - (x0^2 + x1^2)`. For
  binary outputs only the sign matters, so a quadratic score is fine.
- The tangent lines sit inside the true circle (1.7 vs 2). Their tanh softness pushes the
  boundary outward, so read the radius from the boundary scan, not from the line distances.
- Keep a mechanically folded version as an exact fallback, and use the closed-form formula when
  it passes snap.
