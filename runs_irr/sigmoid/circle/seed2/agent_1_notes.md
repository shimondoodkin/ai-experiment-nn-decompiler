# Notes

## What I changed

- The dump is a 2 -> 16 sigmoid -> 1 binary classifier. An ASCII map of the decision over the
  [-3, 3]^2 box showed one round blob centred at the origin.
- 34.6% of a 601x601 grid is class 1, so the area is 0.3464 * 36 = 12.47 ≈ pi * r^2, which gives r ≈ 1.99.
  Bisecting the boundary along 72 rays gave a radius between 1.971 and 2.020 (mean 1.996).
- The 16 sigmoids are edges of a soft polygon around that circle. Most have |w| ≈ 4.5 and a bias of
  about ±8.5, so each edge sits at distance about 2 from the origin. Units 1 and 13 have weights near 0,
  so they only add a constant. The output layer sums the edge "outside" penalties against a positive
  bias. All of that is the polygon approximation of a disk.
- The program is `4 - (x0*x0 + x1*x1)`: inside the circle of radius 2. It gives `PASS (snap)`. Every
  disagreement is in the net's unsure zone, right on the boundary where the polygon's edges and
  corners differ from the true circle by up to about 0.03 in radius.
- I did not try for `PASS (exact)`. Copying the net's polygon wobble would bring back all 16 units
  and would not be a cleaner program.

## Heuristics

- For 2-input binary nets, plot the decision on a grid first. The shape (disk, band, half-plane,
  quadrant) often names the formula right away.
- Estimate the size of a round region from the class-1 area fraction (area = pi r^2), then confirm
  it by bisecting the boundary along rays. If the radius is nearly constant, it is a circle. Snap r^2 to
  an integer.
- A ring of hidden units with similar |w| and similar |bias / |w||, whose weight directions are
  spread around the circle, is a polygon approximating a circle of radius |bias| / |w|.
- Units with weights near 0 are constants; fold them into the output bias.
- Write squares as `x*x` when `**` is not allowed. Return a signed margin (`r^2 - x0*x0 - x1*x1`)
  rather than a 0/1 comparison, so the score is meaningful near the boundary.
