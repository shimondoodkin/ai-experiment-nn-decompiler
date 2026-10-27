# Notes

Network: 2 inputs (x0 in [1, 6.9], x1 in [0.1, 2.5]) -> 8 ReLU -> 3 logits, argmax.
(Looks like iris petal length / petal width.)

## What I did
- Did not try to simplify the 8 hidden neurons algebraically. Instead I rendered the argmax class map
  on a grid: three bands ordered along the x0 + x1 diagonal (0 low, 1 middle, 2 high).
- Bisected the class boundaries at many x1 values to get the boundary x0 as a function of x1.
  - 1/2 boundary: x0 + x1 stayed in 6.58..6.69 over the whole range -> `x0 + x1 > 6.65`.
    (6.6 and 6.7 both fail the check; 6.65 = midpoint passes.)
  - 0/1 boundary: kinked. For x1 <~ 1.4 slope dx0/dx1 ~ -1.32 (x0 + 4/3 x1 ~ 3.5); above it slope ~ -0.9
    (x0 + x1 ~ 3). The actual boundary is the max of the two lines, i.e. class 0 is the union of
    the two half-planes: `x0 + 4*x1/3 < 3.5 or x0 + x1 < 3`. A single line (slopes 1.2, 1.25, 4/3)
    always left confident disagreements near the kink.
- Wrote it as a 3-way decision list: class 0 test first, then class 2, else class 1. The class-0 and
  class-2 regions do not touch, so the order is safe.
- Result: `PASS (snap)`, 0 confident disagreements.

## Heuristics
- For argmax nets with few inputs, skip the weights: plot the class map on a grid, then bisect the
  boundaries along one axis and fit lines to the boundary points. A low-input classifier is usually
  just a few half-planes.
- Check whether `boundary + k*other_input` is constant for a simple k (1, 4/3, 1/2...). If it drifts,
  look for a kink (two slopes) rather than forcing one line.
- A kink of a ReLU boundary is a max/min of two lines -> write it as `or`/`and` of two half-planes
  (union when the region is convex-outward toward the corner, intersection otherwise; test via which
  line gives the larger/smaller boundary value on each side of the kink).
- Emit a decision list ordered by the regions that are cleanly separable; the middle class is the
  fallback `return`.
- When snapping a threshold, try the round value and the half-step neighbours; the midpoint of the
  observed range of the boundary constant is often the only one that passes.
- Unsure-zone disagreements are fine for PASS (snap); concentrate on removing "net confident" ones,
  which point exactly to the badly fitted segment.
