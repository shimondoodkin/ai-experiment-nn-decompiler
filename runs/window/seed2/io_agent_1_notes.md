# Notes

## What the function is
One float input x0 in [-2, 2], binary output (score > 0).
The decision is a window test: **class 1 iff -1/2 < x0 < 1/2** (i.e. |x0| < 1/2).

Raw score shape (from sorting the samples by x0 and looking at finite-difference slopes):
- x0 < ~-0.40: linear, slope +55.52, zero crossing at x0 = -0.5004
- ~-0.40 .. ~0.42: flat plateau at 5.6805 (a clipped max)
- x0 > ~0.42: linear, slope -69.23, zero crossing at x0 = +0.5004
So score ~= min(5.68, 55.5*(x0+0.5), 69.2*(0.5-x0)); only the sign matters.

## What I wrote and why
`agent.py` returns the decision `-0.5 < x0 < 0.5`. The two fitted zero crossings
(-0.50043, +0.50046) are symmetric and within 0.1% of +-1/2, so both were snapped to one
shared half-width of 1/2. The slopes and plateau height are irrelevant to the binary decision,
so they were dropped.

Result: `PASS (snap)` - 0 disagreements where the function is confident; 40 disagreements, all
in the sliver 0.5 <= |x0| <= 0.50045 inside the unsure zone. Using the unsnapped half-width
0.50045 gives `PASS (exact)`, but I kept 1/2 as the readable, intended constant.

## Heuristics
- For a 1-D input, sort samples by the input and print where the class flips; the flip points
  are the thresholds. Then fit a line to each side of each flip to get the exact zero crossing.
- Print finite-difference slopes along the sorted input and report only where the slope
  changes: this exposes the piecewise-linear (ReLU) structure - kinks, plateaus (clipping via
  min/max), and the linear pieces.
- A plateau in the score (slope 0) means a min/max clamp; for a binary output it usually does
  not affect the decision and can be dropped.
- Two thresholds that are mirror images (-a, +a) indicate an abs(x) < a / window test; use one
  shared constant for both.
- Snap thresholds to the nearest simple value (here 1/2) when the fitted value is within ~1% and
  the checker reports the residual disagreements only in its unsure zone.
- When only the decision matters, return the comparison directly rather than reconstructing the
  score's slopes; the slopes can differ per side (55.5 vs 69.2) without changing the decision.
