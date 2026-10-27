# Notes

## What I did
- spec: two float inputs in [-2, 2], binary output (class = out > 0). Only 11% of samples are positive.
- Plotting class on a 20x20 grid showed the positives sit on a thin band along the diagonal x1 = x0.
  So I rewrote the inputs as difference d = x0 - x1 and sum s = x0 + x1.
- Positives have |d| up to 0.2036; negatives start at |d| 0.2025 (x1 > x0 side) and 0.2033 (x0 > x1 side).
  The boundary does not depend on s. A least-squares fit of out on [|d|, d, s, |s|, 1] gives roughly
  out = 13 - 73*|d| (the s terms are about 0), so the score is a symmetric tent around d = 0.
- I snapped the threshold to 0.2. The function is "approximately equal": `abs(x0 - x1) <= 0.2`.
  3 of 2000 samples fall in the 0.200-0.204 sliver and disagree.
- check.py: PASS (snap). The only disagreements (436 of 240k) sit right on |d| = 0.2, inside the
  function's unsure zone. The net's own boundary is about 0.2025, slightly wider than 0.2, so
  getting PASS (exact) would mean un-snapping the threshold. I did not do that.

## Heuristics
- Before fitting anything, draw the class map on a coarse grid. Thin bands, wedges and stripes show
  which input combinations matter (here x0 - x1).
- Try sum and difference coordinates (and their absolute values) for pairs of inputs on the same
  scale. A band along a diagonal means |x_i - x_j| < tol, an approximate-equality test.
- Find the threshold from the largest |feature| among positives and the smallest among negatives.
  Snap it to a round value inside or next to that gap (0.2 here).
- Fit the raw score linearly on candidate features (|d|, d, s, |s|, 1). Coefficients near 0 rule a
  feature out, and a tent shape (large negative weight on |d|) confirms a symmetric distance test.
- When the only mismatches are on the snapped boundary and the checker calls them unsure, the snap
  is right. Do not chase the net's slightly-off threshold.
