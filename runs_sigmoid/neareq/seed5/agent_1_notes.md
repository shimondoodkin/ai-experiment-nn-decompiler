# Notes

## What I did
- Five of the eight sigmoid units (h1_0, h1_2, h1_3, h1_4, h1_6) have input weights of the form
  w*(x0 - x1) with |w| about 7.4-10, so they are soft steps in the single variable d = x0 - x1.
  Their thresholds (-bias/w) sit at d = -0.231, +0.197, +0.197, -0.252, -0.229: two groups, one near
  d = -0.2 and one near d = +0.2.
- The other three units (h1_1, h1_5, h1_7) have small weights (< 1) and large positive biases; over
  [-2, 2]^2 they are slowly varying and act mostly as a constant offset, only nudging the boundary
  by a few thousandths.
- Scanning the net along lines x0 + x1 = const showed the sign change at d = -0.199..-0.207 and
  d = +0.198..+0.205 everywhere: the positive class is the band |x0 - x1| < 0.2.
- Final program: `return 0.2 - abs(x0 - x1)`. Checker: PASS (snap), 0 confident disagreements; all
  disagreements are in the net's unsure zone along the band edges (the net's edges drift by up to
  0.007 because of the slow units).
- I did not pursue PASS (exact): it would require keeping the slow sigmoid units to reproduce the
  small boundary drift, which is noise rather than structure.

## Heuristics
- Group hidden units by the direction of their weight vector. Units whose weights are (w, -w) or
  (-w, w) depend only on the difference x0 - x1; compute each unit's threshold -b/w on that axis.
- Steep sigmoid units (|w| large relative to the input range) are steps; pairs of rising/falling steps
  with opposite output signs at nearly the same threshold usually form one edge of a band.
- Shallow sigmoid units with large biases are near-constant over the input box; treat them as
  part of the output bias first and only keep them if the check fails.
- For binary nets, directly locate the decision boundary by scanning sign changes along lines
  orthogonal to the suspected direction; snap the found thresholds to a simple value (here 0.2)
  and express the region with abs() for symmetric bands.
- A band |a - b| < t can be written as a single return value `t - abs(a - b)`, which keeps the
  output's sign convention (positive = class 1) and gives a margin-like score.
