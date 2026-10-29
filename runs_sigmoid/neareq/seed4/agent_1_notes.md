# Notes

## What the network does
Class 1 iff the two inputs are nearly equal: `|x1 - x0| < 0.2`.

## How I got there
- Six of the eight hidden units (0, 1, 2, 3, 6, 7) have weights of the form `w * (x1 - x0)`
  (the two coefficients agree to within ~0.3%), so they are all functions of one feature
  `d = x1 - x0`. Units 1 and 2 are near-duplicates (same slope ~11.07, same bias ~-2.30).
- Units 1, 2 (rising steps at d ~ +0.21) and unit 3 (a falling step at d ~ -0.20) carry large
  negative output weights (-11, -10, -13): they push the score negative when d is too large
  or too small. Units 0, 6, 7 (soft steps in d with positive weights) and the bias lift the
  middle band above 0.
- Units 4 and 5 have small weights on both inputs and small output weights (0.86, 0.55). They
  only tilt the boundary slightly (the band edges drift by about 0.005 across the domain), so
  I dropped them.
- I scanned the score along d for several fixed values of s = x0 + x1. The sign changes sat at
  d = -0.200..-0.205 and d = +0.195..+0.207 for every s, so the boundary is two parallel lines
  at d = +/-0.2. Snapped both to 0.2 (one shared threshold, symmetric).
- `agent.py` returns `0.2 - abs(x1 - x0)`; it is > 0 exactly in the band.

## Check
`PASS (snap)`, empirical, 285425 inputs: 0 disagreements where the net is confident, 6117
in the net's unsure zone (all on the band edges). An exact match would need the soft
sigmoid tilts, which are training noise rather than meaning.

## Heuristics
- Look at the ratio of the input weights in each first-layer unit. When many units share the
  same direction (here w*(x1 - x0)), rewrite the net as a 1-D function of that feature and
  analyse it along that line.
- Units whose input weights and output weight are both small next to the others mostly tilt
  the boundary a little; drop them first and see whether the check still passes.
- Near-duplicate units (same slope and bias) add up to one unit; with large slopes,
  sigmoid(k*(d - t)) acts as a step at d = t = -bias/slope.
- For a binary output, find the sign changes of the score along the main feature at several
  values of the orthogonal coordinate. If they barely move, the boundary is a set of
  axis-parallel cuts in the feature: snap them to round values, and use abs() when the cuts
  are symmetric (`c - abs(d)` for a band, `abs(d) - c` for outside a band).
- A large negative pair of steps on each side plus a positive bias is the motif for
  "class 1 inside a band" (the net's version of `abs(a - b) < c`).
