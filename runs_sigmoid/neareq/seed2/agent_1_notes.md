# Notes

## What the network does
Binary classifier on two floats in [-2, 2]: class 1 when the inputs are nearly equal,
`|x0 - x1| < 0.2`.

## How I got there
- All eight hidden sigmoid units have weights on x0 and x1 that are (almost) equal and opposite
  for the large-weight units (h0, h2, h4, h6, h5: ratios within ~1%), so they depend only on
  `d = x0 - x1`. The remaining small-weight units (h1, h3, h7) are nearly flat over the domain and
  only bend the boundary slightly.
- Steep units are soft steps: h0/h2 step up at d ~ -0.245, h4 steps at d ~ +0.19 (weight -18),
  h6 steps at d ~ -0.23 (weight -13). Together they form a band "on" between about -0.2 and +0.2.
- Scanned the decision boundary along d at fixed x0 + x1 from -3.5 to 3.5: the crossings stayed at
  d = -0.203..-0.209 and d = +0.201..+0.203, i.e. a symmetric band of half-width 0.2 with drift
  under 0.01. Snapped to 0.2.
- Final program: `return 0.2 - abs(x0 - x1)` (positive inside the band), a signed margin rather
  than a 1/0 so the score has a meaningful magnitude.
- Check: PASS (snap), 0 disagreements where the net is confident; the 1.25% that disagree all lie
  in the net's unsure zone along the band edges. PASS (exact) is not reachable with a simple
  formula because the sigmoid boundary is slightly curved/asymmetric (by ~0.005).

## Heuristics
- Look for hidden units whose input weights are equal and opposite (w0 ~ -w1): they depend only on
  the difference x0 - x1 (equal weights: the sum). Rotate coordinates to (difference, sum) and scan.
- For a sigmoid unit `sigmoid(a*z + b)`, the step location is `z = -b/a` and its sharpness is |a|;
  units with |a| * input-range << 4 are nearly linear/constant and can usually be dropped for a
  binary decision.
- A pair of opposite-signed steep steps on the same direction with big output weights forms a band
  (window) detector; read off the two step locations and the output sign to get `lo < z < hi`.
- Locate the actual decision boundary numerically (scan along one direction at several offsets of
  the orthogonal direction) instead of trusting single-unit thresholds; the combined boundary can
  sit away from any one unit's step.
- If the band is symmetric around 0, write it as `abs(z) < c`; return `c - abs(z)` as a signed margin.
- Snap the boundary to a round number (0.2 here) if the measured drift across the domain is
  smaller than the net's unsure zone.
