# Agent notes

## What the network does

Class 1 exactly when the two inputs are nearly equal: `abs(x1 - x0) < 0.2`.

## How I got there

- Six hidden units have small weights (|w| < 0.8) and moderate biases. Over the whole input box
  their combined contribution to the output is a nearly flat negative offset (-7.4 to -5.4).
  Its only job is to keep the output below 0 away from the band.
- Two units are steep (|w| about 11-12) and both have weights in the direction x1 - x0, with opposite
  signs on equal-and-opposite coefficients:
  - h1_7 = sigmoid(12.3*(x1 - x0) + 2.41) switches on at x1 - x0 > -0.196 (output weight +14.5);
  - h1_3 = sigmoid(11.4*(x1 - x0) - 2.44) switches on at x1 - x0 > +0.214 (output weight -16.5).
  Together they make a bump (+14.5 minus about 6.4 offset, so positive) on the band
  -0.2 < x1 - x0 < 0.2, and negative everywhere else.
- Root-finding along x0 = const gave the real boundaries at x1 - x0 from -0.203 to -0.200 (lower)
  and +0.195 to +0.209 (upper), drifting slightly with x0 because of the flat units. Both
  snap to one shared half-width of 0.2.
- `abs(x1 - x0) < 0.2` passes the checker as PASS (snap): 0 disagreements where the net is
  confident, and 0.91% of inputs differ only in the unsure zone along the band edges.
- PASS (exact) would mean keeping all 8 sigmoids, including the drift of about 0.01 in the band
  edges. That drift is a training artifact, so I did not keep it.

## Heuristics

- **Look for steep units first.** In a sigmoid (or ReLU) net with a binary output, the units with
  large input weights (about 10x the others) carry the decision boundary. Small-weight units with
  large biases are usually a near-constant offset. To confirm, evaluate their sum over the domain
  and check that its range is narrow compared with the steep units' output weights.
- **Pairs of steep units with the same direction and opposite output signs form a band (bump).**
  A unit whose weights are (-a, +a) depends only on x1 - x0. Its boundary sits at -bias / a. Two
  such units, one with a positive and one with a negative output weight, give
  lo < x1 - x0 < hi. A symmetric band becomes `abs(x1 - x0) < c`.
- **Snap shared thresholds.** If the two edges of a band are about ±c (here -0.196 / +0.214 from the
  unit biases, and ±0.20 measured), use one symmetric value c, rounded to a simple decimal.
- **Measure the real boundary, not just the unit's switch point.** The constant offset moves where
  the output crosses 0. Root-find the net output along a few lines across the boundary to get the
  actual edges before choosing the snapped constant.
- For a binary output, returning a comparison (1/0) is enough. The checker uses `raw > threshold`,
  so a boolean output of 1 or 0 against threshold 0 works.
