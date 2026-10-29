# Notes on m55dec4 (2 inputs, 8 sigmoid units, regression)

## What the network computes

Every hidden unit has weights of almost equal size on x0 and x1, so each unit reads one of two
directions only:

- difference `d = x0 - x1`: units 0, 4, 5, 7 (weights like (1.56, -1.58), (-0.81, 0.81));
- sum `s = x0 + x1`: units 1, 2, 3, 6 (weights like (-2.23, -1.84), (1.76, 1.94)).

Probing the net along these axes shows it is essentially a smooth, slightly asymmetric
`|x0 - x1|`: about 0.3-0.4 at d = 0, about 2.9 at d = +3 and 3.1 at d = -3, about 5.2 at d = ±6.
The sum units only add a small bump (about +0.09) near s = 0. Their output weights are small
(0.16, 0.25, 0.07, -0.009).

A plain `abs(x0 - x1)` is off by about 0.3 at d = 0. The snap tolerance is 0.098, so it fails.

## What I changed

1. Rewrote all units in terms of `d` and `s`.
2. Replaced the four d-units with three sigmoid ramps in d. Unit 4 (weight -0.56) is not needed
   once the others are refit. Hidden slopes and offsets are snapped to a 0.25 grid, except one
   slope of 1.6.
3. Replaced the four s-units with one symmetric pair, `sigmoid(2s-1) + sigmoid(-2s-1)`.
   Unit 1 (weight -0.009) is dropped as noise.
4. Picked the output weights with a minimax LP on a 61x61 grid. I then searched a grid of
   0.05-step values for the coarsest set that still has max error < 0.065.

Result: `PASS (snap)` with max |program - net| = 0.058 (tolerance 0.098). There are 4 sigmoids
instead of 8.

A simpler d-only version also passes, with max error 0.086 and less margin:
`3*sigmoid(1.9d-7) + 3*sigmoid(2.1d-2.5) + 6.5*sigmoid(-0.7153d-1.7) - 0.8529`. It is very
sensitive to the slope of the large-weight unit. The best possible d-only fit has error 0.066,
because the net varies by up to 0.13 along s at fixed d. That leaves little room for snapping.
Adding the s-bump brings the best-fit error down to 0.028, which gives room to snap.

An exact pass would need practically all 8 original units, so I did not attempt it.

## Heuristics

- **Look for shared input directions first.** If a unit's weights on two inputs are nearly equal
  or nearly opposite, rewrite it in terms of `x0+x1` and `x0-x1`. Then probe the net on that
  rotated grid (fixed d, varying s, and the reverse). This usually shows the function is mostly
  1-D, here a soft `|x0-x1|`.
- **Estimate the 1-D floor before snapping.** At each fixed d, take the min and max of the net
  over s. Half the largest spread is the best error any d-only program can reach. If it is close
  to the snap tolerance, keep a correction term for the other direction.
- **Refit the output layer after changing hidden units.** It is linear in the output weights,
  so a minimax LP (`linprog` minimising t subject to |A w - y| <= t) gives the best weights. Do this
  instead of snapping each weight separately; snapping one at a time compounds errors quickly.
- **Separate snapping hidden and output parameters.** Snap hidden slopes and offsets by searching
  a small grid of nice values, solving the output LP for each combination. Then search the output
  weights on a fine grid (0.05) and score each set by how coarse its values are, under an error
  limit of about 2/3 of the tolerance.
- **Greedy "coarsest first" snapping can throw away a useful term.** It snapped a small weight to
  0 and lost the term that made the snapping possible. Score coarseness together with error, or
  search jointly.
- **Watch large output weights on shallow sigmoids.** A unit like `6.5*sigmoid(-0.7d-1.7)` over
  |d| <= 6 makes the error very sensitive to the slope (0.01 in slope moves the error by about
  0.04). Snap such units early and let the other parameters absorb the change.
- **Symmetric bumps** across a direction can be written as
  `k*(sigmoid(a*s+b) + sigmoid(-a*s+b))`, one shared pair instead of several one-sided units.
- **Drop units with tiny output weights** (here |w| < 0.01) at once. They are below any
  tolerance.
- **Check the probe yourself before trusting a "nice formula" guess.** `abs(x0-x1)` looked right
  but was 3x over tolerance at d = 0, where the net smooths out the corner.
