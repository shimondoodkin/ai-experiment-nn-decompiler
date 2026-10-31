# Notes

## What the network computes

A 2 -> 16 sigmoid -> 1 regression net on [-3, 3]^2. On a grid it is a round bump centred at the
origin: 1.00 at (0, 0), 0.78 at radius 0.5, 0.60 at radius 0.71, 0.37 at radius 1, 0.12 at
radius 1.5, about 0 beyond radius 2. That fits `exp(-(x0*x0 + x1*x1))`.

That closed form does **not** pass. Its maximum error against the net is 0.027 and the snap
tolerance is 0.0205 (2 % of the output range). The error comes from ripples of about ±0.025 in the
flat tails, mostly near the edges and corners of the box, where the sum of 16 sigmoid ridges
does not cancel exactly. Things I tried that did not fix it:
- A minimax fit of a general Gaussian `A*exp(-(a x0² + b x1² + c x0 x1 + d x0 + e x1)) + k + p x0 + q x1`.
  The best fit is A≈1, a≈b≈1 and the rest ≈0, with a maximum error of 0.025, so it does not pass.
  The ripples are local, and no smooth global term removes them.
- Using the Gaussian as an extra basis function next to a subset of the sigmoids, with the output
  layer refitted by an LP (minimax). It adds nothing. 14 ridges plus the Gaussian are no better
  than 14 ridges alone, and with 13 or fewer ridges the error is above the tolerance whether the
  Gaussian is there or not.

## What I changed (agent.py, PASS snap, max error 0.0168)

- **Pruned 2 of the 16 units** (old h1_4, output weight 0.137, and h1_13, weight 0.053). Both are
  shallow ridges that the remaining units can absorb. After a minimax refit of the output layer
  the error is 0.0013. No other unit can be removed: with 13 units the error is at least 0.055 after a refit.
  Dropping a unit without a refit always gives an error about equal to that unit's output weight.
- **Snapped the ridge slopes and offsets to multiples of 0.1**, then refitted the output weights
  and bias by minimax LP on a 121x121 grid.
- **Snapped the output weights to 0.01 one at a time.** Each step fixes the coefficient that is
  closest to the 0.01 grid and re-solves the LP for the others. Rounding them all at once after
  the LP gave 0.026, which fails. Rounding them one at a time gave 0.0168.
- Renamed the hidden units `r0..r13` (ridges). Comments state the formula they approximate.

## Heuristics

- First tabulate the net on a coarse grid and compare it with familiar shapes (Gaussian bump,
  |x|, products, min/max). Check the ratios at radius 0.5, 0.71 and 1 against exp(-r²):
  0.78, 0.61, 0.37.
- Before snapping anything, compare the maximum error of the clean formula with the snap tolerance.
  For a smooth target built from many sigmoids, the net's own ripples in the flat regions can be
  larger than 2 % of the range. The formula is then the right reading of the net, but no
  program can reach the net with it. Report the formula and keep a snapped net.
- If the residual against the formula looks like noise with no structure, global corrections
  (refit width, tilt, offset) will not help. Test this quickly with a minimax fit before spending
  time on it.
- Pruning: remove one unit at a time and refit the output layer by **minimax LP**, not least
  squares, because the checker measures the maximum error. Units whose removal costs almost nothing
  after a refit are redundant even if their output weight is not tiny.
- Snap in stages. Snap the hidden layer coarsely (0.1), refit the output layer, then snap the
  output weights **one at a time with re-solves**. Rounding everything at once can turn a
  passing fit into a failing one.
- Keep a safety margin. Check the maximum error on a grid finer than the fitting grid. The
  checker adds random and boundary points, so aim at about 80 % of the tolerance or less.
