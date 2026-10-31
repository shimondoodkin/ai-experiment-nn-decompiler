# Notes

## What the network computes

A regression on x0, x1 in [-3, 3]. On a grid the output looks like a distance from the origin:
about 3 at (0, ±3) and (±3, 0), about 4.24 at the corners, and smallest (about 0.37) at the origin.
The best smooth formula I found is about `1.01 * sqrt(x0*x0 + x1*x1 + 0.12) - 0.06` (a softened
`sqrt(x0*x0 + x1*x1)`). Its largest error against the net is about 0.083, and plain
`sqrt(x0*x0 + x1*x1)` is 0.37 off at the origin. The snap tolerance is 0.078.

The residual (net minus formula) does not have a simple structure. It wiggles by about ±0.08 across
the whole square, with peaks at the origin, near the edge x0 = -3 and at the corners. Richer smooth
formulas (general quadratic under the sqrt, plus linear terms) only reached about 0.083. Adding the
net's own units one at a time to the formula did not help much either: with 8 units it was still 0.073.

## What I changed

So `agent.py` is a reduced net, not a closed formula:

- **Pruning:** I dropped 6 of 16 units (4, 6, 8, 10, 13, 15). These are the shallow units (small
  input weights, so near-linear over the box) and those with small output weights. One linear term
  `-0.56 - 0.19*x0 - 0.03*x1` replaces them. With a linear term, 10 units fit to 0.013. Without
  it, 10 units fit only to 0.039, and 9 units reach 0.11.
- **Snapping:** I rounded the hidden weights and biases to multiples of 0.1, which zeroes the tiny
  x1 weights of units 5, 9 and 14, so they depend on x0 only. Rounding to 0.25 gave 0.07, too close
  to the limit.
- **Refit:** I refit the output weights by minimax (an LP over about 7.7k points: random points,
  a grid and the origin), then rounded them to 0.01.
- **Result:** `PASS (snap)`, max |program - net| = 0.048 against a tolerance of 0.078.
- **Kept on purpose:** dropping the small `-0.03*x1` term fails (0.118), so it stays.

## Heuristics

- Print the net on a coarse grid first. Here the grid showed the radial shape right away
  (3 on the axes at radius 3, 4.24 at the corners).
- Compare the max error of the best smooth formula with the snap tolerance. If a minimax-fitted
  formula family levels off above the tolerance and the residual is unstructured wiggle, the net's
  own error makes a closed form impossible. Stop searching for formulas and reduce the net instead.
- For sigmoid nets, units with small input weights are near-linear over a bounded box. Replace them
  with one free linear term `c + a*x0 + b*x1` instead of pruning them outright. Here that term let
  10 units fit to 0.013, where 10 units alone fit only to 0.039.
- Backward elimination with a minimax LP refit of the output layer (`scipy.optimize.linprog`,
  minimize t with |F w - y| <= t) finds the smallest unit set that passes quickly. The tolerance is a
  max-error bound, so minimize the max error, not the least-squares error.
- Snap the hidden weights first, then refit the output weights on the snapped units, then round the
  output weights. Rounding the output weights to 0.01 cost about 0.016 of max error here; keep
  margin for that.
- Include grid, edge and corner points (and the origin) in the fit set. The checker tests corners
  and boundary points, where sigmoid nets often deviate most.
- A tiny-looking coefficient (here -0.03 on x1) can still matter after a refit. Test each removal
  with the checker rather than assuming.
