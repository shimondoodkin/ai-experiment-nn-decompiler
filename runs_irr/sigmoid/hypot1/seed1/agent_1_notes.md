# Notes

## What the network computes

A 2 -> 16 (sigmoid) -> 1 regression net on the square [-2, 2]^2. Sampling it on a grid shows a
symmetric bowl: about 1.03 at the origin, about 2.23-2.29 at the edge midpoints, 2.9-2.97 at the
corners. That is `sqrt(1 + x0^2 + x1^2)` (1, sqrt 5 = 2.236, 3).

The clean formula does **not** pass the check. Its max error against the net is 0.093, and the snap
tolerance is 0.0388 (2% of the output range 1.03..2.97). The net's own error is wavy:
- -0.09 at the (-2, -2) corner (the net is too low there);
- about -0.04 along the x0 = +2 edge;
- about +0.04 near (-0.15, 0.35) and at (-2, 1.5).

I tried the formula plus small corrections, fitting the extra coefficients by an exact minimax LP
on a grid:
- `a*sqrt(1+r^2) + b`: 0.064;
- `+ corner relu(-x0-x1-3.5)`: 0.045;
- `+ relu(-x0-1.5)`: 0.0399;
- `+ x0*x1`: 0.0373.

The last one passes on paper, but with no margin and five fitted coefficients. A fitted
`a*sqrt(b + (x0-p)^2 + (x1-q)^2) + c` with a corner term got 0.042. None of these was a clean
program with margin.

## What agent.py is

A pruned and snapped version of the network itself (PASS snap, max error 0.024):
1. **Split the units.** Eight of the 16 units have tiny input weights (|w| <= 0.35, so z moves by
   at most 1.4 over the box). Together they are almost linear: `0.307 - 0.052*x1`, max deviation
   0.002. I replaced them with a linear term.
2. **Prune the steep units.** I dropped one of the 8 steep units, then a second. After each drop
   I refit the output layer plus `const + x0 + x1` by minimax LP. 7 units gave 0.019 and 6 gave
   0.032.
3. **Refit.** I refit all parameters of the 6-unit model (hidden weights too) to the net by least
   squares, then L4 and L8 losses to push down the max error: 0.0076. For 5 units the same got
   0.030, which leaves too little margin for snapping.
4. **Snap greedily.** One parameter at a time, I tried grids 1, 0.5, 0.25, 0.1, 0.05, 0.01, kept
   the coarsest step whose refit (of the not yet snapped parameters) stayed under 0.02 max error on
   a 401x401 grid, and froze it. Hidden weights ended on 0.05-0.25 steps. Output weights and the
   linear tail have 2 decimals; the tail is sensitive, since rounding `-0.72` to `-0.7` doubles the
   error.

Result: 6 soft steps, each bending the bowl up towards one side or corner of the square, plus a
linear tail. The comments in `agent.py` name each unit by the side it lifts. The intended
formula, `sqrt(1 + x0^2 + x1^2)`, is documented in the program header.

## Heuristics

- **Look at a coarse grid first.** For 1-2 input regression nets, print a 9x9 table of outputs.
  Symmetric bowls and values like 1, sqrt 5, 3 identify `sqrt(1 + r^2)`-type formulas at once.
- **Check the formula's max error before polishing.** If the net's own fit error is bigger than
  the snap tolerance (here 0.09 vs 0.039), no clean formula will pass. Look at the residual map:
  wavy residuals of about the tolerance size mean the net's fit noise must be kept.
- **Use a minimax LP, not least squares, for correction terms.** The check is a max-error check;
  `linprog` over [features, error] gives the best achievable max error for a feature set exactly.
- **Collapse low-slope sigmoid units to a linear term.** With |w|*range <~ 1.5 they are almost
  linear over the box; their sum is fit by `a + b*x` to about 1e-3.
- **Prune greedily with an output refit, then refit everything.** Refitting the hidden weights
  after pruning recovers much more than refitting the output layer alone (6 units: 0.032 -> 0.008).
- **Fit max error with escalating Lp losses.** `least_squares` on `|r|^(p/2)*sign(r)` with p = 2, 4,
  8 is a cheap route to a near-minimax fit of a nonlinear model.
- **Snap greedily with refit.** Snap one parameter to the coarsest grid that keeps error under
  about half the tolerance, refit the rest, freeze it, move on. Leave the linear tail and output
  biases for last; they are the most sensitive.
- **Keep about 40% margin under the tolerance.** The checker probes about 273k points, including
  corners and edges, where nets are worst; a 401x401 grid max was a reliable predictor here
  (0.0239 on the grid, 0.0239 in the checker).
