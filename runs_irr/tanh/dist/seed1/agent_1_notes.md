# Notes

Net: 2 float inputs in [-3, 3], 16 tanh units, regression output (range about 0.65 to 4.33).

## What I changed

1. **Found the units that matter.** Four units have large input weights:
   - h1_0 and h1_11 are a mirror pair: `tanh(0.26*x0 -/+ 0.445*x1 - 0.965)`, output weights 2.64 and 2.77.
     Together they make the output symmetric in x1 and growing with |x1|, and growing with x0.
   - h1_14, `tanh(-0.42*x0 - 0.70)`, weight 2.88: grows as x0 goes negative (left wall).
   - h1_7, `tanh(-0.70*x0 + 0.84)`, weight -0.50: a soft step in x0.
2. **Folded the 12 small units into a line.** The other 12 units have input weights below 0.04, so
   their arguments move by less than 0.25 across the box and tanh is close to linear there. Their
   weighted sum fits `6.485 - 0.105*x0 - 0.030*x1` with max error 0.028; the constant and x0 slope
   were merged into the output bias and slope.
3. **Tied the mirror pair.** Gave h1_0 and h1_11 one shared weight, slope and bias (only the x1 sign
   differs). Refit of the 4-unit model to the net: max error 0.0095. The x1 slope of the linear part
   went to ~0 once the pair was symmetric, so it was dropped.
4. **Snapped greedily, coarse to fine.** One parameter at a time: try the floor/ceiling on grids
   0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, keep the coarsest grid whose least-squares refit of
   the remaining free parameters stays under 0.03 max error (less than half the 0.074 snap
   tolerance), then fix it. The snap order mattered. Snapping the output-side parameters first
   left the mirror pair needing 3 decimals (0.259, 0.958; max error 0.0285). Snapping the
   first-layer weights first gave round values everywhere (0.26, 0.45, 0.98, 0.45, 0.75;
   soft step `-0.5*tanh(1 - 0.75*x0)`) at a slightly larger max error of 0.0324. I kept that one.
5. **Tried and rejected:**
   - dropping the soft-step unit h1_7: refit max error 0.062, too close to the 0.074 tolerance to
     survive snapping;
   - a closed form like `sqrt(x0^2 + x1^2 + c)`: the net is 0.65 at the origin and much lower than
     the radius at the corners (3.84 at (3,3)), so it misses by about 0.45.

Result: `PASS (snap)`, max |program - net| = 0.0324 (tolerance 0.0736), 4 tanh terms instead of 16:
`2.66*(tanh(0.26*x0 - 0.45*x1 - 0.98) + tanh(0.26*x0 + 0.45*x1 - 0.98)) + 2.5*tanh(-0.45*x0 - 0.75)
- 0.5*tanh(1 - 0.75*x0) - 0.17*x0 + 6.625`.
The shape is a bowl: lowest near (0, 0), rising with |x1| (symmetric) and with |x0| (asymmetric:
steeper for negative x0).

## Heuristics

- In tanh/sigmoid nets, units with tiny input weights (argument range across the input box well
  under 1) are close to linear; replace their weighted sum with one least-squares affine (or
  quadratic) fit and fold it into the output bias and a linear term.
- Look for mirror pairs: units whose input weights are equal up to the sign of one input, with
  similar biases and output weights. Tie them (shared parameters) and the output becomes exactly
  symmetric in that input; any leftover linear term in that input usually drops to ~0.
- After pruning, refit the reduced model to the net (not to a guessed target) before snapping.
  The refit absorbs what the removed units did.
- Snap one parameter at a time, coarse grid first, refitting the free ones after each fix. Snap
  Keep a safety margin (here about half the tolerance) so later snaps still fit.
- Snap order changes the result. Run more than one order: first-layer weights first, and output
  weights or linear terms first. Keep the run whose values are roundest overall. Here, snapping
  the first-layer weights first let them round to 2 decimals. The output weights absorbed the
  difference in the refit.
- Before trying to drop a unit, check the refit error against the tolerance: a unit whose removal
  uses most of the tolerance budget leaves no room for snapping and should stay.
- Check a candidate closed form at a few points (origin, axis ends, corners) before fitting it; a
  smooth net trained on a radial-looking target can be far from the exact formula at the corners.
