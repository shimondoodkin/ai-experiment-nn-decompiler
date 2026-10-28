# Notes

## What the network computes
Probing showed an odd function, y(-x) ~= -y(x), with y(1,2,3) ~= 2, y(3,-3,1) ~= 1 and y(t,t,0) ~= t.
That is a smoothed **median of three** (least squares: y ~= 0.995*median). It is a rough one, though:
it differs from the true median by up to 0.66 (for example y(t,2,2) runs from 1.63 to 2.18). So the
program has to copy the net's own smooth shape, not the clean median (max(min(a,b),min(max(a,b),c))
is off by 0.66, beyond the 0.115 tolerance).

## What I changed
1. **Layer 2 collapse.** Flip every layer-2 unit with a negative output weight
   (w*sig(z) = w - w*sig(-z)). After that, all 16 rows have the same sign pattern and similar sizes,
   so layer 2 is one unit copied 16 times. A single sigmoid `A*sigmoid(z)+B` with refitted z holds
   within 0.07 (just averaging the rows gives 0.43, so it needs the refit).
2. **Layer 1 pruning.** 11 of the 16 units have tiny weights (~0.11, 0.11, 0.07) and work in the
   near-linear part of the sigmoid. Together they become one linear term `0.11*(x0+x1) + 0.07*x2`.
   Five units carry the structure.
3. **Centering.** Writing each unit as `sigmoid(.) - 0.5` sends the layer-2 bias to about 0 and the
   output bias to 0, so `y = 14*(sigmoid(z) - 0.5)`, which is odd, as observed.
4. **Snapping the directions** using the pairwise basis (2xi - xj - xk, xj - xk):
   - (1.56,-0.56,-0.97) -> `1.6x0 - 0.6x1 - x2`, and (0.57,-1.59,1.0) is its mirror under x0<->x1
     (it enters with the opposite sign), so the two make one symmetric term `x0_high + x1_high`.
   - (0.64,0.64,-1.26) -> `0.64*(2x2 - x0 - x1)`.
   - (-2.05,1.86,0.2) and (-1.84,2.05,-0.2) -> `2(x1-x0) -/+ 0.1(x0+x1-2x2)`. Their weights are
     opposite (+-0.61), so they are a *difference* of sigmoids: a bump near x0 == x1 scaled by
     (x0+x1-2x2). It looks redundant (two near-copies that cancel), but removing it costs 0.61.
   - Layer-1 biases (all ~0) are dropped.
5. Refit the 6 scalars with directions fixed, then snap them: 0.56, 0.6, 1, 0.11, 0.07, 14.

Result: `PASS (snap)`, max |program - net| = 0.0725 (tolerance 0.1145), from 16+16 units down to
6 sigmoids. Exact is out of reach for a 2-layer sigmoid net without keeping all its weights.

## Heuristics
- **Probe first:** check the symmetry (odd? which input swaps leave it unchanged?), then a few integer
  points, then compare with candidate formulas (median, max, sum...). Measure the max error against
  the tolerance before trusting a "clean" formula. A trained net can be a rough approximation of
  the clean function, so the program must follow the net, not the idea.
- **Sigmoid sign-flip:** `w*sigmoid(z) = w - w*sigmoid(-z)`. Flip units with negative outgoing weight.
  If the rows then line up, the layer is one unit copied many times. Collapse it to one unit and
  refit its row by least squares; plain averaging is not enough.
- **Near-linear sigmoid units:** units with small weights and small biases (|arg| < ~1 over the
  domain) act linearly. Replace them as a group with one linear term in the inputs, then refit.
- **Centered sigmoids:** use `sigmoid(.) - 0.5` (= tanh(./2)/2). Biases then often vanish and the
  function's oddness shows.
- **Pairs of near-identical units with opposite weights** compute a derivative-like bump
  (sig(a+d) - sig(a-d) ~ 2d*sig'(a)). They are not redundant. Keep them as a difference and test
  before deleting.
- **Snap first-layer directions** in a basis of pairwise differences (2xi - xj - xk, xj - xk, plus the
  sum) for order-statistic tasks (min/max/median), and look for mirror pairs under input swaps to tie
  their coefficients.
- **Order of work:** fix the directions (snapped), refit only the few scalar coefficients by least squares, then
  snap those to 2 significant digits, checking the max error at every step.
