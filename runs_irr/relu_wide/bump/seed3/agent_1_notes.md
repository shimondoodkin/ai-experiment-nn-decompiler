# Notes

## What the network computes

Sampling the net on a grid shows a single radial bump centred at the origin: about 1.04 at (0, 0),
0.78 at r = 0.5, 0.39 at r = 1, 0.13 at r = 1.5, then a flat floor of about 0.01 to 0.04. This matches
`exp(-(x0^2 + x1^2))` (exp(-0.25) = 0.78, exp(-1) = 0.37, exp(-2.25) = 0.11) plus a small floor.

Looking at the hidden units as ridge hinges `relu(n.x - d)` (unit direction n, offset d) shows how
the net builds the bump. The live units fall into three rings by offset, with directions spread around
the circle:
- d close to -1.2 (active everywhere except far out in the -n direction): positive weight;
- d close to 0 (hinges through the origin): negative weight, about -0.1;
- d close to +1.2: positive weight.

For one direction, `relu(t + c) - 2 relu(t) + relu(t - c) = relu(c - |t|)` is a tent. A sum of tents
over many directions approximates a radial bump, so the net is a faceted, piecewise-linear Gaussian.

## Why the clean formula alone does not pass

The snap tolerance is 2% of the output range, which is 0.0211. The net's facets deviate from any smooth
surface by up to about 0.03:
- `exp(-r^2) + c` with the best c: max error 0.036 (0.0325 with the scale and width fitted by minimax);
- an anisotropic Gaussian with a linear term and an r^2 floor: 0.031;
- smooth bases (polynomial background up to degree 6 plus several Gaussian widths, each times a
  polynomial): 18 features give 0.028, and 33 to 58 features are needed to get under 0.021. That is
  no simpler than the net;
- a regular tent-ridge sum `C + A * sum_k relu(c - |cos(th_k) x0 + sin(th_k) x1|)` with 6 to 16
  directions: 0.10 or more. One tent width cannot make the Gaussian profile.

Inside a single radial bin (r in 3.0 to 3.1, the corners) the net's values spread by 0.049, which is
more than 2 x 0.0211. So no purely radial function can pass. Part of the net's own faceting has to stay.

## What agent.py is

`agent.py` = constant + a * exp(-(x0^2 + x1^2)) + 33 of the net's own hinge units (out of 64), with the
output weights refitted.
- 7 units are dead on [-3, 3]^2, because their pre-activation is negative at every corner (units 14, 23,
  40, 43, 53, 58, 63). Dropping them alone gives PASS (exact) with 57 units
  (`scratch/agent_exact.py`).
- Unit selection: reweighted-L1 linear programs that minimise the sum of |output weight| subject to
  |program - net| <= 0.017 on a sample. Exchange iterations add the worst points of a 150k-point set.
  The base columns are the constant and exp(-r^2), and they cost nothing in the objective. This kept
  33 units. Without the Gaussian column, the same procedure needs 43 units. So the Gaussian term
  replaces about 10 hinges, and most of the remaining hinges only reproduce faceting noise.
- Final weights: first-layer weights and biases are rounded to 2 decimals. The output weights are then
  refitted by a Chebyshev (minimax) LP. They are rounded to 4 decimals, because rounding them to 3
  moved the max error from 0.018 to 0.0200, too close to the tolerance.
- The hinges are grouped and named by ring (`outer_*`, `mid_*`, `inner_*`).

Result: `PASS (snap)`, max |program - net| = 0.0180 against a tolerance of 0.0211. The checker
with seeds 1, 2 and 3 also passes (0.0177 to 0.0182), so the fit is not tuned to one point set.

The Gaussian scale in the program is 0.73, not 1, and the constant is -0.042, because the hinges carry
part of the bump. Read without the corrections, the function is about `exp(-(x0^2 + x1^2)) + 0.02`.

## Heuristics

- Sample the net on a coarse grid first and compare it with standard shapes (exp(-r^2), 1/(1+r^2), cones).
  A peak of about 1 at the origin with exp(-0.25) at r = 0.5 identifies a unit Gaussian quickly.
- Before snapping, compute the tolerance (2% of the output range) and the net's own deviation from the
  best smooth formula. If the net's facet noise is larger than the tolerance, no clean formula will pass.
  Then the useful target is formula + a small set of the net's own units, not a formula alone.
- Within-bin spread check: bin the outputs by the candidate invariant (here r). If the spread inside a bin
  is more than twice the tolerance, no function of that invariant alone can pass.
- Write ReLU units as ridge hinges (direction angle, signed offset, weight times norm). Sorting by
  offset shows the construction: rings of hinges at offsets -c, 0 and +c with signs +, -, + are
  tents `relu(c - |n.x|)`, and many directions of them make a radial bump.
- Dead-unit test for a box domain: the max of `w.x + b` over the box is `sum(|w_i| * half-width) + w.center
  + b`. If it is negative, drop the unit. This removal is exact.
- For pruning with a max-error criterion, use a linear program, not least squares: a minimax fit for the
  weights, and L1-weighted sparsity (reweighted L1) for choosing units. Put the smooth formula in as a
  free (zero-cost) column so that it absorbs what it can. Use exchange iterations: refit with the worst
  points of a large sample added. Greedy forward selection by least squares did badly here (24 units still
  at 0.072 without the formula).
- One backward-elimination step with a full LP refit per candidate is too slow at 60 units. Reweighted
  L1 gets a similar count in about 12 LPs.
- Round first-layer weights (2 decimals) before the final output-weight refit, not after. Then round the
  output weights only as far as the error budget allows, and re-check after rounding.
- Validate with the checker under several seeds before stopping, so that the result is not tuned to one
  point set.
