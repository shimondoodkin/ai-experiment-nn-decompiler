# Notes

## What the network computes

A 2-input, 16-sigmoid regression net on [-3, 3]^2. Its output is a single bump centred at the
origin with height 1.0 and value 0.37 at radius 1, 0.61 at (0.5, 0.5): essentially
`exp(-(x0*x0 + x1*x1))`. The plain formula does **not** pass the checker: the net has a tail ripple
of up to ±0.05 near the box edges (for example (-0.4, -3) gives 0.049), while the snap tolerance is
0.0207. So the delivered program keeps the net's own sigmoid basis.

## What I changed

- Dropped unit 4 (|w| = 0.33, almost linear and almost constant on the box). A least-squares refit
  of the output layer without it stays at 0.011 max error. Every other single removal costs much more.
  Dropping unit 6 as well needs x0/x1 linear terms (0.0156 max error, passes too). Dropping a third
  unit fails (≥ 0.02).
- Rounded the first-layer weights and biases to the 0.1 grid, then greedily coarsened single values
  to half and quarter steps (3, 1.5, 2.75 ...). A change was kept when the minimax refit stayed below 0.012.
- Refit the output weights and constant by **minimax** (a linear program over ~12k points: random
  points plus a grid), not least squares. At the same grid, minimax gave 0.0088 where least squares gave
  0.016. Output weights are kept to 3 decimals. Rounding them to 2 decimals gave 0.026 and failed, because
  15 small rounding errors add up.
- Wrote each unit in "outward" form: a unit with its boundary on the far side of the origin is
  flipped using `sigmoid(z) = 1 - sigmoid(-z)`, and the 1s are folded into the constant. Every
  unit then reads "switches on past a line on the positive side". Units are named by the angle of
  their normal and grouped into
  - 10 steep **walls** (|w| ≈ 3, bias ≈ -1.5: a line at distance ≈ 0.5 from the origin), all with negative
    output weight. Together they make the polygonal bump: `1.5 - walls`.
  - 5 shallower **tail** units with positive weight that shape the falloff and reproduce the
    net's edge ripple.
- Result: `PASS (snap)`, max |program - net| = 0.0122 (tolerance 0.0207), 15 sigmoids instead of 16.

## Things that did not work

- `exp(-(x0²+x1²))`: max error 0.049. Polynomial corrections (Chebyshev, total degree up to 12): ≥ 0.022.
- A fresh fit of a Gaussian plus 4 or 8 sigmoids: 0.038 and 0.034. A fresh fit of 8 or 12 sigmoids: 0.11 and 0.042.
  The ripple is specific to this net's units, so the net's own units are the right basis.
- Forcing all walls to steepness exactly 3 and offset exactly 1.5 (only the direction kept): 0.026, fails.

## Heuristics

- First evaluate the net on a coarse grid and print it; a recognisable shape (here a radial bump
  with exp(-r²) values) gives the intended formula quickly. Then check the formula's max error
  against the tolerance before investing in it: a net's own approximation error (tail ripple)
  can be larger than the snap tolerance, and then the formula alone cannot pass.
- For sigmoid layers, describe each unit as (steepness |w|, normal angle, boundary distance -b/|w|,
  output weight). Units with equal steepness and distance at many angles are walls of a polygonal
  bump. Very shallow units (|w| · box radius ≲ 1) are near-linear and are the first candidates to drop
  or replace by linear terms.
- Normalise units with `sigmoid(z) = 1 - sigmoid(-z)` so that all boundaries face outward; the sign of
  the output weights then separates "walls" from "corrections", and the constants fold into one.
- After pruning or snapping hidden weights, refit only the output layer, and refit by **minimax (LP)**
  rather than least squares. The acceptance test is a max-error test, and minimax roughly halved the error here.
- Keep more digits on output weights than on hidden weights when there are many units: rounding
  errors in the output layer add up linearly across units.
- Snap hidden weights greedily one value at a time, to coarser grids (0.5, then 0.25), with an
  error budget below the tolerance (about 60% of it) to keep margin for the checker's denser and boundary samples.
- When the ripple has to be kept, a fresh small fit (fewer units, or a formula plus a few units)
  rarely beats pruning the existing units: the existing units already encode the ripple.
