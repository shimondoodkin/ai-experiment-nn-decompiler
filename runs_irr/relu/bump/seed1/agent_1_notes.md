# Notes

## What the network computes

A single bump centred on the origin. The value is about 1.05 at (0, 0), falls roughly linearly to
about 0 at radius 1.4-1.6 in every direction, and is nearly flat further out (|y| < 0.1, with
polygonal ripples). Radial samples match exp(-(x0^2 + x1^2)) well near the centre (0.37 at r = 1,
0.13 at (1, 1)), so the net is probably a piecewise-linear fit of a Gaussian bump. But that formula
alone is 0.115 from the net (snap tolerance 0.023), and a cone relu(1 - r/1.6) is 0.45 off, so no
closed formula passes: the net's own facets have to be kept.

## What I changed

- Unit 7 (`0.0019 x0 - 0.0102 x1 - 0.382`) is never active on the box: removed.
- Unit 4 (`-0.136 x0 + 0.195 x1 + 0.969`) is active on the whole box except a sliver at the corner
  (3, -3): replaced by its linear part, folded into a `+ a*x0 + b*x1` term and the constant.
- 14 ReLU units remain. Each first-layer coefficient was snapped, one at a time, to the coarsest
  grid (1, 0.5, 0.25, 0.1, 0.05, else 2 decimals) for which a minimax (LP) refit of the output
  layer kept the max error under a budget. Seven units then pass through the origin (bias 0 or
  about 0.1).
- The output weights were then snapped one at a time (largest first) with the LP re-fitting the
  rest; four values stay at 3 decimals (0.447, 0.437, 0.048, -0.072) because rounding them to
  2 decimals pushed the corner (3, -3) to 0.028 error.
- Grouped the units into "down" (kinks through the origin, negative weight: the slopes of the
  bump) and "foot" (kinks about 1-1.3 from the origin, positive weight: they cancel the slopes so
  the outside is flat). Result: `foot - down + 0.048 x0 - 0.072 x1 - 1.1`.
- Result: PASS (snap), max |program - net| = 0.0193 against a tolerance of 0.0231.

Tried and rejected: dropping any further unit (best: 0.0295 even after an LP refit); merging the two
near-parallel origin units (0.5, -0.5) and (0.4, -0.6) into one (0.0189 before snapping the output
layer, too close to the tolerance).

## Heuristics

- Sample the net on a coarse grid and along rays from the centre first; a symmetric bump with a
  peak and decaying sides suggests exp(-r^2) or a cone, and checking the error of those formulas
  quickly tells whether a closed formula can pass at all.
- Remove units whose pre-activation is negative on the whole input box (check the box corners:
  |w0|*3 + |w1|*3 < -b means always off).
- Replace units that are positive on (almost) the whole box by their linear part; this adds a
  linear term `a*x0 + b*x1` and changes the constant.
- Snap one coefficient at a time and refit the output layer by minimax LP (min max |error|), not
  least squares: the checker's criterion is the max error, and LS fits leave much larger max
  errors. Try coarse grids first (1, 0.5, 0.25) and fall back to finer ones.
- Keep a safety margin (about 70-80% of the tolerance) on the fitting set, since the checker
  also samples points on the kinks and corners; corners of the box are where rounding errors in
  linear terms grow largest.
- Classify units of a 2-D bump by the distance of their kink line from the centre: lines through
  the centre with negative weight build the slopes, lines at the bump's radius with positive weight
  flatten the outside.
- On a busy machine, LP-based searches (seconds each) are much cheaper than retraining with torch;
  retraining with a high-p norm from a good start can also make the max error worse.
