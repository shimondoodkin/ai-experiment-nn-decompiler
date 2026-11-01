# Notes

## What the network computes

On a grid, the 2-input regression net looks like a radial bump: about 1.06 at the origin, about 0.36
at radius 1, and near 0 beyond radius 1.5. It is close to `exp(-(x0^2 + x1^2))`. That formula fails
the check, though (max error 0.098 against a tolerance of 0.023), because the net's tails ripple by up
to ±0.1 (for example -0.095 near (1.2, 2.1)). The program has to reproduce the net's own
piecewise-linear shape, not the smooth formula.

## Structure found

Each unit was normalized as relu(a·x + b) with its kink at distance -b/|a| from the origin. The units
fall into three groups:
- **Cone** (8 units): kinks within 0.1 of the origin, all with negative output weights. Together they
  give the polygonal tip, roughly -k·|x|.
- **Inner ring** (4 units): kinks at distance about 1.2, active on the origin side, positive weights.
- **Outer ring** (4 units): kinks at distance about 1.2, active outside, positive weights. The inner
  and outer rings both flatten the cone into the plateau near 0.

## What I changed

1. **Normalization.** Each unit is rescaled so that its largest input coefficient is ±1. ReLU is
   positively homogeneous, so the scale moves into the output weight exactly.
2. **Snapping.** Hidden coefficients and offsets were snapped to multiples of 0.05. The ring offsets
   came out at ±1.4 (or ±1.2 where the direction is axis-aligned). In Euclidean terms these are all
   the same radius, about 1.2. One offset (1.45) was snapped to the shared 1.4.
3. **Refit.** After snapping, the output weights and bias were refit by minimax (an LP on a 121×121
   grid), then rounded to 3 decimals.
4. **Grouping and naming.** The units are grouped into `cone`, `inner` and `outer` sums.

Result: `PASS (snap)`, max |program - net| = 0.0193 (tolerance 0.0232).

## What failed

- **Formula:** exp(-r²) misses the tail ripples (0.098).
- **Pruning:** removing any unit fails, even with a minimax refit. The best removal (a small cone
  unit) gives 0.0176 with the original weights, but it fails once the weights are snapped.
- **Exact cone apex:** setting every near-origin offset to exactly 0 raises the error to 0.026, so the
  ±0.05 offsets stay.
- **Coarse snaps:** snapping to 0.1 gives 0.036, and to 0.2 gives 0.065.
- **Idealized structure:** a cone through the origin plus a single shared-radius ring, re-optimized
  (Adam on an L16 loss), only reached 0.040. That fit did show the ring as roughly a hexagon (outward
  normals near 0°, 60°, 120°, 180°, 240° and 300°, weights about 0.42, R about 1.22). It is a nice
  description, but it is not within tolerance of this net.

## Heuristics

- First look at the output on a coarse grid and guess the target function (here a Gaussian bump).
  Then check the formula. If the net's tails ripple by more than the tolerance, the formula cannot
  pass, and the job becomes simplifying the net itself.
- Normalize each ReLU unit by its largest |coefficient| (or by its norm) and push the scale into the
  output weight. After that, offsets read as kink positions and units can be compared.
- Group units by kink distance from the origin (-b/|a|). For radial functions this gives a cone at
  the origin (negative weights) and rings at a fixed radius (positive weights) that cut the cone off.
- A unit active toward the origin, relu(n·x + R), equals n·x + R - relu(-n·x - R). Inner and outer
  ring units are the same kind of kink, apart from a linear term.
- After snapping hidden weights, always refit the output layer with a minimax LP, not least squares.
  The acceptance test is a max-error bound, and least squares can fail where minimax passes
  (least-squares pruning gave 0.03, minimax 0.018).
- Snap in steps (0.01, then 0.02, 0.05, 0.1) and measure each step. Small offsets near 0 often matter
  more than they look: zeroing them can cost more than coarse snapping of the directions.
- With a max-error tolerance about 2% of the range, a 16-unit irregular net often allows no pruning.
  Spend the effort on snapping and readable grouping instead.
- In scratch folders, do not name a script after a stdlib module (`struct.py` broke the numpy import).
