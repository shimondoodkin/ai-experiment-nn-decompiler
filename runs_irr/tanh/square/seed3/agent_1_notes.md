# Notes

## What I did

- The dump is a 1-input, 8-unit tanh layer with a linear readout, x0 in [-2, 2].
- Sampled the net on a grid (scratch/look.py). The output is symmetric around 0, about 0 at
  x0 = 0, about 1 at x0 = ±1 (0.981 / 1.009) and about 4 at x0 = ±2 (3.941 / 3.968).
  That is the shape of x0².
- Replaced the whole net with `return x0 * x0`. Check: PASS (snap), max |program - net| = 0.059
  against a snap tolerance of 0.079 (2% of the output range). The largest gaps are at the domain
  edges (x0 = -2: net 3.941 vs 4), where the net under-shoots the parabola.
- An exact pass would need practically all 8 tanh units (atol 1e-5), so the dump itself is the
  exact version; nothing in between is more readable. Some units look redundant (units 1 and 3 have
  nearly the same weights and opposite-sign partners 4, 5, 7 with slopes ~0.13), which is the
  typical pattern of a tanh net building a smooth even function out of nearly linear pieces.

## Heuristics

- For a 1-input regression net, sample the output at integer and half-integer inputs first.
  Values close to 0, 1, 4 at 0, ±1, ±2 (or other recognisable sequences) point to a simple
  formula; try that formula directly with the checker before trying to simplify units.
- Check symmetry: f(-x) ≈ f(x) suggests an even function (x², |x|, cosh-like); f(-x) ≈ -f(x) an odd one.
- Tanh units with small slopes (|w| ≲ 0.15 over a domain of width 4) are almost linear in the
  domain; pairs with near-equal weights and opposite-sign readout cancel or merge into a line.
  Their sum is mostly a linear/quadratic correction, not a feature.
- The worst error of a snapped formula is usually at the domain edges, where the net extrapolates
  poorly; compare the margin there against the snap tolerance before accepting.
- When the snap-tolerance margin is comfortable (here 0.059 vs 0.079), prefer the clean formula
  over a refit with fitted constants.
