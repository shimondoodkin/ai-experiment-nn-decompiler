# Notes

## What the network computes

A 2 -> 16 tanh -> 1 regression net on [-3, 3]^2. A grid of outputs at integer points showed
1.0 at the origin, 0.37 at (±1, 0) and (0, ±1), 0.135 at (±1, ±1), and about 0 elsewhere. That is
the Gaussian bump `exp(-(x0^2 + x1^2))` (e^-1 = 0.368, e^-2 = 0.135).

## Why the plain formula is not enough

`return exp(-(x0*x0 + x1*x1))` FAILS: max error 0.035, tolerance 0.0205, about 1.2% of the domain
outside tolerance. The net's deviation from the Gaussian is noise-like training ripple of up to
±0.035. Most of it is on the borders |x| ~ 3, with a few patches inside (around r ~ 2). Things that
did not close the gap:
- A tuned Gaussian (amplitude, separate widths, center shift, offset), minimax fit: 0.032.
- exp(-r^2) plus a sparse subset of the net's own units (greedy forward selection, minimax LP):
  15 of the 16 units were needed to get under tolerance. The ripple is spread over all units.
- Rounding the net's weights without refitting: 0.01 steps gave 0.0236 (FAIL). Hidden weights
  are very sensitive.
- Fixing the bump coefficient at 1 (so the tanh part is only the ripple): 0.0220 (FAIL).

## What agent.py is

`0.7904 * exp(-(x0^2 + x1^2)) + ripple`, where `ripple` is the net's 16 tanh units:
- hidden weights and biases snapped to multiples of 0.1 (unit 5's x1 weight snaps to 0);
- the 17 output coefficients plus the bump coefficient refit by a minimax linear program on a
  grid, dense border lines and random points against the net, then rounded to 1e-4.

Check result: PASS (snap), max |program - net| = 0.0187 (tolerance 0.0205).

Dropping one unit (unit 4) still passed at 0.0193, and dropping two (units 4 and 14) at 0.0199.
I kept all 16 for margin, since the collector re-checks with more counterexamples. With hidden
weights snapped to 0.2 it fails (0.0208).

Honest summary: the function is exp(-(x0^2 + x1^2)). An exact-enough program has to copy the
net's training ripple, which needs about all 16 units, so the readable formula alone cannot pass
a 2% check.

## Heuristics

- Evaluate the net on an integer grid first and compare it with simple closed forms. Values like
  1, 0.368, 0.135 point straight to exp(-r^2).
- Before snapping, check the *formula's* error against the tolerance and map where it fails. If
  the failures are scattered ripple, mostly at the domain edges, the net is "formula + training
  noise", and no small symbolic tweak (amplitude, width, offset) will fix it.
- Keep the formula as a basis column and refit the net's hidden units around it. Refitting the
  output layer by minimax LP after snapping the hidden weights recovers most of the rounding
  error. Hidden weights rounded to 0.1 work with a refit, but fail badly without one.
- Let the formula's coefficient float in the refit. Forcing it to 1 left the tanh units unable to
  absorb the whole residual here.
- Use minimax (LP), not least squares, when the acceptance test is a max-error bound. Include
  dense points on the domain border: tanh nets often deviate most at the box edges.
- Greedy forward and backward unit selection with LP refits quickly shows whether the residual is
  low-rank (a few units) or spread over the whole layer.
- HiGHS dual simplex can fail on large LPs. Fall back to `highs-ipm`.
- Leave margin under the tolerance (about 10%). The re-check uses more counterexamples than a
  single local run.
