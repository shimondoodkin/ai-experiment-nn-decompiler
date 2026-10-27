# Notes

## What the network computes
Two bool inputs, binary output (class 1 if value > 0). Evaluating all 4 inputs:

| x0 | x1 | active hidden units | output |
|----|----|--------------------|--------|
| 0 | 0 | only h1_0 (0.270) | +5e-8 -> 1 |
| 0 | 1 | only h1_0 (0.286) | +1.3e-7 -> 1 |
| 1 | 0 | only h1_0 (0.161) | -4.3e-7 -> 0 |
| 1 | 1 | only h1_0 (0.177) | -3.6e-7 -> 0 |

- h1_1, h1_2, h1_3 have pre-activations that are negative on every corner of {0,1}^2
  (e.g. h1_1 max = 0.194 - 0.626 < 0), so they are dead and removed.
- The only live unit h1_0 is always positive, so the network is affine:
  out = 4.5e-6 * (0.270 - 0.108 x0 + 0.016 x1) - 1.16e-6. Its sign flips on x0 only
  (the x1 term is too small to cross zero), i.e. the decision is `not x0`.
- The whole output scale is ~1e-7, below the checker's atol, so the net is technically "unsure"
  everywhere; still, the sign pattern is clean and consistent, so I kept the net's classes.

## Final program
`return 1 - 2 * x0`  -> +1 for x0=0, -1 for x0=1 (symmetric margin around the 0 threshold).
check.py: PASS (exact), 0 disagreements (a constant would also "pass" via ties, but disagrees on 2 inputs).

## Heuristics
- For small discrete (bool) input spaces, enumerate every input and read the truth table directly;
  that beats snapping weights one by one.
- Check each ReLU's max pre-activation over the input box (sum of positive weights * hi + bias);
  if it is <= 0 the unit is dead and can be dropped. If its min is >= 0 the ReLU is the identity.
- When all live units are always-on, the net collapses to one affine function; then only the sign
  pattern (for classification) matters, and tiny coefficients that never change a sign can be dropped.
- For binary outputs, rescale the final score to small integers (e.g. +-1) with a margin around the
  threshold rather than reproducing tiny raw magnitudes.
- Beware of a checker that treats near-zero scores as ties: a trivial program can "pass"; confirm
  zero disagreements, not just the PASS label.
