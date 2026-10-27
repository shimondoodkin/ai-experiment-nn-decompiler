# Notes

## What the network does
Class 1 iff `|x1 - x0| <= 0.2` ("x0 and x1 are approximately equal").

## What I changed and why
- Every hidden neuron except h1_2 has weights of the form `a*(x1 - x0) + b` (coefficients on x0 and x1
  are equal and opposite), so the only real feature is `diff = x1 - x0`.
- h1_2 (`-0.136*x0 + 0.051*x1 - 0.61`) is negative everywhere on [-2,2]^2 -> always 0 -> removed.
- h1_0, h1_3, h1_4 are three copies of the same unit `~3.5*relu(diff - 0.07)` with large negative output
  weights (~-9.5 each): a steep penalty for diff > 0. h1_5 `~5.2*relu(-diff - 0.05)` with weight -10.8
  is the mirror penalty for diff < 0. h1_1, h1_6, h1_7 are broad positive "bias" units that are active
  over the whole region near the diagonal and contribute ~+11.5 there.
- Result: a ridge along x0 == x1. Scanning the decision boundary along lines of constant x0+x1 gives
  diff in about [-0.2028 - 0.0006*s, 0.2030 - 0.0009*s] -- a symmetric band of half-width ~0.2 with a
  negligible tilt. Snapped to the single shared threshold 0.2.
- Checked: `PASS (snap)`, 0 disagreements where the net is confident; all residual disagreements are in
  the thin sliver 0.1997 < |diff| < 0.2063 (the net's boundary is slightly tilted/asymmetric).
  `PASS (exact)` would require reproducing that tilt, i.e. essentially the raw weights, so I kept the clean
  formula.

## Heuristics
- Look for neurons whose input weights are (nearly) equal and opposite: they depend only on a difference
  (or with equal signs, a sum) of inputs. Re-express the whole net in that derived variable.
- Bound each pre-activation over the input box (min/max via weight signs); a ReLU that is never positive
  is dead and can be dropped, one never negative is linear.
- Neurons with proportional weight vectors (same direction, similar bias/scale ratio) are duplicates;
  merge them by summing output weights times scale.
- Large negative output weights on relu(+d - c) and relu(-d - c) pairs, plus positive "bias" units,
  form a band/ridge detector: output > 0 iff |d| < threshold. Find the threshold numerically by scanning
  the decision boundary, then snap to a round number.
- For binary outputs, measure the boundary directly (bisection/scan along the derived variable for
  several values of the orthogonal variable) rather than reasoning about every unit's magnitude;
  if the boundary barely moves, it is a constant threshold.
- Prefer returning a comparison (1/0) for binary outputs; the score's magnitude doesn't matter.
