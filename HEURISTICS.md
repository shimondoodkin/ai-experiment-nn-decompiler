# Heuristics learned by Claude Code

Appended by `nndc agent collect`, one section per refactoring attempt.

## absdiff/seed0 — agent_1 (m0969d1, 2026-10-27)

- Group hidden neurons by the direction of their input weights. Weights like (a, -a) mean the neuron only sees a difference x0 - x1, and (0, a) means it only sees one input.
- Two neurons on the same direction with opposite signs, relu(k*d) and relu(-k*d), make an abs-like V shape. Add up (output weight * input slope) on each side. If both sides come to about 1, the group is abs(d). If they differ, it is a*relu(d) + b*relu(-d).
- Several neurons can share one kink. Sum their effective slopes instead of snapping each neuron on its own.
- Neurons with small output weights, or with kinks at the edge of the input range, are often noise. Pairs with nearby kinks and opposite slopes largely cancel. Remove them and use the checker to confirm that the leftover error is within tolerance.
- Snap biases that are tiny compared with the weights (relative bias < 0.1) to 0, so the kinks sit exactly at 0.
- Try the simplest candidate formula early. The checker's report of the maximum deviation tells you whether the remaining terms matter.

## xor/seed0 — agent_1 (m212faf, 2026-10-27)

- With a small discrete input domain (here 2 bools = 4 points), enumerate every input and read
  off the truth table first; the decision rule is often a textbook boolean function (XOR, AND, majority).
- For binary/argmax outputs only the sign/ordering matters; the score magnitude can be dropped, so a
  comparison expression (returning 1/0) is a valid replacement for a whole weighted sum.
- Units whose pre-activation is negative on every reachable input are dead; delete them.
- Biases that equal minus one weight (e.g. 2.08*x0 - 2.08) put the ReLU kink exactly on an input
  corner; snap them to integer thresholds like relu(x0 + ... - 1).
- Recognize standard ReLU motifs: relu(1 - a - b) = NOR, relu(a + b - 1) = AND, relu(b - a) = b AND NOT a
  for booleans.
- Verify with the checker after each simplification; prefer the simplest formula that passes exact.

## neareq/seed0 — agent_1 (md5d6e0, 2026-10-27)

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

