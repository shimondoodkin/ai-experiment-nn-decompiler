# Notes

The dump has two hidden ReLU neurons with mirrored weights: relu(x0 - x1) and relu(x1 - x0),
summed with weight 1 each. Since relu(a) + relu(-a) = |a|, the network computes the absolute
difference `abs(x0 - x1)`. All coefficients were already exactly ±1 with zero bias, so no snapping
was needed. check.py reports PASS (exact).

## Heuristics

- A pair of ReLU neurons with negated weight vectors and equal output weights c is c*|w.x + b|;
  with opposite-sign output weights (c, -c) it is the linear term c*(w.x + b).
- relu(a) - relu(a - k) is a clamp of a to [0, k]; relu chains of this form often encode min/max/clip.
- Look for mirrored or scaled-copy neurons first; they collapse to abs, linear or clamp expressions.
- Exact-unit weights and zero biases suggest the dump is already the clean formula; verify with the
  checker before adding complexity.
