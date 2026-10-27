# Notes

## What changed and why
- The dump has two hidden ReLUs, `relu(x0 - x1)` and `relu(x1 - x0)`, both added to the output with weight -1.
  Since relu(d) + relu(-d) = |d|, the hidden layer collapses to `abs(x0 - x1)`.
- The output is `0.2 - |x0 - x1|`, so class 1 means "x0 and x1 are within 0.2 of each other"
  (an approximate-equality test). The bias 0.2 is the named constant `tolerance`.
- Weights were already ±1. No snapping was needed. `check.py` reports PASS (exact). The only
  disagreements are float rounding at the exact boundary |x0 - x1| = 0.2, which the checker
  puts in the unsure zone.

## Heuristics
- A pair of ReLUs with opposite weight vectors (`relu(w.x + b)` and `relu(-w.x - b)`) and equal
  output weights is `abs(w.x + b)`. Unequal output weights give a piecewise-linear "leaky" form.
- A bias minus an absolute difference, compared to 0, is a band test: |a - b| < c. If the sign
  is reversed, it is an "inputs differ by more than c" test.
- Keep the real-valued score, not just a boolean, so the check can pass exactly rather than
  only by snapping.
- Before snapping anything, look for algebraic identities (abs, min, max) among the hidden units.
  They often remove the whole hidden layer.
- Boundary disagreements of about 1e-16 are float rounding, not logic errors.
