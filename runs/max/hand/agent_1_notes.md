# Notes

## What I changed
- The dump: `relu(x0 - x1) + relu(x1) - relu(-x1)`. All weights were already exact (+-1), no biases.
- `relu(x1) - relu(-x1)` is the identity `x1`, so the net is `relu(x0 - x1) + x1`.
- `relu(a - b) + b = max(a, b)`, so the network computes `max(x0, x1)`.
- Result: `return max(x0, x1)`; `check.py` reports PASS (exact).

## Heuristics
- A pair of neurons `relu(w.x) - relu(-w.x)` with matching output weights is a linear pass-through: replace it by `w.x`.
- `relu(a - b) + b` = `max(a, b)`; `b - relu(b - a)` = `min(a, b)`; `relu(a) + relu(-a)` = `abs(a)`.
- First collapse pass-through pairs into linear terms, then look for max/min/abs patterns in what remains.
- Check that the simplified algebra is valid over the whole input domain (these identities hold everywhere, so no domain-dependent pruning was needed here).
