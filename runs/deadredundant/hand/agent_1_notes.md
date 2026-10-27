# Notes

Original: 2 -> 4 -> 1 ReLU network.

Changes:
- `h1_0 = relu(x0 + x1)`: inputs are in [0, 1], so the sum is never negative and the relu is a no-op. It becomes `x0 + x1`.
- `h1_1 = relu(-x0 - x1 - 1)`: the argument is at most -1 over the input box, so the neuron is always 0. It is dead, and its weight of 5 does not matter. Removed.
- `h1_2` and `h1_3` are the same neuron, `relu(x0 - x1)`, each with output weight 0.5. Merged into one `relu(x0 - x1)` with weight 1.
- Result: `x0 + x1 + relu(x0 - x1)`. Using `a + relu(b - a) = max(a, b)` with a = x1, b = x0, this is `x0 + max(x0, x1)`.

check.py: PASS (exact), max error 1.8e-7.

## Heuristics
- Use the input bounds to evaluate each pre-activation's range. If it is always <= 0, the neuron is dead, so delete it no matter how large its output weight is. If it is always >= 0, the relu is the identity, so drop the relu.
- Look for duplicate neurons (identical weights and bias). Merge them by adding their output weights. Two halves that sum to 1 are a common sign of this.
- Rewrite `a + relu(b - a)` as `max(a, b)`, and `a - relu(a - b)` as `min(a, b)`. Pairs of relus often reduce to min, max or abs: `relu(t) + relu(-t) = abs(t)`.
- Check the simplified closed form exactly with the checker, not just on a few samples.
