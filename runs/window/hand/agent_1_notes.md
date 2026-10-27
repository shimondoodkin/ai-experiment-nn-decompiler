# Notes

## What changed and why
- The dump has two hidden ReLUs, `relu(x0)` and `relu(-x0)`, each with output weight -1, plus a bias of +0.5.
- `relu(x) + relu(-x) == abs(x)`, so the two neurons together compute `-abs(x0)`. With the bias added, the network is `0.5 - abs(x0)`.
- Meaning: class 1 when `|x0| < 0.5`, which is a band of width 1 centred on 0 (an "is x0 near zero" detector). The weights were already clean (±1, 0.5), so nothing needed snapping.
- `check.py` reports `PASS (exact)`. The only disagreements are at x0 = ±0.5 exactly, where the net's score is about 1.5e-8 (float32 noise on the boundary), inside the unsure zone.

## Heuristics
- A pair of ReLUs with opposite input weights (`relu(w*x+b)`, `relu(-w*x-b)`) and equal output weights is `abs(w*x+b)` times that output weight. Collapse it into one `abs`.
- A pair with opposite input weights and opposite output weights is just linear: `relu(z) - relu(-z) == z`.
- For a binary output, write the score in a form whose sign is easy to read, like `c - abs(x - m)`. It reads directly as "within c of m", and you can name the centre and the half-width.
- Before snapping, check whether the weights are already exact simple values. If they are, aim for `PASS (exact)` with an algebraically equivalent formula rather than a rewritten decision rule.
- Disagreements that sit exactly on the decision boundary with near-zero scores are float noise, not errors in the formula.
