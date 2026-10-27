# Notes

## What I changed and why

- The dump is a single ReLU neuron: `relu(x2 - x5 + 0.0005*(x0+x1+x3+x4+x6+x7))`, output scaled by 1.
- The six 0.0005 weights contribute at most 6 * 0.0005 = 0.003 over the [0,1] input box, well under the
  snap tolerance (0.02), so they were treated as training noise and snapped to 0.
- Weights +1 / -1 on x2 / x5 kept as is; the output scale of 1 was folded away.
- Result: `return relu(x2 - x5)`, i.e. the positive part of the difference x2 - x5 (a "how much does x2 exceed x5" feature).
  `check.py` gives `PASS (snap)`, max deviation 0.003, all within the snap/unsure zone.
- Keeping the 0.0005 terms would reproduce the net exactly, but they carry no meaning, so I preferred the clean formula.

## Heuristics

- Bound each small weight's total contribution by |w| * (input range) summed over inputs; if that sum stays below
  the snap tolerance, drop those terms entirely.
- Weights within a few percent of 0, ±1 or small integers are almost always the intended values; snap them.
- Fold identity output layers (weight 1, bias 0) and single-neuron hidden layers straight into the return expression.
- A single ReLU of a signed difference `relu(a - b)` means "excess of a over b"; recognize common motifs
  (relu(a-b) + b = max(a, b), relu(a-b) - relu(b-a) = a - b, etc.) when there are more neurons.
- Inputs whose weights are all snapped to zero are unused arguments; keep them in the signature only.
- Verify with the checker after each snap; if failures appear, look at the listed inputs to see which snapped term mattered.
