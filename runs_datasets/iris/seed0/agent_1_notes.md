# Notes

## What I changed and why

1. **Dead neurons removed.** Layer-1 pre-activations are linear, so their range over the input box is
   attained at the 16 box corners. Neurons 2, 3, 4, 7 are negative on every corner (always 0) and were
   dropped. Neuron 1 is active on ~9% of the box but removing it changed the decision on 0.008% of
   random points (all in the unsure zone), so it was dropped too.
2. **Always-on ReLUs linearised.** Neurons 5 and 6 are active on ~97% of the box; replacing
   `relu(z)` with `z` changed no confident decision. Their contributions fold into the output biases
   and input weights, so each output becomes `linear(x) + c * relu(u(x))` with one remaining ReLU `u`
   (neuron 0).
3. **Argmax turned into a decision list.** Empirically: class 2 only occurs where `u = 0`, class 0
   only where `u > 0`. The rule "class 0 if o0 > o1, else class 1 if o1 >= o2, else class 2"
   reproduced the net. With `u = 0`, `o0 - o1` is negative on the whole box, so `o0 > o1` already
   implies `u > 0`, and on that side `relu(u) = u` — the class-0 test becomes a single linear
   inequality. Class 2 never occurs with `u > 0`, so `o1 >= o2` can use the `u = 0` linear form.
   Result: two linear half-space tests.
4. **Snapping.** Normalised each inequality by one coefficient and grid-searched nearby simple values
   (halves / quarters / 0.05 steps), keeping those with zero confident disagreements:
   - class 0: `x0 + 4.11 x1 - 5.10 x2 - 4.68 x3 - 0.33 > 0`  ->  `x0 + 4 x1 - 5 x2 - 4.5 x3 > 0`
   - class 1 vs 2: `x0 + 1.98 x1 - 2.75 x2 - 5.06 x3 + 10.03 >= 0`  ->  `4 x0 + 8 x1 - 11 x2 - 20 x3 + 40 >= 0`
     (i.e. `x0 + 2 x1 - 2.75 x2 - 5 x3 + 10`; the same snap was found under every normalisation).
   The second boundary (class 1 vs 2) was the sensitive one: plain half-steps failed, quarter-steps worked.
5. Verified: `PASS (snap)`, 0 confident disagreements; also 0 with checker seeds 1–3.
   Inputs look like Iris (sepal length/width, petal length/width): class 0 = small petals.

## Heuristics

- Compute each first-layer neuron's pre-activation range on the box corners (exact for linear
  pre-activations): always-negative neurons are dead, always-positive ones are linear.
- Also test "mostly on" neurons by linearising them and counting confident disagreements; ReLUs that
  only clip in a far corner usually don't matter.
- Ablate neurons one by one and measure decision changes; drop the ones that only affect the unsure zone.
- For argmax outputs only differences of outputs matter: subtract one output and think in terms of
  pairwise margins. Try to rewrite argmax as an ordered decision list (`class A if oA > oB else ...`)
  and check it empirically.
- When a single ReLU remains, check which side of it each class lives on; often the class boundary
  only exists on one side, which turns `relu` into either `0` or the identity there and makes the
  boundary a plain linear inequality.
- Snap whole inequalities (not individual weights): normalise by one coefficient, try several
  normalisations, grid-search simple values, and keep the candidate with zero confident
  disagreements; if half-steps fail, go to quarter-steps. A snap that appears under multiple
  normalisations is more likely the "true" simple boundary.
- Re-check with different sampling seeds to make sure a snap isn't passing by luck.
