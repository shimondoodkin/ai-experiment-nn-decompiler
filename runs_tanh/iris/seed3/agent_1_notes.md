# Notes

## What the network is

4 float inputs, 8 tanh hidden units, 3 outputs, argmax. The input ranges (x0 in [4.3, 7.9],
x1 in [2, 4.4], x2 in [1, 6.9], x3 in [0.1, 2.5]) look like a classic 3-class measurement dataset.
I did not use that guess for anything. Feature names are taken from the weights only.

## What I changed

1. **Constant unit removed.** Over the whole input box, hidden unit 3's pre-activation lies in
   [-11.0, -4.0], so `tanh` is -1 everywhere (error below 7e-4). I folded it into the output biases.
2. **Scores taken relative to class 1.** Argmax depends only on differences, so I return
   `argmax(score0, 0, score2)` with `score0 = o0 - o1` and `score2 = o2 - o1`. That removes one
   output row.
3. **Removed one more unit (7 to 6 live units) by merge and refit.** Old units 2 and 7 point in
   almost the same direction (cosine 0.99, activation correlation 0.97). I dropped unit 7,
   re-solved the output weights by least squares on the other units' activations, then refit all
   weights with a decision-aware loss. Removing any one of the other units this way left confident
   disagreements, and so did a 5-unit refit (it stalled at 6–10 confident disagreements).
4. **Snapped every coefficient with snap-and-refit.** I snapped one coefficient at a time, closest
   to the grid first: first layer to multiples of 0.1, output layer to multiples of 0.25. After
   each snap I refit the coefficients still free, and kept the snap only if no confident
   disagreement was left. Three leftovers needed a finer step (0.05 / 0.025: `0.325`, `-0.45`,
   `-0.7`). One output weight snapped to 0, so `big_a` only feeds `score2`.
5. **Named the features by their dominant inputs:**
   - `small`: x2 and x3 low. It is the main evidence for class 0.
   - `big_a`, `big_b`, `big_c`: x2 and x3 high. They are the evidence for class 2.
   - `long_x2`: x2 large compared with x1.
   - `narrow_x1`: x1 small.

   `long_x2` and `narrow_x1` lower both scores, so they favour class 1.

**Result:** `PASS (snap)`, with 0 confident disagreements. The 4815 disagreements are all inside
the unsure zone, which covers 3% of the domain. I also checked two fresh input sets built the same
way with other seeds (11 and 12, about 237k points each): 0 confident disagreements in both.
`PASS (exact)` cannot be reached once the coefficients are snapped, because tanh outputs move
continuously.

I also tried distilling into a smaller tanh net from scratch with plain MSE (2–7 units). It
reached the logits roughly but always left hundreds to thousands of confident disagreements. A
random start is much worse than starting from the original net's weights.

## Heuristics

- **Saturated units.** Check every hidden unit's pre-activation range over the whole input box.
  If |z| > 4 everywhere, tanh/sigmoid is a constant. Fold it into the next layer's bias.
- **Argmax nets.** Subtract one output from all the others (fix a reference class at 0). This
  removes one score row and shows which features separate which pair of classes.
- **Merge candidates.** Look for hidden units whose weight vectors have |cosine| > 0.95. Units
  that are nearly parallel, or anti-parallel with a different bias, are redundant.
- **Remove units by re-solving the output layer.** Drop a unit, solve the output layer by least
  squares on the remaining activations, then fine-tune.
  - Rank the units to try by the disagreement count right after the least-squares step.
  - The unit with the fewest disagreements is usually the removable one. It is not always the
    unit with the smallest output weight.
- **Decision-aware refit loss.** Use a small MSE on the score differences plus a squared hinge.
  The hinge asks the target class to beat the others by about half the net's own top-2 gap, and
  only at points where the net is confident. Plain MSE distillation leaves boundary errors that
  the checker counts.
- **Snap one coefficient at a time.**
  - Snap the coefficient closest to the grid first, refit the free ones, and keep the snap only
    if the check still has zero confident disagreements.
  - Joint quantization-aware training (straight-through rounding) on a 0.1 grid oscillated
    badly; one-at-a-time snapping was far more reliable.
  - The last few coefficients often need a finer grid (0.05 / 0.025) because nothing is left free
    to compensate.
- **Re-check on fresh points.** Re-check the final program on input sets with other random seeds,
  to make sure the refit did not overfit the checker's own sample.
- **Expected outcome for small tanh classifiers.** A full symbolic rewrite (linear rules, a
  decision list) does not hold over the whole input box. The pairwise boundaries agree with a
  linear fit on only about 98% of points. Keeping the tanh features and simplifying their
  coefficients is the realistic target.
