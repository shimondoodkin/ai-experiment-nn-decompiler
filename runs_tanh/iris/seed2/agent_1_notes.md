# Notes

Result: `agent.py` gives PASS (snap) against the net, with 0 disagreements where the net is confident.
It also had 0 confident disagreements on my own 2.4M-point validation set (uniform points, face/corner-biased
points and a 25^4 grid). The dump itself is only PASS (exact) within the unsure zone: float64 vs float32
already flips about 870 points there.

## What I changed

1. **Constant unit.** The pre-activation of unit 7 is in [-12.4, -4.5] over the whole box, so tanh is -1
   there. I folded it into the output biases.
2. **Dead unit.** Unit 1 has small output weights. Setting it to 0 changed no confident decision on 1.66M
   test points, so I removed it.
3. **Near-duplicate units merged.** Units 0, 2 and -6 (unit 6 negated, because tanh is odd) all compute
   about tanh(-x1 + x2 + x3 - 0.6), and their output weights are in proportion. I replaced them with
   one unit: the mean of the three affine rows, with the summed output weights. That left 4 tanh units.
4. **Scores relative to class 1.** Argmax only depends on score differences, so I subtracted o1 and
   return `argmax(score0, 0, score2)`. That removes one output row, and unit `x3_vs_x1` drops out of
   score0 entirely (its weight there was 0.14).
5. **Snapping.** I snapped greedily, trying coarse steps first (1, 0.5, 0.25, 0.1, 0.05, 0.01) for every
   weight. I kept a snap only if a pool of 0.5M confident points near the boundary still had 0 errors.
   Output weights became small integers or halves: -11, 2, -1.5, 1.5 / -5, 5, 3, -6.
   Unit weights ended on 2-decimal values.
6. **Tried and rejected: pure linear rules.**
   - Class 0 vs the rest is almost a halfspace, about x2 + 1.05 x3 < 0.945 x1 + 0.45.
   - Class 2 vs class 1 is close to the AND of two halfspaces.
   - The LP max-margin was only about 0.003-0.008, measured in x3 units. A single class-0 line already
     failed at box corners: the boundary bends where the dominant unit saturates. That left no room to
     snap, and it would have needed 4+ lines with 4-decimal coefficients.
   - The 4-unit tanh form is shorter and snaps much better.

Reading of the program: score0 is driven almost entirely by `-11 * petal_vs_x1`. Class 0 is "x2 + x3
small relative to x1". Class 2 vs 1 is mainly `small_petal` and `petal_vs_x0`, which is "x2 and x3
large", with the bar raised for large x0 and x1. If the inputs are the Iris measurements, these are
setosa / versicolor / virginica.

## Heuristics

- Check each unit's pre-activation range over the input box first. A unit saturated over the whole box
  (|z| > ~4 for tanh/sigmoid) is a constant; fold it into the next layer's bias.
- Ablation test: set a unit to 0, or to its mean, and count confident disagreements on a large pool. Units
  with small outgoing weights often drop out with no change.
- For odd activations (tanh), compare weight rows up to sign: w and -w are duplicates. When normalized rows
  and outgoing-weight ratios agree, merge them into one unit (mean row, summed outgoing weights). Try it
  even when rows differ by ~20%; argmax decisions tolerate it.
- For argmax outputs, subtract one class's score from all of them: `argmax(s0, 0, s2)`. One fewer row, and
  units whose weight differences are ~0 disappear from a score.
- Before trying to fit linear rules, check how much slack the unsure zone allows: run an LP max-margin
  separator on confident points. If the margin is tiny, piecewise-linear rules will not snap. Keep the
  activation form and snap its weights instead.
- Build the test pool with box-face/corner-biased samples and points bisected onto the net's class
  boundaries, not just uniform samples. Simple rules failed first at corners.
- Snap greedily, coarse step first, accepting only zero confident errors. Snap output weights first
  (they become integers easily), then unit rows. Validate on a fresh independent pool afterwards.
