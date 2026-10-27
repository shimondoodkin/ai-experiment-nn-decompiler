# Agent notes

## What the network does
4 inputs (ranges match the iris dataset: x0 sepal length, x1 sepal width, x2 petal length,
x3 petal width) -> 8 ReLU units -> 3 class scores, argmax. It is an iris-style classifier:
class 0 = small petals, class 2 = large petals, class 1 = in between.

## What I changed and why
- Dead units: interval arithmetic over the input box showed h1_1 (max pre-activation -0.63)
  and h1_5 (all weights and bias negative, inputs positive) are never active. Removed.
- h1_0 is always active (min 0.12), so it is just a linear term. h1_7 is active on ~6% of the
  box and has tiny, nearly equal output weights - irrelevant to the decision.
- Instead of keeping the 8x3 layer, I sampled 3M uniform points over the box, labelled them
  with the net, kept only the points where the net is confident (top-2 score gap above the
  checker's unsure-zone margin), and found that the classes are separable by a two-step
  decision list with linear cuts:
  1. class 0 vs rest, 2. class 2 vs 1 among the rest.
- Max-margin LP (with the x2 coefficient fixed to 1) gave the feasible coefficient region;
  I then tried simple fractions / small integer ratios and measured the gap between the two
  confident populations on 3M points, picking the simplest coefficients with a positive gap
  and a threshold in the middle of the gap:
  - class 0 if  x2 + 1.1*x3 - 0.2*x0 - 0.6*x1 < 0.7   (feasible threshold range 0.64..0.75)
  - else class 2 if  2.75*x2 + 5*x3 - x0 - 2*x1 > 10.25 (feasible range 10.09..10.41;
    equivalently 11*x2 + 20*x3 - 4*x0 - 8*x1 > 41)
  - else class 1.
  Rounder candidates (e.g. x2 + x3, or 3*x2 + 5*x3) had negative gap, i.e. misclassified
  confident points, so they were rejected.
- Result: `PASS (snap)` with 0 confident disagreements; also re-checked with sampling seeds
  1-5 (all pass with 0 confident disagreements). `PASS (exact)` would need the full hidden
  layer (only removing the dead units h1_1, h1_5 keeps it exact), which I traded for the
  readable two-rule form.

## Heuristics
- Bound each hidden unit's pre-activation over the input box with interval arithmetic
  (positive weight -> hi, negative -> lo). Units with max <= 0 are dead; units with min >= 0
  are linear and can be folded.
- For argmax outputs, look at score differences between classes rather than raw scores; units
  with nearly equal output weights across classes do not affect the decision.
- For small classifiers, label a dense uniform sample with the net, drop the points inside the
  checker's unsure zone (top-2 gap < margin), and test whether a one-vs-rest decision list of
  linear cuts separates the remaining points (LinearSVC / LP). Order the list by the most
  isolated class first.
- Snap coefficients by fixing the largest coefficient to 1 (or to a nice integer scale), solve
  a max-margin LP for the feasible region, then enumerate simple fractions and keep the
  simplest set whose confident-point gap stays positive; put the threshold in the middle of
  the gap (rounded to a half-step or quarter-step).
- If simple fractions fail at one scale, rescale the whole inequality (e.g. divide by the
  smallest coefficient) - ratios like 1:2:2.75:5 can be much cleaner than the x2=1 form.
- Verify on a larger sample than the checker uses and with other random seeds, since the
  confident gap for a linear cut can be narrow.
