# Notes

## What the dump is

4 float inputs -> 8 sigmoid units -> 3 outputs, argmax. The hidden units fall into four
correlated groups (corr > 0.9 on the input box):
- A = h0, h6, h7, roughly sigmoid(0.6 x0 + x1 - 1.5 x2 - 2.25 x3 + 3.5);
- B = h3, h4, which mainly drive class 0;
- C = h1, h5;
- D = h2 (x3 large).

The class differences are then o1 - o2 ~ 25 A - 7 B - ..., o0 - o1 ~ 16 B - 7 A - 6 C, so
the net works like a decision list: class 0 when B is on, otherwise class 1 when A is on,
otherwise class 2.

## What I did

1. Sampled the checker's own input set (seed 0) and labelled it with the net. "Confident" means
   outside the unsure zone (top-2 gap >= 0.515).
2. Fitted a logistic regression for each class pair on the confident points. All three pairs
   were linearly separable with no errors. So the 8 sigmoids reduce to two planes, used as a
   decision list:
   - `x2 + 0.94 x3 - 0.88 x1 - 0.02 x0 <= 0.5` gives class 0;
   - otherwise `x3 + 0.597 x2 - 0.412 x1 - 0.236 x0 > 1.953` gives class 2;
   - otherwise class 1.
3. Snapping. For a fixed direction w, the feasible thresholds are the interval
   [max w.x over the negative class, min w.x over the positive class]. Its width (the "gap") is
   concave in w. I maximised it with coordinate and random search, then tried rounder w:
   - Plane 0|12: max gap 0.068. 2-decimal coefficients keep a gap of 0.060, and 0.5 lies inside
     the feasible interval. Dropping x0 (gap 0.013) or using 1/0.9/0.875-style values (negative gap)
     fails.
   - Plane 1|2: max gap only 0.033. 2-decimal coefficients (0.6, 0.42, 0.24, threshold 1.92)
     pass seed 0 but fail one confident point on seed 5. That point is in a box corner (x1 low,
     x2 high, x3 low), where the net's real boundary bends. I kept the 3-decimal max-margin plane.
     No nicer forms (1/4, 2/5, 3/5, 2/3, ...) separate the classes.
4. Checked the result with `check.py` (PASS snap, 0 confident disagreements). I also ran
   `check_program` with seeds 1-5, and all passed with 0 confident disagreements.

The result is not exact: about 4k inputs in the unsure zone disagree. The sigmoid boundaries
are slightly curved, and a plane cannot follow them inside the zone.

## Heuristics

- For argmax/binary nets, first label the checker's input set with the net, keep only the
  confident points, and test whether each class pair is linearly separable in the raw inputs.
  Sigmoid/tanh nets with few hidden units often are, even when the hidden layer looks
  complicated. In that case the whole net becomes 1-2 linear comparisons in a decision list
  (one class vs rest first, then the remaining pair).
- Correlated hidden units (|corr| > 0.9 over the input box) are copies of one feature. Summing
  their output weights per group shows the decision-list order (which class each group turns on).
- To snap a separating plane, keep the direction separate from the threshold. For each direction,
  the allowed thresholds form an interval [max over the negatives, min over the positives].
  Its width is concave in the direction, so local search finds the max-margin direction.
  Then round the direction as far as the interval stays non-empty, and pick a round threshold
  inside it.
- Logistic regression is a good starting point, but it is not max-margin. A plane with a tiny
  margin passes one input set and fails on another seed. Re-check with a few other
  `check_program` seeds before trusting a snapped plane. Failures show up in box corners, far
  from the training data, where the net's boundary curves.
- When the margin is thin (gap < ~0.05 in input units), prefer 3-decimal coefficients over a
  "nicer" but fragile plane.
