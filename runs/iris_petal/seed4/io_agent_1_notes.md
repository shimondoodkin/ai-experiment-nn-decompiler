# Notes

## What the function is

There are two inputs: x0 in [1, 6.9] and x1 in [0.1, 2.5]. The output is an argmax over 3 classes,
with class counts of roughly 350 / 1100 / 540 in the samples. The input ranges match Iris petal
length and petal width in cm, so the classes read as setosa / versicolor / virginica. The three
raw outputs are piecewise linear (ReLU-like): one global linear fit per output leaves errors of
about 2. Only the class boundaries matter for an argmax, though.

## What I did

1. A depth-4 decision tree and per-bin boundary tracing (for each x0 slice: the largest x1 of the
   lower class and the smallest x1 of the upper class) showed two oblique, nearly parallel
   boundaries with slope about -1. That points to thresholds on x0 + x1.
2. A first try with `x0 + x1 < 3.25 -> 0`, `< 6.6 -> 1`, else 2 FAILed. The 1/2 boundary is not
   at slope -1: at small x1 the net still says 1 where the sum is above 6.6.
3. I fitted the score differences (out2 - out1 and out0 - out1) by least squares, using only
   samples near each boundary (|difference| < 3). This gives the boundary much more precisely
   than a classifier fit does:
   - out2 - out1 is exactly linear near the boundary (residual 4e-6), with
     `x0 + 1.08298*x1 = 6.7551`. Snapped: 13/12 = 1.0833 and 6.75, so `12*x0 + 13*x1 = 81`.
   - out0 - out1 has two linear pieces, a ReLU kink near x1 of about 2: `x0 + 0.983*x1 = 3.262`
     and `x0 + 0.989*x1 = 3.274`. Both snap to `x0 + x1 = 3.25`.
4. With the raw coefficients and the two-piece 0/1 boundary (a `min` of the two pieces), check.py
   gives **PASS (exact)**. The snapped program in `agent.py` gives **PASS (snap)**: 0 confident
   disagreements, and 1415 in the unsure zone.
5. Snap choices: for the 1/2 boundary, 13/12 had fewer unsure-zone disagreements than 1.1
   (1415 vs 2278), and it gives a clean integer form, so I used 13/12. A coefficient of 1 there
   FAILs. For the 0/1 boundary, coefficient 1 with threshold 3.25 is within tolerance.

Final program: class 0 if `length + width < 3.25`, class 1 if `12*length + 13*width < 81`,
else class 2.

## Heuristics

- For argmax or binary outputs, model the decision boundaries, not the raw outputs. Fit the
  pairwise score difference (out_j - out_i) by linear least squares, using only samples near
  that boundary (small |difference|). For a ReLU net the difference is often exactly linear
  locally (residual about 1e-6). That gives far more precise coefficients than a classifier fit
  (SVM or logistic) on the labels.
- A max residual near 1e-6 means one linear piece. A larger residual means a kink. Split the
  samples (for example by one input) and refit each part to find the pieces. Join the pieces with
  `min` or `max` to get an exact boundary, then check whether they snap to the same line. They
  often do, and one line is enough for PASS (snap).
- Normalise each boundary by its largest coefficient (divide by the x0 coefficient) before
  snapping. Then look for 1, simple fractions (13/12 is close to 1.083), and round thresholds
  (quarter steps). Multiply through to an integer form (12a + 13b < 81) when that reads cleanly.
- When two snaps both pass, prefer the one with fewer unsure-zone disagreements, as long as it is
  still simple.
- Read the domain from the input bounds: [1, 6.9] x [0.1, 2.5] with 3 classes is Iris petal
  length and width. Name the variables to match.
- Trace a boundary per slice (the largest value of the lower class and the smallest value of the
  upper class in each bin of the other input). This shows quickly whether the boundary is axis
  aligned, oblique, or bent.
