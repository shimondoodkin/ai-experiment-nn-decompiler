# Notes

Inputs: x0 in [1, 6.9], x1 in [0.1, 2.5], 3-way argmax -> Iris petal length / petal width.

What I did:
- Whole-domain linear fits of each output had large residuals (piecewise-linear net), so instead I
  looked at the class map (nearest-sample text grid): two roughly straight, oblique boundaries,
  0|1 at small petals and 1|2 at large petals.
- Fitted each pairwise score difference (out_i - out_j) only on samples near that boundary
  (|diff| small, class in {i, j}) and normalised the x0 coefficient to 1:
  0|1: x0 + 1.72*x1 = 3.83; 1|2: x0 + 1.06*x1 = 6.72. This raw version passed (snap).
- Snapped: tried a small grid of simple coefficients with check.py. Final:
  setosa if x0 + 1.75*x1 < 3.85; else versicolor if x0 + x1 < 6.65; else virginica.
  PASS (snap): 0 confident disagreements; the rest are in the unsure zone (the 0|1 boundary is
  slightly curved, max local-linear residual ~0.57, so an exact match would need several pieces).
- Coarser snaps (1.5, 2 for the x1 coefficient; 6.7 for the 1|2 threshold) failed with confident
  disagreements, so 1.75 / 6.65 are the simplest that work.

## Heuristics
- For argmax outputs, model the decision boundaries, not the raw scores: fit out_i - out_j only on
  samples near the i|j boundary (small |difference|, class in {i, j}); global fits get polluted by
  ReLU kinks far from the boundary.
- Normalise a linear boundary so one coefficient is 1; the others then snap to readable values
  (1, 1.75, ...) and the threshold becomes a feature-space constant.
- Print a coarse nearest-sample class map first; it shows ordering of classes (here 0 -> 1 -> 2
  along a size axis), so a cascade of if/elif thresholds suffices instead of argmax of three scores.
- Recognise known datasets from the input ranges (1-6.9 and 0.1-2.5 = Iris petal length/width) to
  name variables and classes.
- Snap by trying a small grid of nearby simple values with the checker and keep the simplest that
  still passes; tighten only the term that produces confident disagreements.
