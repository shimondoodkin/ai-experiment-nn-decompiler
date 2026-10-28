# Notes

## What the network does
- 4 inputs -> 8 sigmoid units -> 3 linear outputs, argmax.
- The 8 hidden pre-activations are almost collinear (|corr| 0.91-1.00 over the box): they all
  measure roughly one direction, "+x2 +x3 -x1 (-x0 a little)". Units come in sign-flipped pairs
  (h0/h6 vs h4/h5, h2/h3 vs h1/h7).
- Over the box only two class pairs ever compete: 0|1 and 1|2. Classes 0 and 2 never touch.
  So the classes are ordered along that direction: class 0 low, class 1 a middle slab, class 2 high.
- I bisected about 2500 points onto each decision boundary (where the two top outputs are equal)
  and ran an SVD on them. The smallest singular value was 0.001-0.002 (vs 0.5-0.9 for the
  other directions), so both boundaries are planes to within about 0.002 on average and at most
  0.025 in input units. The sigmoids bend them only a little.

## Program
    low_score  = x2 + 0.75*x3 - 0.66*x1 - 0.14*x0   ; class 0 if low_score  < 0.4
    high_score = x3 + 0.62*x2 - 0.42*x1 - 0.27*x0   ; class 2 if high_score > 1.8
    otherwise class 1

- Plane fits (leading coefficient fixed at 1): 0|1 x2 = 0.138 x0 + 0.661 x1 - 0.749 x3 + 0.393;
  1|2 x3 = 0.263 x0 + 0.424 x1 - 0.625 x2 + 1.846.
- Snapping: the unsure zone is narrow (score < 0.49 with outputs spanning about 10), so the
  coefficients can't go much past 2 decimals. Simpler fractions failed: 1/8, 2/3 or 0.15 for
  the 0|1 plane, and 0.25 with 0.4 for the 1|2 plane. I searched a 0.01 grid around each fit for
  zero confident disagreements and kept the candidate with the fewest decimal digits. Integer
  versions also exist, but they need scale 15-19 (15*x2 + 11*x3 - 10*x1 - 2*x0 < 6;
  19*x3 + 12*x2 - 8*x1 - 5*x0 > 36), so they aren't more readable.
- Result: PASS (snap), with 0 confident disagreements on the checker set and on 4 more input
  sets (seeds 11-14, about 243k points each). PASS (exact) isn't reachable with planes, because
  the true boundaries are slightly curved sigmoid level sets.
- If the inputs are the classic 4 flower measurements (in the order sepal length, sepal width,
  petal length, petal width), then the rule reads "petal size beats sepal width". That is only a
  guess from the ranges and is not used anywhere.

## Heuristics
- For argmax nets, first find which class pairs are actually adjacent. Look at the top-2 pair
  where the gap is small. Non-adjacent pairs need no boundary in the program. If the adjacent pairs
  form a chain (0|1, 1|2), the program is an ordered if-chain with one plane per link.
- Fit boundaries directly instead of the outputs. Bisect random segments whose ends have different
  classes onto the boundary, then SVD the points. A tiny last singular value means the boundary is a
  plane, whatever the activation function, and the singular vector gives the coefficients.
  Fitting the output logits linearly gave poor R^2 (0.6) even though the decision boundaries
  were near-perfect planes.
- Strongly correlated hidden pre-activations (|corr| > 0.9) suggest the net is mostly one-dimensional,
  and its smooth sigmoid boundaries are close to planes.
- Normalise each plane so the dominant input has coefficient 1. Then snap the other coefficients
  and the bias by a grid search that minimises "decimal digits used" subject to zero confident
  disagreements. The unsure-zone width decides how far you can snap. Check the result on a few
  input sets with other seeds before trusting it.
- Integer-coefficient planes only appear at large scales (15-20) when the margin is tight. Prefer
  2-decimal coefficients in that case.
- Separate class-ordering checks (`if low < t0: return 0`, `if high > t1: return 2`) are safe
  when the two boundaries don't cross inside the box. Check that no confident class-0 point lies
  above the second plane.
