# Notes

## What the function is

There are 4 float inputs, and their bounds (4.3–7.9, 2.0–4.4, 1.0–6.9, 0.1–2.5) are exactly the Iris
feature ranges. The output is an argmax over 3 classes. So this is an Iris classifier: x0 is sepal
length, x1 sepal width, x2 petal length and x3 petal width. The classes are 0 setosa, 1 versicolor
and 2 virginica.

## What I did

1. **Logits are not linear.** A least-squares linear fit of each logit leaves residuals of 5–7. A
   one-hidden-layer ReLU MLP fit to the logits still leaves errors of 0.2–0.6 with 8 hidden units, so
   the underlying net has many kinks. I stopped trying to recover it exactly.
2. **Decision boundaries are close to linear.** Multinomial logistic regression misclassifies 2 of
   the 2000 samples. A linear SVM for each pair of classes misclassifies 2 (0 vs 1), 1 (1 vs 2) and
   0 (0 vs 2). Classes 0 and 2 almost never touch. That gives a chain: setosa | versicolor |
   virginica.
3. **Fitting each boundary from the logit differences.** I regressed `out_j - out_i` on the inputs,
   using only samples near the i/j boundary where the third class is lowest. The 1/2 boundary is
   locally linear to about 0.01. The 0/1 boundary has residuals of about 0.5 because the net bends it
   in the corners of the input box (large x0, small x2, large x3).
4. **Snapping.** I normalised each boundary so the x3 coefficient is 1. Then I rounded the other
   coefficients and re-ran `check.py`:
   - The x0 coefficient is -0.2 (that is, -x0/5) in **both** boundaries, so it is shared.
   - 0/1: `x3 + 0.9*x2 - 0.55*x1 - 0.2*x0 - 0.5`. This had fewer unsure-zone disagreements (1026)
     than the raw least-squares fit (1168).
   - 1/2: `x3 + 0.55*x2 - 0.4*x1 - 0.2*x0 - 2.05`.
   - Coarser snaps failed with confident disagreements: rounding x1 to -0.5 in both boundaries, x2 to
     0.5 in the 1/2 boundary, or the bias off by ±0.05–0.1. The bias is the most sensitive
     parameter.
5. **Result:** `PASS (snap)`. There are 0 disagreements where the function is confident and 1026
   in the unsure zone, out of 200016 inputs. The remaining disagreements come from the bent 0/1
   boundary at the box corners, far from real Iris data. I did not get `PASS (exact)`, because that
   would need the full piecewise net.

## Heuristics

- Read the input bounds first. Matching them to a known dataset (here Iris) gives you variable names,
  the class order and the expected boundary structure (a chain of classes).
- For argmax outputs, model the pairwise **decision boundaries**, not the logits. The logits of a ReLU
  net can be highly nonlinear while the class boundaries stay almost flat.
- Check which class pairs actually border each other: a pairwise linear SVM, or a count of points
  where the third class is lowest. If the classes form a chain, an if/elif cascade of two
  hyperplanes is shorter than an argmax over three scores.
- Fit a boundary by regressing the logit difference on samples **near** that boundary, with the
  third class excluded. The residual tells you whether the boundary is flat (about 0.01) or kinked
  (about 0.5).
- Normalise each hyperplane by one coefficient, preferably one shared across boundaries, before
  snapping. Look for coefficients that come out the same in several boundaries.
- Snap one coefficient at a time and re-check. Bias terms tolerate much less rounding than slopes.
  Re-tune the bias after snapping the slopes, because that can even reduce disagreements.
- Use a local evaluator on the samples that counts disagreements only where the margin
  (top logit - second logit) is above the checker's unsure threshold. It screens candidates before
  the slower full check.
