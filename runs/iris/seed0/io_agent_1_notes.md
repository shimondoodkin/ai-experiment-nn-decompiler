# Notes

## What I did

- `spec.json` bounds (4.3-7.9, 2.0-4.4, 1.0-6.9, 0.1-2.5) are exactly the Iris feature ranges:
  x0 sepal length, x1 sepal width, x2 petal length, x3 petal width; 3 argmax classes
  (0 setosa, 1 versicolor, 2 virginica). Class 0 has the small petals and wide sepals, class 2 the large petals.
- Linear fits to the raw outputs leave a large error, so the outputs are not linear. A one-hidden-layer ReLU
  fit (torch, Adam then LBFGS) with 4 units reaches MSE about 1e-4. One of those units is never active in the
  input box, so the function is effectively 3 ReLU units. That raw program passes `check.py` (snap).
- Two of the three units are active about 97% of the time, so each pairwise decision boundary is nearly a plane.
  I sampled the fitted model densely and took the points where two classes tie for the top output. An SVD plane
  fit gave residuals of about 0.003, so both boundaries are planes. Classes 0 and 2 never share a boundary,
  so two sequential plane tests decide everything:
  - 0|1 normal ~ (-0.196, -0.808, 1, 0.92), offset 0.064 -> snapped to `x2 + 0.9*x3 - 0.8*x1 - 0.2*x0 < -0.1`
  - 1|2 normal ~ (-0.197, -0.392, 0.543, 1), offset -1.98 -> snapped to `x3 + 0.55*x2 - 0.4*x1 - 0.2*x0 > 2`
    (equivalently `x0 + 2*x1 - 2.75*x2 - 5*x3 + 10 < 0`).
- Snapping the x2 coefficient on the 1|2 plane to 0.5 instead of 0.55 broke the check badly, because
  x2 spans about 6 units and 0.05*6 shifts the plane by 0.3. The coefficient on the variable with the widest
  range has to stay more precise. I re-fitted the offset after each snap.
- Result: `agent.py` gives `PASS (snap)` with 0 confident disagreements and about 1558 in the unsure zone.
  Those are mostly at large x3, where a hidden unit switches off and the 1|2 boundary bends slightly.

## Heuristics

- Check the spec bounds against well-known datasets (Iris, etc.). Those ranges tell you the meaning of the
  variables and classes and suggest names.
- First fit linear models to the raw outputs. If the residual is large, fit a tiny ReLU MLP with increasing
  width. The width where the loss collapses is the true number of active units. Prune units that are never
  active over the input box.
- For argmax outputs, look at the decision boundaries, not the raw scores. Find near-tie points of the top two
  classes on dense samples and fit a plane to each pair with SVD. A small last singular value means the
  boundary is planar, and you can replace the whole net with a few `if` plane tests.
- Check which class pairs are ever adjacent. Pairs that never touch let you write a sequential if/elif chain.
- Normalize each plane by different coefficients (by x0, x2, x3) to find the form where the numbers are
  nicest (here 1, 2, 2.75, 5, 10).
- Snap coefficients relative to the input range: an error of delta on a coefficient moves the boundary by about
  delta * range. Snap wide-range inputs more finely, and re-fit the offset after each snap.
- Build a fast local evaluator (dense samples from a fitted surrogate, with the checker's "unsure" margin) so you
  can try many snapped variants before calling `check.py`.
