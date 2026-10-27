# Notes

## What the function is
The inputs are Iris features (x0 sepal length, x1 sepal width, x2 petal length, x3 petal width;
the bounds are the Iris dataset ranges), and there are 3 argmax outputs, so the classes are setosa / versicolor / virginica.

## How I got there
1. A linear least-squares fit to each raw output left errors up to about 7, so the outputs are not linear. That means a ReLU net.
2. I fitted 4-H-3 ReLU MLPs (Adam, then LBFGS, in float64) to the raw outputs. H=8 reproduced them to 5e-6.
   Only 4 hidden units were ever active on the samples: one was always on (purely linear), two were on for about 98%
   of samples, and one was on for 40%. The other 4 were dead.
3. A program written straight from that fit (4 decimals) already gave PASS (snap).
4. Simplifying while keeping the argmax unchanged:
   - I subtracted the versicolor score from all three scores, so versicolor = 0. Argmax is invariant to a shared shift.
   - I replaced the two 98%-active units with their linear parts. The checker showed no change, so their kinks
     lie where they don't affect the decision.
   - That left setosa = linear, virginica = linear + c*relu(h).
     - On the box, setosa > 0 and virginica > 0 never both happen. So argmax becomes a 2-step decision list:
       "setosa if s>0, else virginica if v>0, else versicolor".
     - The setosa score never sees an active h in the region that matters, so the h term dropped out.
     - virginica: v = max(L, L + c*h) > 0 is the same as L > 0 or (L + c*h) > 0. Within the box, the L > 0 branch
       never fires alone, so a single half-space is left.
   - Each half-space test can now be rescaled on its own. I normalised by the petal-length coefficient and snapped
     the coefficients to 0.1 steps. The bias was searched on a fine grid, measuring disagreement against the fitted
     net on 300k uniform points. Then I confirmed with check.py.
5. Final (agent.py):
   - setosa if petal_len + 1.1*petal_wid + 0.1*sepal_len < 0.8*sepal_wid + 1.65
   - virginica if petal_len + 1.8*petal_wid > 0.4*sepal_len + 0.7*sepal_wid + 3.4
   - versicolor otherwise
   - Result: PASS (snap), with 0 disagreements where the function is confident and 1124 in the unsure zone.
   - A less rounded version (-x0 + 8x1 - 10x2 - 10.5x3 + 15.75 > 0; -x0 - 1.75x1 + 2.5x2 + 4.5x3 - 8.5 > 0) cuts the
     unsure-zone disagreements to 453. Coarser snaps (0.25/0.5 steps, dropping sepal terms) led to confident disagreements.

## Heuristics
- Fit a linear model first. A large residual means there are hidden ReLUs.
- Then fit small ReLU MLPs of growing width in float64 (Adam, then LBFGS) and take the smallest width that drives the
  residual to ~0. Count how often each unit is active on the samples:
  - dead units: drop them;
  - always-on units: fold them into the linear part;
  - units that are almost always on: try replacing them with their linear part.
- For argmax outputs, subtract one class's score from all scores (shared shift invariance) so that class becomes 0.
- If two non-reference scores are never positive together, turn the argmax into an if/elif decision list of
  half-space tests. Each test can then be rescaled on its own, which makes snapping much easier.
- c*relu(h) with c > 0 inside a "score > 0" test is an OR of two half-spaces. Check whether one of them is redundant on the input box.
- Snap each hyperplane by normalising on its dominant coefficient, rounding the weights to a grid, and re-searching the
  bias (the threshold absorbs most of the rounding error). Score candidates by disagreement rate on many random
  in-box points against the fitted surrogate, and only run the real checker on the best ones.
- Name the inputs from their bounds when they match a well-known dataset (here Iris).
