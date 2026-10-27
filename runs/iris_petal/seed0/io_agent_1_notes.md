# Notes

## What the function is
Two float inputs, x0 in [1, 6.9] and x1 in [0.1, 2.5], with three outputs scored by argmax.
These ranges match Iris petal length and petal width, so I named them `petal_length` and
`petal_width`. The classes are 0 = setosa, 1 = versicolor, 2 = virginica.

## What I did
1. A global linear fit of each raw output left large residuals (up to 3.5), so the scores are
   piecewise-linear (a ReLU net). Reproducing the scores was not needed, because the output is argmax.
2. I drew a coarse class map from nearest samples. It showed three bands separated by two roughly
   straight, non-crossing boundaries: 0|1 at low length and width, and 1|2 at high values. Class 0
   never touches class 2 inside the domain.
3. For each boundary I took the samples of the two classes involved with |out_a - out_b| < 1 and
   fitted out_a - out_b linearly. The residual was about 1e-6, so each boundary is exactly one line
   near the decision surface:
   - 0|1: x0 + 2.0516*x1 = 4.0960
   - 1|2: x0 + 1.1003*x1 = 6.8039
4. Snapping:
   - x0 + 2*x1 < 4.1 and x0 + 1.1*x1 > 6.8 gives `PASS (snap)`. All 1264 disagreements are inside
     the function's unsure zone. This is `agent.py`.
   - The unsnapped coefficients give `PASS (exact)`. This is `agent_exact.py`, kept for reference.
   - An intermediate version (2.05 instead of 2) cut the unsure-zone disagreements to 243, but it
     still only reaches snap level and is less readable, so I dropped it.

Final program (`agent.py`):
```
if petal_length + 2 * petal_width < 4.1:   return 0  # setosa
if petal_length + 1.1 * petal_width > 6.8: return 2  # virginica
return 1                                           # versicolor
```

## Heuristics
- Read the input bounds first. Familiar ranges (here, Iris petal measurements) suggest variable names
  and the meaning of each class.
- For argmax or binary outputs, model the decision boundaries, not the raw scores. Scores of a ReLU
  net are piecewise-linear and messy, but the boundaries are often simple.
- Draw a coarse ASCII class map from nearest samples to see how many boundaries there are, their
  shape, and which classes touch.
- To fit a boundary between classes a and b, fit out_a - out_b linearly using only samples close to
  the boundary (small |out_a - out_b|). A residual near 0 means the boundary is a single linear piece
  there. A large residual means there is a kink, so shrink the window or split it.
- Normalize each boundary so one input has coefficient 1. The other coefficient and the threshold
  then become readable (x0 + 2*x1 < 4.1) and are easy to snap.
- For an ordered three-class problem with non-crossing boundaries, write it as an if-chain
  (class 0 if below boundary A, class 2 if above boundary B, otherwise class 1) instead of
  reproducing three scores.
- Count sample misclassifications for each snapped candidate before running the checker. Keep the
  simplest snap that passes, and keep the unsnapped fit as an exact fallback.
