# Notes

## What the function is
`out = 0.2 - |x0 - x1|`, decision `out > 0`: "are x0 and x1 approximately equal (within 0.2)?".

## How I found it
- Nearest-sample class map on a coarse grid showed the positive class only on a thin diagonal band x0 ~= x1.
- A plain linear fit was poor (max error 2.58), so I least-squares fit a library of candidate features
  (x, |x|, x0*x1, x^2, |x0-x1|, |x0+x1|, 1). The fit was exactly `-1*|x0-x1| + 0.2` with residual ~2e-7,
  i.e. the raw score itself is piecewise linear and exact, not only the decision.
- Coefficients already sat on snap values (-1 and 0.2), so no snapping was needed.

## Result
`python check.py agent.py` -> `PASS (exact)`. The 206 listed disagreements are all at the exact
boundary |x0-x1| = 0.2, where float rounding gives -1.7e-16 vs the function's tiny positive value;
all are in the "unsure" zone, none where the function is confident.

## Heuristics
- First plot (or nearest-sample map) the decision on a grid; a band along a diagonal suggests |x0 - x1|
  (or |x0 + x1| for the anti-diagonal), a ring suggests a distance/norm.
- Fit the raw score, not just the class: regress `out` on a small library of ReLU-network-friendly
  features (inputs, |inputs|, |sums/differences| of input pairs, constant). A near-zero residual
  means the formula is exact; sparse non-zero weights name the formula directly.
- ReLU nets naturally produce abs-values (relu(a)+relu(-a)), so include |x_i - x_j| and |x_i + x_j|
  features early.
- Express the result with named quantities (gap, tolerance) so the program reads as a rule
  ("within tolerance"), and keep the score form to get exact rather than just decision agreement.
- Boundary-only disagreements of order 1e-16 are float rounding at score == 0, not a model error.
