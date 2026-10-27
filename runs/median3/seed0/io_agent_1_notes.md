# Notes

## What I did

- Spec: 3 float inputs in [-3, 3], one regression output.
- A linear fit (coefficients about 0.4 each, max error 1.7) was poor, so the function is not linear.
- I tested simple symmetric order statistics against the output: max, min, sum, median.
  The **median of the three inputs** matched with max |error| 0.043 over 2000 samples
  (median |error| 0.004, slope of out against median 1.0007, intercept -0.001).
- The leftover errors are small, with no pattern, and are largest where two inputs are close
  together or near the corners of the box. That is the smoothing error of the network
  around the kinks of min/max, not extra structure, so I did not model it.
- `agent.py`: `median = max(min(x0, x1), min(max(x0, x1), x2))`.
- `check.py`: `PASS (snap)`, max |program - net| = 0.092 against a snap tolerance of 0.120,
  0 inputs beyond tolerance. Every remaining difference is in the unsure zone (two inputs nearly
  equal, or close to the domain edge). `PASS (exact)` would mean copying the network's
  approximation noise, which would not make the program more readable or more correct.

## Heuristics

- Fit a linear model first. If the coefficients are all about equal (here about 1/3 each) but the
  fit is bad, the function is probably a symmetric nonlinear combination of the inputs, such as an
  order statistic (min, max, median) or a sorting-network piece.
- Compare the output directly against cheap candidate features (min, max, median, sum, mean, abs,
  relu of each input or of sums) by their max error before trying any regression. One candidate
  with a near-zero error explains the whole function at once.
- Confirm a candidate by regressing out on it. A slope of about 1 and an intercept of about 0 mean
  there are no hidden scale or offset constants to snap.
- Look at the largest residuals. If they cluster where inputs are close together (the kinks of
  min/max) or at the edges of the domain, they are the network's approximation error, not missing
  terms. Stop there instead of adding correction terms.
- Median of three in the min/max-only language: `max(min(a, b), min(max(a, b), c))`.
- Name the intermediates (low, high, median) so the formula reads as what it computes.
