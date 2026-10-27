# Notes

## What I did
- Read `spec.json`: two float inputs in [-3, 3], one regression output.
- First sample (0.82, 2.86 -> 2.04) suggested the absolute difference |x0 - x1|.
- Fitted a separate linear model on each side of the diagonal x0 = x1 by least squares:
  x0 > x1 gives (1.0002, -1.0001, -0.0003); x0 < x1 gives (-0.9999, 1.0001, 0.0001).
  Both snap to +/-1 coefficients with a zero intercept, i.e. y = |x0 - x1|.
- Residual vs |x0 - x1| on the samples: mean 4e-5, std 0.0019, max 0.02; the minimum output
  is 0.015 rather than 0, so the function does not quite reach zero on the diagonal.
- `agent.py`: `gap = x0 - x1; return abs(gap)`. `check.py` result: PASS (snap), max deviation
  0.030, all within the snap tolerance. The largest deviations sit on the diagonal near
  x0 = x1 ~ 2.4, where the function outputs ~0.03 instead of 0. That is a small rounding of the
  kink, not structure worth encoding, so I did not chase PASS (exact).

## Heuristics
- Look at one or two samples by hand first; a recognisable closed form (abs diff, max, min,
  sum) often shows up immediately.
- For a suspected kink, split the data along the candidate boundary and fit a linear model
  on each piece; if the coefficients are close to +/-1 or 0, snap them and recombine with
  abs/max/min/relu.
- Compare the residual's mean and std with its max: a near-zero mean and tiny std with a
  modest max point to approximation noise concentrated at the kink (the function rounds the
  corner), not a missing term.
- Check the output's minimum or maximum against the ideal formula's value (here 0 on the
  diagonal) to see where the function departs from the clean formula.
- Stop at PASS (snap) when the remaining deviations are only in the checker's unsure zone
  around a kink; reproducing them would need special-case constants that hurt readability.
