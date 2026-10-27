# Notes

The function is `score = 0.5 - |x0|`, decision `score > 0`, i.e. "x0 is within 0.5 of zero"
(a band / window detector centred on 0).

How it was found: sorted the samples by x0 and looked at finite-difference slopes. Slope is +1
for x0 < 0 and -1 for x0 > 0 with a single kink at x0 = 0 (peak value ~0.5). That is a scaled,
shifted absolute value. Residual of `0.5 - abs(x0)` against all samples: max 6e-8 (float noise),
so no extra hidden kinks. `check.py` gives PASS (exact).

## Heuristics

- For 1-D inputs, sort by input and compute finite-difference slopes; ReLU nets are piecewise
  linear, so the slope list directly shows the pieces and the kink locations. Ignore isolated
  slope blips between two adjacent close samples (numerical noise, not real kinks).
- Slopes of +a / -a meeting at one kink mean `c - a*|x - k|` (or `a*|x - k| + c`); read c from the
  value at the kink and k from the kink position.
- Snap slopes to integers and constants to simple fractions, then verify with the max residual over
  all samples; a residual near float precision (~1e-7) confirms the exact form.
- For a binary output, return the score itself rather than the 0/1 decision: it lets the checker
  confirm the exact form, not just the boundary.
- Name the constant by its role (here the half-width of the accepted window).
