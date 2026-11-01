# Notes

## What I did

- The net is 1 input -> 8 sigmoid units -> 1 linear output, x0 in [-2, 2], regression.
- Sampled the net on a 0.1 grid (scratch/look.py) and compared with a few candidate shapes.
  The output is positive, convex, monotone increasing, about 1.00 at x0=0, about 2.72 at x0=1,
  about 0.37 at x0=-1, 0.134 at -2 and 7.32 at 2: this is exp(x0).
- Replaced the whole net with `return exp(x0)`.
- Check: PASS (snap), max |program - net| = 0.069 against a snap tolerance of 0.144
  (2% of the output range). The largest gap is at the right edge (x0 = 2), where the net
  undershoots exp (7.32 vs 7.39); elsewhere the error is mostly below 0.03.
- PASS (exact) is not reachable with a clean formula: the net's own fitting error against exp is
  ~0.07, far above the 1e-5 exact tolerance. Keeping the 8 sigmoids would be exact but
  is not a refactor.

## Heuristics

- For 1-input regression nets, tabulate the output on a coarse grid first and check the
  landmark values: f(0), f(1), f(-1), f(+-edge). Values like 1, e=2.718, 1/e=0.368, e^2=7.39
  identify exp immediately; similarly check x^2, |x|, sqrt, log, 1/x, sin.
- Check ratios between consecutive grid points: a constant ratio f(x+h)/f(x) means an
  exponential (here ~1.105 per 0.1 = e^0.1).
- Sums of many sigmoid units with mixed signs and similar slopes are usually a smooth fit of
  one elementary function; do not try to simplify unit by unit, fit the whole curve.
- The net's error against the true function is largest at the domain edges (sigmoid fits
  flatten out there), so expect the snap residual to peak at the boundary, not in the middle.
- If the clean formula is within the snap tolerance everywhere, prefer it to any
  "corrected" formula with fitted coefficients.
