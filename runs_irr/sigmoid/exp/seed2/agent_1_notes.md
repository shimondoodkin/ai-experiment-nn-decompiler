# Notes

## What I did
- The dump is 1 input -> 8 sigmoid units -> linear output, regression on x0 in [-2, 2].
- All but one unit have positive output weight times positive slope (unit 1 has negative slope
  and negative weight, so it is increasing too), so the net is monotone increasing. The units
  with steeper slopes have thresholds further right (z = 0 at x0 = 1.84, 2.06, 1.92, 2.08 for
  units 0, 2, 5, 6), which piles up curvature at the right end: a convex, growing curve.
- Tabulating the net on a 0.1 grid next to exp(x0) gave matching values everywhere
  (net(0) = 0.996, net(1) = 2.716, net(-1) = 0.368, net(2) = 7.33 vs e^2 = 7.39).
- `agent.py` is `return exp(x0)`. Check: PASS (snap), max |program - net| = 0.059 (at x0 = 2,
  the edge of the box where the net undershoots), snap tolerance 0.144.
- PASS (exact) is not reachable with a closed formula: the net differs from exp by up to 0.06,
  far above atol 1e-5. Only the original 8-sigmoid sum would be exact.

## Heuristics
- For 1-input regression nets, first tabulate the net on a coarse grid and compare with a short
  list of familiar functions (x, x^2, |x|, exp, log, sqrt, sin, sigmoid/tanh of x). A sum of
  same-direction sigmoids is just a smooth monotone curve; the formula is easier to see from the
  values than from the weights.
- Read monotonicity from signs: sign(slope) * sign(output weight) equal for all units means a
  monotone function. Steeper units placed near one end of the box signal convexity there
  (exp-like growth); units spread evenly with similar slopes signal a near-linear function.
- Check telltale values: f(0) ~ 1, f(1) ~ e, f(-1) ~ 1/e identifies exp; f(0) ~ 0 with even
  symmetry suggests x^2 or |x|.
- The worst error of a learned approximation is usually at the box edges (here x0 = 2), where the
  net saturates or undershoots; compare the formula's edge error with the snap tolerance
  (2% of the output range) before trying to fit extra coefficients.
- Prefer the clean formula over a fitted a*exp(b*x)+c when the clean one already passes snap;
  the extra constants only fit the net's training error.
