# Notes

## What I changed and why

- The dump is a 1 -> 8 -> 1 sigmoid net on x0 in [0.5, 4] with a regression output.
- I evaluated it on a grid of 15 points and compared with candidate formulas (1/x, log x, sqrt x).
  It matched 1/x closely everywhere (2.000 -> 1.986 at x=0.5, 1.000 -> 0.998 at x=1, 0.250 -> 0.249 at x=4).
- `agent.py` is just `return 1 / x0`. `check.py` gives PASS (snap): max |program - net| = 0.0139,
  at the left edge x0 = 0.5, where the net undershoots the steep 1/x curve; the snap tolerance is 0.0347.
- PASS (exact) would need to keep essentially the whole net (atol 1e-5); even the smallest unit
  (h1_7, weight -0.0062) changes the output by ~1.7e-3 across the domain, so no unit can be pruned exactly.
  The formula is the useful, readable result.

## Heuristics

- For 1-input regression nets, first tabulate the net on a coarse grid and compare against a short list
  of elementary functions (1/x, log, sqrt, x^2, exp, sin, abs). A domain that avoids 0 (here [0.5, 4])
  hints at 1/x or log.
- Many sigmoid units all with negative input slopes and positive output weights (a decreasing,
  convex sum of decreasing steps) suggest a decreasing convex function such as 1/x or exp(-x).
- The largest error of a smooth-function fit is usually at the domain edge where the target is steepest;
  check there first against the snap tolerance.
- Don't chase PASS (exact) by pruning sigmoid units: every unit contributes a non-constant term well above
  1e-5, so exact equality needs the full net. Prefer the closed formula if it passes snap.
