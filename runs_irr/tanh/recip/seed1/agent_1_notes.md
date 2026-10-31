# Agent notes

## What the network computes

The dump is a 1 -> 8 tanh -> 1 regression net on x0 in [0.5, 4]. Sampling it on a grid showed
a smooth, monotone decreasing, convex curve: 1.976 at 0.5, 0.996 at 1, 0.497 at 2, 0.336 at 3,
0.245 at 4. These values are almost exactly 1/x0 (2, 1, 0.5, 0.333, 0.25).

## What I changed

- Replaced the whole network with `return 1 / x0`.
- Confirmed with a least-squares fit of `a + b/x0` over 2001 grid points: a = 0.00056, b = 0.9992,
  so the snapped values a = 0, b = 1 are what the data says.
- Residual net - 1/x0 is a small oscillation (|r| ~ 0.003 typical, sign changes across the range),
  i.e. the network's fitting wiggle, not structure. The largest gap is at the steep end x0 = 0.5
  (net 1.976 vs 2.0, error 0.0235), where tanh units underfit the curvature; still inside the
  snap tolerance (0.0346).
- `check.py`: PASS (snap), empirical over 201473 inputs, max |program - net| = 0.0235,
  0 inputs beyond tolerance. Exact pass is not reachable with a closed form, since the net's own
  wiggle is larger than the exact tolerance.

## Heuristics

- For 1-input regression nets, first tabulate the output on a coarse grid next to candidate
  elementary functions (x, x^2, sqrt(x), log(x), exp(x), 1/x). A positive domain that excludes 0
  (here [0.5, 4]) is a strong hint for 1/x, log(x) or sqrt(x).
- Read values at "nice" inputs (1, 2, 4): f(1) ~ 1, f(2) ~ 1/2, f(4) ~ 1/4 identifies 1/x
  immediately; f(1) ~ 0 would suggest log, f(4) ~ 2 sqrt.
- Confirm a guessed basis function g with a linear least-squares fit of a + b*g(x); if a ~ 0 and
  b ~ 1 (or another simple number), snap them.
- Check the residual's shape: an oscillation with several sign changes is fitting noise of the
  hidden units and can be dropped; a systematic trend means a missing term.
- Expect the largest error where the target has the most curvature (the steep end of 1/x near the
  low bound); smooth-activation nets underfit there. Check that this edge error is still within
  the snap tolerance rather than adding correction terms.
- Individual hidden units of such a net are not meaningful one by one (they overlap and cancel);
  look at the total function rather than trying to simplify unit by unit.
