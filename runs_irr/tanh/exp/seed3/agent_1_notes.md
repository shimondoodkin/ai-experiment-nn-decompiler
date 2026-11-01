# Notes

## What the network computes

`dump.py` is a 1 -> 8 -> 1 tanh net on x0 in [-2, 2]. Tabulating it on a 0.1 grid gave
0.119 at -2, 0.994 at 0, 2.712 at 1 and 7.335 at 2: positive, monotone, convex, value about 1 at 0
and about e at 1. That is exp(x0). Comparing side by side with exp(x0), 2^x0 and x0^2 left no doubt.

## What I changed

All eight tanh units and the output layer are replaced by `return exp(x0)`.

- Check: `PASS (snap)`, empirical over 205,963 inputs. Max |program - net| = 0.054 (at x0 = 2),
  snap tolerance 0.144 (2% of the output range of about 7.2).
- The worst disagreement is at the right edge, where exp is steepest and the net's tanh units are
  already saturating. In the middle of the range the net and exp agree to about 0.01.
- `PASS (exact)` would mean keeping the network's own sum of tanh units, which is the dump, not a
  formula. So I did not try for it.

No unit-by-unit simplification was needed. The units have no individual meaning: four nearly
flat units (slopes about ±0.1, saturated near ±0.9) together act as a constant plus a small linear
term, and the steeper units (slopes -1, -0.86, -0.98, +1.5) bend the curve. They are a basis that
fits a smooth convex curve, not separate pieces of logic.

## Heuristics

- For a 1-input regression net, tabulate it on a fine grid before reading the weights. Values at
  x = 0, 1, -1 and the sign of the curvature quickly identify common functions (exp: 1, e, 1/e;
  2^x: 1, 2, 1/2; x^2: 0, 1, 1; log, sqrt, 1/x by their domain behaviour).
- Compare the table against a few candidate closed forms side by side. When one matches to within
  about 1% of the range everywhere, try it directly in the checker before fitting anything.
- Small tanh/sigmoid nets fitting a smooth function are usually a basis expansion. Nearly flat,
  saturated units (|slope| about 0.1, |bias| > 1) add up to a constant plus a slight tilt. Don't
  try to give them meaning one by one; identify the overall function instead.
- Expect the largest error at the edges of the input box, especially where the target is steepest
  (here the right end of exp). Check that error against the snap tolerance before deciding the
  closed form is wrong.
- If the formula is the net's intended function, keep it unscaled (no fitted 0.99·exp(x)). A clean
  formula within snap tolerance is better than a refit that only chases the net's training error.
