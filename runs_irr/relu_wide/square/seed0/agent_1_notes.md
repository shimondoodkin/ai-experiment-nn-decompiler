# Notes

## What the network does

One float input in [-2, 2], one regression output, 32 ReLU units in one hidden layer.

- Sampling the net on a grid gives 0.001 at x0 = 0, 0.995 at x0 = 1, 1.003 at x0 = -1, about 3.99 at
  x0 = +-2. It is symmetric and convex.
- 23 of the 32 units have their kink inside [-2, 2]. Every kink adds a positive slope change
  (most about 0.43–0.49, a few small ones), so the net is convex piecewise linear. The rest have their
  kink outside the box and act as linear or zero terms there.
- A least-squares fit of a*x^2 + b*x + c over a dense grid gives a = 0.9994, b = 0.0001, c = 0.0002.
  So the target is `x0 * x0`.

## What I changed

- Replaced the whole net with `x0 * x0`.
- Check: `PASS (snap)`, empirical, 287,648 inputs. The largest error is 0.0117 at x0 = -2, against a
  snap tolerance of 0.0798 (2% of the output range). The error comes from the net's linear pieces
  between knots (chords and tangents of the parabola).
- I did not aim for `PASS (exact)`. That would mean keeping all 23 kinks, which is the net's
  approximation error rather than meaning. The parabola is the readable program.

## Heuristics

- For a 1-input regression net, first sample the output on a grid (for example 41 points) and look at
  the values at the integers and at 0. Values like 0, 1, 4 at 0, +-1, +-2 show a square straight away.
- List the kinks (-b/w) inside the box and the slope change at each (w * out_weight * sign(w)).
  If they are all the same sign and spread evenly with similar sizes, the net is a convex (or concave)
  piecewise-linear interpolation of a smooth curve. Fit a low-degree polynomial instead of keeping
  the kinks.
- Units whose kink is outside the input box are linear (or zero) across the whole domain. They can
  be folded into one affine term or dropped.
- Least-squares fit of a small polynomial basis (x^2, x, 1) over a dense grid, then round the
  coefficients to simple values (here 1, 0, 0), then check. Rounding is fine when the fit
  residual is well inside the snap tolerance (2% of the output range).
- The largest error of such a snap is usually at the box edge, where the net has fewer kinks to
  follow the curve.
