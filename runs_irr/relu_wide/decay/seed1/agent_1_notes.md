# Notes

## What I did

- The dump is a 1-input ReLU net (1 -> 32 -> 1), so it is a piecewise-linear function of x0.
  I computed every hidden unit's kink `-b/w` and kept the 20 inside [-1, 3]. The other 12 units are
  either always off or always linear on the domain. Then I printed the slope and value on each piece.
- The slope rises steadily from about -2.30 at x0 = -1 to about -0.38 at x0 = 3. The function is
  decreasing and convex, and all values are positive. That points to an exponential decay.
- Values at the integer points: f(0) = 2.9996, f(1) = 1.8147, f(2) = 1.1040, f(-1) = 4.937,
  f(3) = 0.665. Successive ratios are about 0.605 ~ e^(-1/2), and f(0) is about 3. So
  f(x0) ~ 3 * exp(-x0/2).
- Check: max |program - net| = 0.0092 over 279k inputs, against a snap tolerance of 0.085
  (2% of the 4.27 output range). Result: `PASS (snap)`. The largest error is at the kink near
  x0 = -0.41, where the net's linear pieces cut the convex curve.
- I did not aim for `PASS (exact)`. That would mean keeping roughly 20 kinks with their trained
  coefficients, which is just the dump again and not a formula. The 2-term closed form is the
  readable version.

Final program: `return 3 * exp(-x0 / 2)`.

## Heuristics

- For a 1-input ReLU net, list the kinks (`-b/w`) inside the input box, then the slope and value on
  each piece. Units whose kink is outside the box are constants or plain lines on the domain and
  fold into the bias and slope.
- Slope that changes in one direction (convex or concave) with many small kinks means a smooth
  curve, not a real piecewise rule. Try the standard smooth families before snapping kinks:
  exp, log, sqrt, 1/x, x^2.
- To test for an exponential, look at ratios of values at evenly spaced points (f(x+1)/f(x)).
  If they are constant, the rate is log(ratio). Snap the rate to a simple fraction (here -1/2)
  and the scale to f(0) (here 3).
- Compare the smooth formula's max error with the snap tolerance (2% of the output range) before
  running the checker. A piecewise-linear net with ~20 kinks fits a smooth curve to well within 1%.
- Many tiny units (|w| ~ 1e-3 or less, or very small output weights) are training noise. Ignore
  them when you look for the shape.
