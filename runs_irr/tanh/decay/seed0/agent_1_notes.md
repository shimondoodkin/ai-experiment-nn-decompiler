# Notes

## What I changed and why

The dump is a 1 -> 8 -> 1 tanh net on x0 in [-1, 3], regression output. I sampled it on a grid
(41 points). The output is smooth, positive and monotone decreasing, with a falling slope:
f(0) = 2.998, f(1) = 1.821, f(2) = 1.103, f(3) = 0.665. The ratio between steps of 1 is nearly constant
(about 0.607 = e^-0.5), which points to an exponential.

A least-squares fit of log(f) against x0 gives a = 3.0011, b = 0.5004 in a * exp(-b * x0). Snapped to
a = 3 and b = 1/2:

    net(x0) = 3 * exp(-x0 / 2)

Check: `PASS (snap)`, max |program - net| = 0.016 (largest at the edge x0 = -1, where the net reads 4.930
and the formula gives 4.946); the snap tolerance is 0.085. None of the eight tanh units means anything
alone. They are a basis that approximates the curve, so the program drops all of them.

An exact pass would need the net's own edge errors, so I did not chase one.

## Heuristics

- For 1-input regression nets, sample the dump on a grid first and read the shape before looking at weights.
  Units of a smooth-activation net are usually a basis, not features.
- Test for elementary functions with simple transforms. A constant ratio f(x+1)/f(x) means an exponential.
  Then fit log(f) linearly and snap the scale and rate to simple numbers. A constant difference means a line.
  A constant log-log slope means a power.
- Look at the values at integer inputs (f(0), f(1), ...). They often show the constants directly
  (f(0) = 3 here).
- Expect the net to be least accurate at the edges of the input box. The max error of the clean formula
  is usually there, so judge a snapped formula by the residual in the interior.
