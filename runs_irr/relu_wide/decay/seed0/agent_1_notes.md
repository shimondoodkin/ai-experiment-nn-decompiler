# Notes

## What I did

- The net is 1 input -> 32 ReLU units -> 1 output, so it is a piecewise-linear function of x0 on [-1, 3].
  11 of the 32 units have their kink outside the box (they are constant or linear there), and the other
  21 kinks are spread fairly evenly over the box.
- I listed the pieces: the slope goes smoothly from -2.31 at x0 = -1 to -0.23 at x0 = 3, always negative and
  always getting less steep. The function is decreasing and convex: a smooth curve sampled by a spline,
  not a few meaningful breakpoints.
- Values at integer inputs: f(-1) = 4.934, f(0) = 2.995, f(1) = 1.815, f(2) = 1.097, f(3) = 0.666. Each step of 1
  multiplies by about 0.605-0.607, which is exp(-1/2) = 0.6065. So the curve is exponential.
- A least-squares line fit to log(f) gives slope -0.502 and amplitude 2.996. I snapped these to -1/2 and 3.
- Result: `return 3 * exp(-x0 / 2)`. The checker gives PASS (snap): max |program - net| = 0.0121, against a
  snap tolerance of 0.085. The largest error is at x0 = -1, the edge of the box, where the spline undershoots
  the curve a little.
- I did not look for an exact piecewise-linear rewrite: with 21 real kinks it would not be any more readable
  than the dump, and the formula explains the net.

## Heuristics

- For a 1-input ReLU net, compute the kink positions -b/w and drop the units whose kinks are outside the input
  box. Then tabulate the slope on each piece. Many kinks with slopes that change steadily and monotonically mean
  a smooth function approximated by a spline: look for a closed formula rather than snapping breakpoints.
- Find the curve family from ratios and differences at evenly spaced points. A constant ratio f(x+1)/f(x) means
  exponential (a*exp(k*x)). Constant differences mean linear, constant second differences mean quadratic, and a
  linear log-log plot means a power law.
- Fit the family's parameters with least squares (for an exponential, a linear fit of log f), then snap them to
  simple values (3, -1/2) and check the snapped formula against the net. The 2% snap tolerance is usually wide
  enough to absorb the spline's own error.
- The largest disagreement between a spline net and its smooth target is usually at the edges of the input box,
  where the net has the fewest kinks and the least training data. Check that the edge errors fit inside the
  tolerance.
- Units with tiny output weights (|v| < 0.01) or kinks outside the box hardly matter. Once a formula fits, there is
  no need to account for them one by one.
