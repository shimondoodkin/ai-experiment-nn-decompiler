# Notes

## What the network is
1 input x0 in [-2, 2], 8 ReLU units, regression output.

- Units 4 (`-0.057*x0 - 0.70`) and 7 (`0.157*x0 - 0.40`) are negative over the whole domain: dead, removed.
- The other 6 units each put one kink on the line, at x0 = -1.353, -0.714, -0.077, 0.567, 1.2374 and 1.2426.
  The last two are 0.005 apart, and between them the slope is 0, so they act as one kink at about 1.24.
- The segment slopes are -3.37, -2.08, -0.80, 0.48, 1.78, 3.23: increasing, so the function is convex, and the
  slope steps are nearly equal (about 1.29). Values: f(-2)=3.97, f(-1)=1.05, f(0)=-0.02, f(1)=1.03, f(2)=3.91.
  That is a piecewise-linear approximation of x0^2.

## Versions tried (all checked with check.py)
- `x0 * x0`: FAIL, max error 0.0917 (at the merged kink x0=1.24, where the net is below the parabola), tolerance 0.0805.
- `x0 * x0 - 0.0125`: PASS (snap), max error 0.0792. The net's error against x0^2 ranges over [-0.092, +0.067];
  its midpoint is -0.0125 = -1/80, so this offset centres the ripple. **This is agent.py.**
- `max` of 6 lines with 2-decimal coefficients (scratch/lines.py), the convex PWL written as a max of its pieces:
  PASS (snap), max error 0.011. More faithful but much less readable.

## Heuristics
- For a 1-input ReLU net, list each unit's kink `-b/w`; units whose kink is outside the input box and that are
  negative across the box are dead, and those positive across it are linear and fold into the constant/slope.
- Two kinks closer than ~1% of the domain, with a near-zero slope between them, are one kink: merge them.
- Compute the slope of every segment. If slopes increase monotonically the function is convex, and it can be
  written exactly as `max(line_1, ..., line_k)` with no ReLUs (decreasing slopes: `min`).
- Equal slope steps between equally spaced kinks mean a parabola (constant second derivative). Slope step /
  kink spacing ~ 2a gives the coefficient (here 1.29 / 0.64 ~ 2, so x^2).
- A fitted PWL of a smooth curve sits below it at the knots and above it between them (for convex curves).
  When the plain formula just misses the snap tolerance, add the constant that centres the error band,
  `(min err + max err) / 2`, rather than changing the formula.
- Always sanity-check the output range first: the snap tolerance is 2% of it, which decides whether a smooth
  formula can replace a coarse PWL fit at all.
