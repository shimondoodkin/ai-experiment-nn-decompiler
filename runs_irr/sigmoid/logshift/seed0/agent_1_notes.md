# Notes

## What I did
- The dump is 1 input -> 8 sigmoid units -> 1 linear output, x0 in [-1.5, 4], regression.
- Sampled the net on a grid (scratch/look.py). It is smooth, increasing and concave, with
  net(-1) = 0.0002, net(0) = 0.691, net(4) = 1.7915. These match ln(x0 + 2): ln 1 = 0,
  ln 2 = 0.693, ln 6 = 1.792.
- Replaced the whole net with `log(x0 + 2)`.
- Residual net - log(x0 + 2): mean 1.5e-5, rms 0.0029, |r| <= 0.007 on [-1.25, 4]. The largest
  difference is 0.027 at the left edge x0 = -1.5, where the log curves most sharply and the
  8 sigmoids under-shoot it. The snap tolerance is 0.049.
- Check: `PASS (snap)`, max |program - net| = 0.0266, no inputs beyond tolerance.
  `PASS (exact)` would need atol 1e-5, which only the sigmoid sum itself gives; not pursued.

## Heuristics
- For a 1-input regression net, sample it on ~20 points first and read off landmark values:
  the zero crossing, the value at 0, the value at the domain ends. Match them against
  standard functions (log, sqrt, exp, 1/x, x^2) with a simple shift or scale.
  A zero crossing at an integer x = a together with f(a+1) ~ 0.693 points to log(x - a + 1).
- Concave, increasing and unbounded-looking over the range, with slope decreasing like 1/x:
  try log or sqrt. Check the slope ratio between ends: for log(x+c) it equals the inverse ratio of x+c.
- Many sigmoid units with mixed signs and similar slopes are usually a smooth-curve fit,
  not separate features. Don't simplify unit by unit; fit the whole curve with one formula.
- Look at the residual profile, not only its max. A small oscillating residual with mean ~0
  everywhere except one domain edge means the formula is right and the net is the one that
  is wrong near a singularity or steep part (here log near x = -2).
- The largest disagreement of a smooth formula usually sits at the domain edge closest to a
  singularity; compare that error with the snap tolerance (2% of the output range) early.
