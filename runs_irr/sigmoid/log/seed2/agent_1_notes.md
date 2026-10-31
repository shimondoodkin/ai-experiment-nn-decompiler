# Notes

## What the network computes

1 float input on [0.1, 5], regression, 8 sigmoid units. Sampling it on a grid showed values
that match `log(x0)` to two or three decimals: -0.008 at 1, 0.694 at 2, 1.388 at 4, 1.602 at 5.
So the net is a sigmoid-sum fit of the natural log.

## What I changed

- Replaced the 8 sigmoid units with `log(x0)`. Over most of the domain the residual
  `log(x0) - net` is within [-0.013, +0.028] (x0 > 0.2).
- Plain `log(x0)` fails at the low end: at x0 = 0.1 the net gives -2.19 against log's -2.30
  (error 0.112, tolerance 0.076). The log's slope there (10 per unit) is steeper than the
  sigmoid units can follow, so the net flattens out.
- Clamping the argument, `log(max(x0, c))`, reproduces that flattening. I scanned c:
  0.115 gives a max error of 0.063, 0.12 gives 0.070, 0.125 gives 0.111 (fails). I kept 0.12
  as the rounder value. It still passes with some margin.
- Final: `return log(max(x0, 0.12))`. PASS (snap), max |program - net| = 0.070 against a tolerance
  of 0.076. The largest error is at the corner x0 = 0.1.

A linear patch (`max(log x, 6x - 2.79)`) does not work as a single max, because the line rises
above log for large x. It would need an extra condition, so the clamp is the simpler fix.

## Heuristics

- For a 1-input regression net, sample it on a grid first and compare it with the common
  elementary functions (log, sqrt, exp, x^2, 1/x). Positive-only domains like [0.1, 5] are a
  hint for log, sqrt or 1/x.
- The net-minus-formula residual tells you where the net is underfit. Large errors at a domain
  edge where the formula is steep (log near 0, 1/x near 0, exp at the top) are typical:
  a finite sum of sigmoids cannot follow an unbounded slope.
- Fix edge underfit by clamping the formula's argument (`f(max(x, c))` or `f(min(x, c))`) and
  scan c on a fine grid. The clamp copies the net's flattening and leaves the interior unchanged.
- Find the corner where the error peaks (here x = lo) and check it directly. Snap errors in
  smooth 1-D fits peak at the domain ends.
- Prefer a rounder snap constant when it still passes with margin. If a rounder value fails,
  keep the less round one that passes.
