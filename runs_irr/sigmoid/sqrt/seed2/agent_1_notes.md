# Notes

## What I did

- The dump is a 1 -> 8 -> 1 sigmoid net on x0 in [0, 4] (regression). I tabulated it on a grid
  next to a few candidate functions (sqrt, log1p). It tracks sqrt(x0) closely: 1.006 at 1, 1.409 at 2,
  1.738 at 3, 1.988 at 4; for x0 >= 0.2 the gap to sqrt is at most 0.021.
- The one place it departs from sqrt is near 0: net(0) = 0.162, because a sum of sigmoids
  cannot reproduce the infinite slope of sqrt at 0. Plain `sqrt(x0)` fails there (error 0.162,
  snap tolerance 0.0365 = 2% of the output range 1.83).
- I tried shifted roots (`sqrt(x0 + 1/32)`, `sqrt(x0 + 1/40)`): they fail (error 0.05-0.06),
  because the shift also lifts the curve where it is already right. A floor works better: the
  max error of `max(sqrt(x0), c)` is 0.045 at c = 0.15, 0.036 at c = 1/6 (too tight),
  0.027 at c = 0.185, rising again above 0.19. I picked c = 3/16 = 0.1875.
- Final program: `max(sqrt(x0), 0.1875)`. Result: PASS (snap), max |program - net| = 0.0257,
  tolerance 0.0365. An exact pass would need the eight sigmoid units kept, so none of the
  dump is worth keeping: the eight units together make one sqrt curve.

## Heuristics

- For a 1-input regression net, tabulate it first and compare it against a short list of
  standard functions (x, x^2, sqrt, log, log1p, exp, 1/x, abs, sin). A whole sigmoid layer often
  collapses into one such function.
- Check the endpoints of the domain separately: smooth nets cannot follow a function with an
  infinite slope (sqrt at 0, log near 0) or a pole, so the net "rounds off" there. A floor or
  ceiling with `max`/`min` repairs the endpoint without disturbing the rest of the curve; a shift
  inside the function (sqrt(x + c)) usually spreads the error over a wider range and fails.
- Scan the repair constant over a handful of simple values and pick one with margin, not the
  first one that barely passes: the empirical checker samples many points and bisects onto
  boundaries, so a borderline constant can fail on points you did not try.
- Compute the snap tolerance (2% of the net's output range here) before choosing between
  candidates; it tells you how much smoothing near the edges you can ignore.
