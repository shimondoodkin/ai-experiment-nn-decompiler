# Notes

## What I changed and why

- The dump is a 1 -> 8 -> 1 tanh net on x0 in [0.1, 5]. Tabulating it next to a few standard functions
  showed it tracks `log(x0)` to within ~0.004 over almost the whole range (net(1) = -0.002,
  net(2) = 0.6933, net(4) = 1.390, net(5) = 1.612). The eight tanh units are a fit of that one curve.
- `return log(x0)` alone failed by a hair: max error 0.0771 vs snap tolerance 0.0768, all at the lower
  corner x0 ~ 0.1, where log goes steeply to -2.303 and the net only reaches -2.225 (the net cannot
  follow the steep part of log near the edge of the box).
- Fix: floor the input at 0.11 (`log(max(x0, 0.11))`). This leaves log exact on [0.11, 5] and only
  flattens the last sliver. Result: PASS (snap), max |program - net| = 0.048. Alternatives that also pass:
  `max(log(x0), -2.2)` (0.046) and `log(x0 + 0.004)` (0.042); I preferred the floor because it keeps
  the formula untouched on 99.8% of the domain.
- PASS (exact) is not reachable with a closed form: the net is an approximation of log with ~0.003
  wiggles everywhere.

## Heuristics

- For 1-input regression nets, first tabulate the output on a grid alongside candidate functions (log,
  sqrt, 1/x, x^2, exp, sin) and look for a value match (here net(1)=0, net(2)=ln 2, net(e)=1). This
  beats trying to simplify units one at a time.
- An input range starting just above 0 (like [0.1, 5]) hints at log, 1/x or sqrt.
- Nets fitting functions with a steep end (log near 0, 1/x near 0) lag the true function at that
  corner. When the clean formula fails only at a box corner, clamp the input (or output) just inside
  the corner instead of adding terms; measure the error by region to see this.
- Check the snap tolerance (2% of output range) against the max error before adding complexity: a
  tiny fix at one edge is usually enough.
