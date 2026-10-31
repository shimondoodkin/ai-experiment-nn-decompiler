# Notes

## What the network does

One input `x0` in [0.1, 5], regression output, 8 tanh hidden units. Tabulating the dump over the
domain showed the output tracks `log(x0)` closely: 0 at x0 = 1, 0.693 at 2, 1.099 at 3, 1.609 at 5,
-0.693 at 0.5 (errors below 0.01 over most of the range).

## What I changed

- Replaced the 8 tanh units with `log(x0)`.
- Plain `log(x0)` failed only near the lower bound: at x0 = 0.1 the net gives -2.209 while
  log gives -2.303 (error 0.094, snap tolerance 0.076). The net cannot follow the steep part of
  the log curve with its smooth units, so it levels off. Flooring the log at -2.2
  (`max(log(x0), -2.2)`, active for x0 < 0.111) fixes this: max error 0.060, `PASS (snap)`.
- Alternatives that also pass: `log(max(x0, 0.11))` (max err 0.062) and `log(x0 + 0.005)`
  (max err 0.048, smallest error but the 0.005 shift is less meaningful). I kept the floor, since
  it changes the formula only at the edge where the net is known to be inaccurate.
- `PASS (exact)` is not reachable with a short formula: the net differs from log by up to ~0.02
  in the interior (wiggles from the tanh fit), far above the 1e-5 exact tolerance.

## Heuristics

- For 1-input regression, tabulate the net on a grid and print candidate elementary functions next
  to it (sqrt, log, 1/x, x^2, exp). Fixed points like f(1) = 0, f(2) = 0.693 identify log at a glance.
- Domains that start just above 0 (e.g. [0.1, 5]) hint at log, 1/x or sqrt, which are undefined or
  steep at 0.
- When the obvious formula fails only at a domain edge, look at where the function is steepest: a
  small smooth net under-shoots steep regions. Clamp the formula there (max/min with the net's edge
  value) instead of hunting for a different formula.
- Compare error against the snap tolerance (2% of output range) early; it tells you how much edge
  deviation can be absorbed and whether a fix is needed at all.
- Prefer the fix that keeps the true formula intact over most of the domain (a floor) over one that
  shifts it everywhere (an additive offset inside the log), even if the shift has slightly lower error.
