# Notes

## What I did

- `dump.py` is a 1 -> 8 -> 1 sigmoid net on x0 in [0, 4], regression output.
- Tabulating it at steps of 0.1 showed it tracks `sqrt(x0)` closely: net(1) = 1.001,
  net(2) = 1.412, net(3) = 1.734, net(4) = 1.992 (sqrt: 1, 1.414, 1.732, 2).
- Plain `sqrt(x0)` failed only near zero: net(0) = 0.152 while sqrt(0) = 0 (snap tolerance 0.0368).
  The net cannot follow sqrt's infinite slope at 0; it starts at 0.15 with slope about 1.6.
- Fix: `max(sqrt(x0), 1/6)`. Max error 0.028 (at x0 = 0.028), PASS (snap). A floor of 0.17 gives
  0.027, about the same; I kept the simple fraction. A piecewise line `0.15 + 1.5*x0` for x0 < 0.05
  gets 0.019 but adds a branch and two constants for no change in verdict.
- Shifted forms such as `sqrt(x0 + 0.023)` fail (error 0.05 near x0 = 0.1).
- Structure of the net, for the record: five units (h1_1, h1_3, h1_5, h1_6, h1_7) share slope
  about 0.55 and offsets -1.1 to -1.7, together a gentle concave ramp; h1_0 and h1_2 add more
  curvature; h1_4 (slope -3.9, weight -1.47) is the steep rise at the left edge. None of these
  survive as separate terms in the formula.
- PASS (exact) is not reachable with a closed form: the net is only an approximation of sqrt.

## Heuristics

- For a 1-input regression net, tabulate the output first and compare with a short list of
  standard functions (x, x^2, sqrt, log, exp, 1/x, sin). Matching values at integer points
  (here 1, 2, 4 -> 1, 1.41, 2) identifies the function quickly.
- Sigmoid nets cannot reproduce infinite or very large slopes at the domain edge (sqrt at 0,
  log near 0). Expect the error to sit at that edge; patch it with a floor/ceiling via
  `max`/`min` rather than distorting the whole formula with shifts.
- Choose the patch with the fewest constants that passes, and prefer a simple fraction for it;
  report the alternative that is more accurate but bigger.
- Groups of hidden units with nearly the same slope and staggered offsets are one smooth ramp
  piece, not separate features; they usually disappear into the closed form.
- Check the max error location reported by the checker: if all failures cluster at one edge, the
  core formula is right and only the edge needs handling.
