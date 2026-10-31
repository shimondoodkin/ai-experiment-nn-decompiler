# Agent notes

## What the network does

A 1-input, 8-unit tanh net on x0 in [0, 4]. Tabulating it at steps of 0.1 showed it matches
sqrt(x0) to about 0.005 everywhere except near 0: the net gives 0.126 at x0 = 0 with a slope of
about 1.7, where sqrt(0) = 0 with infinite slope. A sum of eight smooth tanh units cannot follow
the infinite slope, so the net rounds off the corner.

## What I changed

- Replaced all eight tanh units with `sqrt(x0)`.
- Plain `sqrt(x0)` fails (max error 0.126 at x0 = 0; tolerance 0.037).
- `max(sqrt(x0), 1/8)` passes snap with max error 0.030. That leaves little margin.
- Final version: below x0 = 1/16, use the tangent line of sqrt at 1/16, which is `1/8 + 2*x0`.
  Above it, use `sqrt(x0)`. The constants come from the tangent (sqrt(1/16) = 1/4, slope
  1/(2*1/4) = 2), not from fitting, and they fit the net's 0.126 at 0 and its slope of about 1.7.
  Max error is 0.0165 at x0 of about 0.106, where the net dips just below sqrt.
- Result: `PASS (snap)`. An exact pass is not possible, because the net is a sum of tanh
  functions and not sqrt.

## Heuristics

- For a 1-input regression net, tabulate it first and compare with common functions (sqrt,
  log1p, x^2, exp, abs, steps). Whether it is concave, and its value and slope at the domain edges,
  usually tell you which function it is.
- Smooth nets cannot follow singular slopes (sqrt at 0, log near 0, the corners of abs). Expect
  the error against the ideal formula to sit at that point, and handle it on its own instead of
  dropping the formula.
- To patch a singular point, first try a C1 tangent-line extension at a round knot (here 1/16,
  giving 1/8 + 2x). Its constants come from the formula itself, so you don't fit any free numbers.
  A constant floor (`max(f, c)`) is simpler but less accurate.
- Check the simplest formula first. The checker's max error and where it occurs show which region
  needs a patch.
- The snap tolerance is 2% of the output range. For a function with a large range, a small
  rounded-off region near a singularity often fits in it, so a short piecewise patch is enough.
