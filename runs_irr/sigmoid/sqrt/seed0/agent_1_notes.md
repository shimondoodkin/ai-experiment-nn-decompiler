# Notes

## What I found
- The dump is a 1 -> 8 sigmoid -> 1 regression net on x0 in [0, 4].
- Evaluating it on a grid gave 0.0 -> 0.136, 1 -> 1.000, 2 -> 1.414, 3 -> 1.735, 4 -> 1.997, which is sqrt(x0).
  For x0 >= 0.3 the gap to sqrt(x0) is at most 0.0067.
- `sqrt(x0)` alone fails only near 0. There the net reads 0.136 at x0 = 0 and rises almost linearly
  (slope about 1.8): a smooth sigmoid sum cannot follow sqrt's vertical tangent.
- Fix: put a constant floor under the root, `max(sqrt(x0), 0.15)`. It passes with max error 0.025, against a
  snap tolerance of 0.037. A floor of 0.16 has a slightly smaller error (0.024), but 0.15 is the rounder number
  and passes as well.
- No exact pass is possible with a closed form here: the net is a sum of 8 sigmoids, not sqrt.

## What I changed
- Replaced all 8 hidden units and the output layer with `max(sqrt(x0), 0.15)`.

## Heuristics
- For 1-input regression nets, first tabulate the output at integer and simple inputs (0, 1, 2, 3, 4) and
  compare with common functions (sqrt, log1p, x^2, exp, 1/x, sin). Values like 1, 1.414, 1.732, 2 identify sqrt
  immediately.
- When the target function has a singular slope (sqrt at 0, log at 0, abs at its kink), the net smooths it, and
  the error concentrates at that point. Patch it locally with a clamp, `max(f(x), c)` or `min(...)`, rather
  than changing the formula globally. `sqrt(x + c)` shifts the error elsewhere and fails.
- Check that a patching term stays inactive away from the problem region. A line `a + b*x` used inside `max`
  with sqrt overtakes it for larger x; a constant floor cannot.
- Compute the residual net - formula on a fine grid to see where the formula is weak before reaching for
  extra terms. Here it was below 0.007 everywhere except x < 0.05.
- Pick the roundest constant that passes, within the snap tolerance.
