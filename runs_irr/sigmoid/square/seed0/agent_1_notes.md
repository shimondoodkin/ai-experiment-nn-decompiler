# Notes

## What I did

- The net is 1 input -> 8 sigmoid units -> 1 linear output, x0 in [-2, 2], regression.
- Sampled the dump on a 0.1 grid over [-2, 2]. The output is symmetric in x0, 0 at x0 = 0,
  1.00 at x0 = +-1, 2.25 at +-1.5, ~3.97 at +-2: it tracks x0^2 to within ~0.01 inside the
  box and sags slightly (~0.03) at the edges, where the sum of sigmoids starts to saturate.
- The eight sigmoids are a smooth spline of a parabola: rising units (positive slope) build
  the right branch, falling units (negative slope) the left branch, and the bias and the
  small-weight units (h1_1, h1_2) fill in the flat bottom. None of them means anything alone.
- Replaced the whole net with `return x0 * x0`.
- `check.py`: PASS (snap), max |program - net| = 0.0285 (at x0 = +-2) against a snap
  tolerance of 0.0795. An exact pass would require keeping the sigmoid spline itself, so
  I did not try for one.

## Heuristics

- For 1-input regression nets, tabulate the output on a grid first and compare it with simple
  candidates (x, x^2, |x|, exp, sin, cosh) before you look at individual units. The shape is
  often obvious from 20-40 points.
- Check symmetry (f(-x) vs f(x)) and values at 0, +-1, +-half-steps. Round values at integer
  inputs (0, 1, 4) point straight at a polynomial.
- A one-layer sigmoid net whose units come in two families (positive slopes with offsets on
  one side, negative slopes mirrored on the other) is usually a smooth piecewise fit of a
  convex/even function. Read the units as spline pieces, not as features.
- Expect the largest error at the box edges: a sum of sigmoids saturates there and
  underestimates fast-growing functions. Edge sag smaller than the snap tolerance is not a
  reason to add correction terms.
- Try the simplest formula against the checker before building a piecewise version. If it
  passes with a comfortable margin (here ~36% of the tolerance), stop.
