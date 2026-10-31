# Agent notes

## What the network does

One float input x0 in [0.1, 5], 32 ReLU units, regression output. Sampled, the output is increasing
and concave: f(0.1) = -2.27, f(1) = 0.004, f(2) = 0.70, f(5) = 1.62. Those values match the natural
logarithm (ln 0.1 = -2.30, ln 1 = 0, ln 2 = 0.69, ln 5 = 1.61), and the local slope tracks 1/x (7.3 on
the first segment near x = 0.14, about 0.23 at the right end near x = 4.4).

## What I changed

- `agent.py` is `return log(x0)`. The checker gives PASS (snap): max |program - net| = 0.0328, at
  x0 = 0.1, the left edge, where the net's first linear piece cannot follow the steep log curve. The snap
  tolerance is 0.078 (2% of the output range of 3.89).
- I also built an exact version (`scratch/exact_prog.py`), PASS (exact), max error 4.5e-7:
  - 18 of the 32 units have their kink outside [0.1, 5], so each is either always zero or one linear
    term. I folded them into a single line, 7.3224 * x0 - 3.0020.
  - The other 14 units become `d * relu(x0 - k)`, with the slope change d = c*a and the kink k = -b/a.
    A unit that is on at the left edge (a < 0) becomes its line plus `-c*a * relu(x0 - k)`.
  - Nearly all the slope changes are negative, so the function is concave. Those 14 knots are the net's
    piecewise-linear approximation of ln(x).
- I kept the formula as the deliverable because it is the readable program. The exact form is there if
  equality with the net is needed.

## Heuristics

- For 1-input nets, first sample the output on a grid and compare it with common functions (log, sqrt,
  exp, 1/x, x^2, abs, sin) at landmark points: f(1) = 0 suggests log, f(0) = 0 with slope blow-up
  suggests sqrt. Then check the best candidate directly with the checker.
- The error of a smooth formula against a ReLU approximation is largest where the curvature is highest,
  usually at a domain edge. If the formula fails only there, the net is still that formula, and the
  tolerance decides whether it passes.
- Classify each first-layer unit by its kink k = -b/a: outside the domain it is a constant 0 or a pure
  line, so fold it into one affine term. Inside the domain it is a knot with slope change c*a.
- Rewrite every in-domain unit in "left-anchored" form: a single line valid at the left edge plus
  `d_i * relu(x - k_i)` for the knots, sorted. This is exact and makes the shape readable: the signs of
  d_i give convexity or concavity, and the slope on each segment lets you test f'(x) against candidates
  (here f' ~ 1/x, so log).
- Tiny slope changes (|d| < 1e-2) at nearly the same knot as a larger one (0.472/0.482/0.483 here)
  are training noise; merge or drop them if the exact form must be shorter.
