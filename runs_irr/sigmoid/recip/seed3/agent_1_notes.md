# Agent notes

## What I changed and why

- `dump.py` is a 1-input, 8-unit sigmoid layer with a linear readout on x0 in [0.5, 4].
- I evaluated the net on a grid and compared it with a few candidate functions. It matches 1/x0
  closely at every grid point: 1.974 at 0.5, 0.995 at 1, 0.502 at 2, 0.331 at 3, 0.253 at 4.
  exp(-x) and 2^-x do not fit.
- I replaced the whole net with `1 / x0`. The eight sigmoids are a smooth fit of a convex,
  decreasing curve, and no single unit means anything on its own: six units fall with x and two
  rise slightly (weights +0.35 and -0.24), and those two only correct the shape.
- Check: `PASS (snap)`, max |program - net| = 0.0257 against a tolerance of 0.0344.
  - The largest gap is at the left end, x0 = 0.5, where the net reaches 1.974 instead of 2. The
    net underfits the steep part of 1/x.
  - Everywhere else the gap is below 0.006.
  - A program that tracks the net more closely would mean keeping fitted coefficients, so I kept
    the clean formula.

## Heuristics

- For 1-input regression nets, first tabulate the net on about 15 points and compare it with a
  short list of simple functions: 1/x, sqrt, log, exp(-x), x^2, x*(c-x), and |x-c|. A smooth
  sigmoid or tanh layer with a linear readout often fits one of these, and the unit-by-unit
  structure is just a basis expansion that carries no meaning.
- Check the domain bounds against the candidates. A domain that starts at a positive value such
  as 0.5 points to functions that are singular or undefined at 0: 1/x, log x, sqrt x.
- Look at the signs of the readout weights. If most units fall with x and the readout weights are
  positive, the function is decreasing. Small units of the opposite sign usually only correct
  curvature.
- Expect the largest error at the steep end of the domain (here the small-x end of 1/x). The net
  underfits there, so judge the candidate by its error in the interior, and use the snap
  tolerance (2% of the output range) to see whether the edge gap is acceptable.
- Prefer the exact, simple formula when it passes snap. Fitting extra affine corrections
  (a/x + b) to chase the network's edge errors adds coefficients without adding meaning.
