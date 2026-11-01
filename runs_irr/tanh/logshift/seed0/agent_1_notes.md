# Agent notes

## What I did

The dump is a 1-input, 8-tanh-unit, 1-output regression net on x0 in [-1.5, 4].
There is only one input, so I tabulated the net on a grid (scratch/look.py) instead of working through the units.

Values found: f(-1) = 0.0006, f(0) = 0.6906 (about ln 2), f(4) = 1.7904 (about ln 6), f(-1.5) = -0.679 (about ln 0.5).
The curve is concave, rising, and slows down as x grows. It has a zero at x = -1. That points to log(x0 + 2).

Final program: `return log(x0 + 2)`.

Check: PASS (snap), max |program - net| = 0.0142 (snap tolerance 0.049). The worst point is the left edge x0 = -1.5,
where log curves most and the 8 tanh units can't follow it. Inside the domain the RMS error is 0.0024 and the mean
bias is -0.0018. A PASS (exact) is not reachable with a closed form, because the net is only a tanh approximation of the
curve, accurate to about 1e-2.

All 8 hidden units were dropped. Each one is a piece of the smooth approximation and means nothing alone.

## Heuristics

- For 1-input nets, tabulate first. A table of about 20 points usually shows the function family before any unit-level work.
- Find anchor points: zeros, the values at integers, and values that match known constants (ln 2 = 0.693, ln 6 = 1.792,
  sqrt 2, pi, e). A zero at x = -c+1 plus f(-c+2) = ln 2 identifies log(x + c).
- Read the shape: concave, rising and slowing down suggests log or sqrt; convex and growing suggests exp or a square.
  Compare the ratios of successive differences to tell log from sqrt.
- The net's error is largest at the domain edges, most of all where the target curves most (near log's singularity).
  Judge a candidate by its interior fit, and expect a snap pass rather than exact.
- Try the clean closed form before fitting extra scale or offset coefficients. If it passes snap, the network's small
  bias (about 1e-3 here) is training noise, not structure.
