# Notes

## What the net does

The dump is a 1 -> 8 -> 1 tanh net on x0 in [0, 4]. Sampling it shows it computes the square root:
at x0 = 0.25, 1, 2.25, 4 it gives 0.497, 0.998, 1.498, 1.996. A least-squares fit `a*sqrt(x0) + b` gives
a = 0.999 and b = -0.0002, so it is `sqrt(x0)` with no scale and no offset.

## What I changed

- I replaced the eight tanh units with `sqrt(x0)`.
- `sqrt(x0)` alone fails only at the left end. sqrt has infinite slope at 0, and the tanh sum cannot follow
  that, so at x0 = 0 the net gives 0.127 instead of 0. Its slope there is about 1.9. The error 0.127 is
  above the snap tolerance (0.037, 2% of the output range). Everywhere else the error is below 0.015.
- I tried two fixes:
  - `sqrt(x0 + 1/64)` shifts the curve so that it starts at 0.125. It fails at x0 ~ 0.077, where it is
    0.04 too high.
  - `max(sqrt(x0), c)` adds a floor. With c = 1/8 it passes with max error 0.032. With c = 0.15 the
    max error is 0.023, because the floor sits between the net's value at 0 (0.127) and its value where
    the floor meets sqrt (0.169 at x0 = 0.0225). I kept c = 0.15 for the larger margin.
- Result: `max(sqrt(x0), 0.15)`, PASS (snap), max |program - net| = 0.023. No exact version is
  possible with a simple formula: the net is a smooth approximation of sqrt and differs from it by up to
  0.015 away from 0.

## Heuristics

- For a 1-input regression net, tabulate it at round inputs (quarters, perfect squares) and compare
  with candidate functions (sqrt, log1p, x^2, exp). Values like 0.497, 0.998, 1.498, 1.996 at 0.25, 1,
  2.25, 4 point straight to sqrt. Then fit `a*f(x) + b` by least squares to read off scale and offset.
- Smooth activations (tanh, sigmoid) cannot follow an infinite slope (sqrt at 0, log at 0). Expect the
  largest error at the end of the domain where the target's slope blows up, not in the middle.
- Fix such an end with a clamp (`max(f(x), c)` or `min`), not by shifting the argument: a shift moves
  the whole curve near the end, a clamp changes only the region where the net flattens.
- Choose the clamp level between the net's value at the boundary and its value where the clamp meets
  the formula; that balances the two worst errors. A round number in that range (here 0.15) keeps the
  formula readable.
- Look at the error over the whole domain before snapping: if the formula is within a few percent
  everywhere except one end, only that end needs a special case.
