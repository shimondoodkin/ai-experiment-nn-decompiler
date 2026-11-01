# Notes

## What I did

- The dump is 1 input -> 8 sigmoid units -> 1 linear output on x0 in [-1, 3]. All eight
  units have negative slopes and positive output weights, so the output is a smooth,
  monotone decreasing, convex curve: no kinks or steps worth keeping unit by unit.
- I sampled the net on a grid. Values: 4.92 at -1, 2.994 at 0, 1.824 at 1, 1.099 at 2,
  0.680 at 3. Ratios to the value at 0 are about 1.649, 0.609, 0.367, 0.227, which are
  e^(1/2), e^(-1/2), e^(-1), e^(-3/2). So the curve is 3 * exp(-x0/2).
- A least-squares fit of a*exp(b*x)+c gave a = 2.9991, b = -0.50008, c = 0.0005, which
  confirms the snap a = 3, b = -1/2, c = 0.
- `agent.py`: `return 3 * exp(-x0 / 2)`. Check: PASS (snap), max |program - net| = 0.0256
  against a snap tolerance of 0.0848 (2% of the 4.24 output range). The largest error is at
  the edge x0 = -1, where the net undershoots the exponential slightly (sigmoid sums flatten
  at the box edges). PASS (exact) would need tolerance 1e-5, which only a copy of the net's
  own wiggles could reach, so I kept the clean formula.

## Heuristics

- For a 1-input regression net, sample the output on a dense grid first and look at the
  shape (monotone? convex? sign of slopes and output weights) before working on units.
- If all hidden units slope the same way and all output weights have the same sign, the
  output is monotone and usually a smooth elementary function; skip per-unit simplification
  and fit the whole curve instead.
- Test for an exponential by taking ratios y(x)/y(0) at equally spaced x: a constant ratio
  per step (here e^(-1/2)) means a*exp(b*x). Check that y(x+d)/y(x) is constant across the
  range and that an added constant c fits to about 0.
- Fit a*exp(b*x)+c with three free parameters, then snap a, b, c to nice values (integers,
  halves) and confirm with the checker; the snap tolerance is 2% of the output range, so
  compare the max error against that.
- Errors of a clean formula against a sigmoid-sum net are largest at the domain edges, where
  the net's basis runs out; that is expected and not a reason to add correction terms.
