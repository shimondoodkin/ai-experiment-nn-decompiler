# Notes

## What I did

- The dump is a 1-input regression net with 8 tanh hidden units on x0 in [-2, 2].
- I sampled it on a grid. The values rise smoothly and get steeper as x0 grows: 0.120 at -2,
  0.994 at 0, 2.712 at 1 and 7.323 at 2. That matches e^x (0.135, 1, 2.718, 7.389).
- I replaced the whole network with `return exp(x0)`.
  - Largest error against the net: 0.066, at x0 = 2, the edge of the domain where the net
    flattens out a little.
  - Snap tolerance: 0.144 (2% of the output range of 7.2).
  - `check.py` result: PASS (snap) over 206k inputs.
- No unit was worth keeping separately. Units 1, 3, 4 and 5 have very small slopes (|w| < 0.11),
  so they are close to constants plus a small linear term. The other four tanh units together
  build the convex curve. None of them is a meaningful step on its own.
- An exact pass would need the tanh terms back. The point of the network is the formula, so I
  kept the snap-level version.

## Heuristics

- For a 1-input regression net, first print a table of values on a grid. Compare the table with
  common functions (x, x², |x|, e^x, log, sqrt, sin, step) at landmark inputs: 0, ±1 and the edges
  of the domain. f(0) ≈ 1 and f(1) ≈ e point to exp.
- Measure the largest error of a candidate formula on a dense grid before calling the checker,
  and compare it with the snap tolerance (snap_tol_frac × output range).
- Smooth nets fit their curve worst at the edges of the domain, where training data is sparse
  and tanh units saturate. Expect the largest error at the edges (here at x0 = 2), not in the
  middle.
- Tanh units with a tiny slope (|w| much smaller than the input range implies) act as a
  constant plus a linear term. They are padding, not structure.
- A sum of several tanh units that gives a monotone convex curve is often a smooth elementary
  function such as exp or x². Try the function before simplifying unit by unit.
