# Notes

## What the network computes

The output is a round bump centred at the origin, close to `exp(-(x0*x0 + x1*x1))`. It peaks at
about 0.98 at (0, 0), is about 0.78 at radius 0.5 and about 0.37 at radius 1, and falls to about 0
beyond radius 2. A free Gaussian fit (amplitude, centre, axis widths, cross term, offset) comes out
almost exactly at amplitude 1, centre 0 and width 1 (`0.98*exp(-(0.97*x0^2 + 0.95*x1^2)) + 0.002`).

## Why the formula is not the program

The output range is about 1.03, so the snap tolerance is 0.0206 (max error, at every point). The
network has its own ripples of about ±3% everywhere, and up to ±5% at the edges and corners of the
square. Because of them:

- `exp(-(x0^2 + x1^2))` misses the net by up to 0.061 (13% of sample points are beyond tolerance);
- the best free Gaussian still misses by 0.059;
- I fitted a Gaussian plus k free tanh units to the net (least squares, 4 restarts each).
  k = 6 gave a max error of 0.042 and k = 8 gave 0.035, both still beyond tolerance. Copying the
  ripples takes about as many units as the net already has.

No hidden unit can be dropped either. Removing any one of them and refitting the output layer
(least squares) gives at least 0.031 max error.

## What `agent.py` is

It keeps the 16 tanh units:
- first-layer weights and biases rounded to 2 decimals;
- the output layer refitted by linear least squares on about 215k points (uniform and grid) against
  the original net, then rounded to 3 decimals.

Result: `PASS (snap)`, max |program - net| = 0.0066, tolerance 0.0206. Rounding the first layer to
1 decimal fails (0.048). Rounding the output weights to 2 decimals after the refit gives 0.025, which
also fails. The comment at the top of `agent.py` gives the readable meaning,
`exp(-(x0^2 + x1^2))`.

## Heuristics

- Print the output on a coarse grid first. For 2 inputs, a 13x13 table is enough to recognise a
  bump, ridge, plane or step, and to read off its centre and width (here value 1/e at radius 1 means
  `exp(-r^2)`).
- Test the guessed formula against the snap tolerance using the **max** error, not the RMS. An RMS of
  0.014 looked good, but the max error was 3x the tolerance.
- If the residual (formula minus net) is an unstructured ripple across the whole domain, it is the
  network's own fitting error. It cannot be snapped away, and modelling it costs as many units as the
  net. In that case, ship the snapped net with the formula as documentation.
- Snapping recipe for tanh/sigmoid nets: round the first layer (weights and biases) to the coarsest
  precision that still passes, then **refit the output layer by linear least squares** on the
  rounded hidden units, then round the output weights. The refit absorbs most of the rounding error.
  Here it allowed 2-decimal inputs and 3-decimal outputs at 1/3 of the tolerance.
- Greedy unit pruning with an output-layer refit is a cheap test of redundancy. If dropping the best
  unit already exceeds the tolerance, stop.
- Nonlinear refits with numerical Jacobians are very slow. Use an analytic Jacobian and a few
  thousand points.
