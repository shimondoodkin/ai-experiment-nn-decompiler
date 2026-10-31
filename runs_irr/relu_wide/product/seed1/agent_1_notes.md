# Notes

## What I changed and why

- `dump.py` is a 2 -> 64 -> 1 ReLU net. About 12 units have weights below 1e-4 (dead units); they contribute nothing.
- Almost all live units point along the diagonals (x0 + x1) or (x0 - x1), with output weights of
  opposite sign for the two families. That is the ReLU signature of `((x0+x1)^2 - (x0-x1)^2) / 4`,
  i.e. a product, with each square approximated by a set of hinges.
- Evaluating the net on a 9x9 grid over [-2, 2]^2 gave values that match `x0 * x1` to about 0.03
  (e.g. net(-2,-2) = 3.93, net(-2, 2) = -3.97, net(0, x) ~ 0, net(1, 1) = 0.98).
- `agent.py` is therefore just `return x0 * x1`.
- `check.py`: PASS (snap), max |program - net| = 0.073, snap tolerance 0.159, nothing beyond it.
  The largest deviations are at the corners (|x0| = |x1| = 2), where the net under-shoots the
  product because the hinge approximation of the square runs out at the edge of the box.
- PASS (exact) would require reproducing the piecewise-linear hinge approximation itself, which
  is the noise of training, not the function; I kept the formula.

## Heuristics

- Delete units whose input and output weights are both tiny (< 1e-4); they are dead.
- Look at the directions of the first-layer weights first. If they cluster on (1, 1) and (1, -1)
  with output weights of opposite signs per cluster, suspect a product x0*x1 (polarization identity
  ((a+b)^2 - (a-b)^2)/4). Clusters only on one axis with same-sign outputs suggest a square or abs.
- Before snapping individual weights, evaluate the net on a small grid (integer and half-integer
  points). A regression net's table usually reveals the formula directly (here: a multiplication table).
- Errors of a ReLU fit to a smooth function are largest at the corners of the input box; a formula
  that misses only there is the intended one, not a wrong one.
- A smooth formula cannot pass `exact` against a piecewise-linear net; accept `snap` when the
  residual is plainly fitting noise.
