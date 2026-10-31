# Notes

## What I did

- The dump is a 1-input regression net: 8 sigmoid units, all with slope 1.0-3.8, plus a bias of 0.89.
  Reading the weights unit by unit tells little. All units are increasing except two small negative-weight
  units with negative slope, which also add to an increasing function. So the net is monotone increasing and convex-looking.
- I tabulated the net on a 0.1 grid over [-2, 2] (`scratch/look.py`) next to x^2 and exp(x). The values are
  0.139 at -2, 0.368 at -1, 1.002 at 0, 2.718 at 1 and 7.34 at 2, which is exp(x0) to about 3 digits everywhere
  except the last tenth before x0 = 2, where the net bends below exp (7.34 vs 7.39).
- `agent.py` is `return exp(x0)`. Check: PASS (snap), empirical over 214k inputs, max |program - net| = 0.048
  (at x0 = 2) against a snap tolerance of 0.144. No exact program is possible without keeping the sigmoid sum,
  because the net is only an approximation of exp.

## Heuristics

- For a 1-input regression net, tabulate the output on a regular grid before reading weights. Compare it with
  common candidates (x, x^2, |x|, exp, log, sqrt, sin, 1/x) at landmark points: f(0), f(1), f(-1), and the
  ratio f(x+1)/f(x). A constant ratio (here e) means exp; constant second differences mean a quadratic.
- Many sigmoid units with similar positive slopes and staggered offsets, all feeding positive output weights,
  are a sum-of-steps approximation of a smooth monotone convex function. Do not snap them unit by unit; fit the
  whole curve.
- The largest net-vs-formula error is usually at the edge of the input box, where the training data ends and the
  last sigmoid saturates. Check the edges first when a formula seems close.
- If a simple formula passes the snap check by a wide margin, stop there: the remaining error is the network's
  approximation error, not structure that a refactor should keep.
