# Notes

## What the network is

A 1 -> 32 -> 1 ReLU net on x0 in [0.1, 5], regression. A 1-input ReLU net is a piecewise-linear
curve, so I first sampled it. The values match the natural logarithm closely:
net(1) = -0.0009, net(2) = 0.6949 (ln 2 = 0.6931), net(4) = 1.3857 (ln 4 = 1.3863),
net(0.1) = -2.282 (ln 0.1 = -2.303), net(5) = 1.6098 (ln 5 = 1.6094).

## Final program (`agent.py`)

    def net(x0):
        return log(x0)

`check.py`: PASS (snap), max |program - net| = 0.0205 (at x0 = 0.1, the steepest end, where
the net's linear pieces cut under the curve), against a snap tolerance of 0.0778 (2% of the
output range 3.89). The mean offset is -0.0009, so the bias needs no constant term; adding
-0.001 makes the largest error worse.

## Exact alternative (not submitted)

`scratch/agent_exact.py` is the pruned network. Of the 32 units, 15 have their kink outside
[0.1, 5]. Those that are always on fold into one line, and those that are always off are
dropped. That leaves a line plus 17 kinked ReLU terms. It gets PASS (exact), max error 3.4e-7.
It describes the same curve with 17 knots, so I kept `log(x0)` as the readable version.

## Heuristics

- For a 1-input ReLU net, compute each unit's kink -b/w. Units whose kink falls outside the
  input box are linear or zero on the domain. Fold the always-on ones into one affine term and
  drop the always-off ones. That alone gives an exact, smaller program.
- Before snapping individual weights, tabulate the net at landmark inputs (0.1, 0.5, 1, 2, 4,
  the bounds) and compare with standard functions (log, sqrt, 1/x, x^2, exp). A positive domain
  that starts near 0 (here [0.1, 5]), with net(1) close to 0 and net(2x) - net(x) roughly
  constant, points to log.
- Kink density shows curvature. Many kinks crowded near the low end of the domain, with large
  output weights, mean the function is steep and curved there (log, 1/x, sqrt). Knots spread
  evenly suggest a quadratic.
- The largest error against a smooth formula usually sits at the most curved end of the domain,
  where the piecewise-linear fit is coarsest. Judge the formula by the max error relative to
  the output range, not by single points.
- Check whether a constant offset helps by looking at the mean residual, not the max. A tiny
  mean residual means no offset.
