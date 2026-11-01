# Notes

## What I changed

The dump is a 1 -> 8 -> 1 sigmoid net. Every hidden unit has a negative slope and every output weight
is positive, so the net is a smooth, monotonically decreasing, convex-looking curve on [-1, 3].

Tabulating it showed f(0) = 2.998 and an almost constant ratio between f(k) and f(k+1):
f(1)/f(0) = 0.608, f(2)/f(1) = 0.604, f(3)/f(2) = 0.617, all close to exp(-1/2) = 0.6065.
That points to an exponential decay a * exp(c * x0).

A least-squares fit of a * exp(c * x0) gave a = 3.0002 and c = -0.4997; adding a constant term
gave d = 0.0008, which is negligible. Snapped to a = 3 and c = -1/2:

    net(x0) = 3 * exp(-x0 / 2)

`check.py`: PASS (snap), max |program - net| = 0.0245, which is at x0 = -1, the edge of the box
(snap tolerance 0.085). Away from the edge the residual is below 0.01. It looks like oscillating
fit error from the net, not a missing term: the snapped formula is as good as the unsnapped fit
(0.0236 there). PASS (exact) would need the eight sigmoids kept almost as they are (atol 1e-5),
so I kept the formula.

## Heuristics

- For a 1-input sigmoid net whose units all have the same slope sign and whose output weights all
  have the same sign, the output is monotone. Tabulate it at integer inputs before analysing units.
- Check ratios f(k+1)/f(k) at equal steps. A constant ratio means an exponential a*exp(c*x), with
  c = log(ratio). Check differences too: constant differences mean a line, constant second
  differences a quadratic.
- Fit the candidate family by least squares with an extra constant term. If the fitted constant is
  about 0 and the parameters land near simple values (3, -1/2), snap them, and check that the
  snapped max error is about the same as the unsnapped fit's. If it is, the leftover error is the
  net's own approximation error and there is nothing more to recover.
- The largest disagreement of a smooth net with its formula is usually at the edges of the input
  box, where training data is sparse and the sigmoids saturate.
