# Notes

## What I found
- The dump is a 1 -> 8 sigmoid -> 1 regression net on x0 in [0.1, 5].
- Tabulating it next to candidate functions showed it is almost exactly `log(x0)`:
  net(1) = -0.007, net(2) = 0.695, net(3) = 1.096, net(4) = 1.391, net(5) = 1.603
  (log: 0, 0.693, 1.099, 1.386, 1.609). Mean |net - log| over the domain is 0.004.
- Plain `log(x0)` fails the snap check only on x0 in [0.1, 0.105]: the log goes to -2.303 at
  x0 = 0.1, while the net, built from bounded sigmoids, bends away and reaches only -2.210.
  Max error 0.093 vs tolerance 0.076.

## What I changed
- Replaced all eight sigmoid units by one formula: `max(log(x0), -2.2)`.
  The clamp at -2.2 (about log(0.11)) models the net's flattening at the lower edge.
- Result: PASS (snap), max |program - net| = 0.057 (tolerance 0.076), at x0 ~ 0.111.
- Candidates tried: `log(x0)` (0.093, fail), `log(x0 + 0.01)` (0.080, fail),
  `max(log(x0), -2.25)` (0.074, pass with little margin), `max(log(x0), -2.2)` (0.057, chosen).

## Heuristics
- For 1-input regression nets, first tabulate the net on a grid next to standard functions
  (identity, x^2, sqrt, log, exp, 1/x). A wide-range smooth monotone curve with net(1) ~ 0 and
  net(2) ~ 0.693 is a log.
- A sum of sigmoids is bounded and smooth, so it cannot follow a function that diverges at a
  domain edge (log near 0, 1/x near 0). Expect the largest error at that edge; fit the formula
  on the interior and fix the edge with a clamp (`max`/`min` at a constant near the net's edge
  value) rather than distorting the formula (shifting the argument made the interior worse).
- Measure the error profile (where |formula - net| exceeds tolerance) before adding terms;
  often the failure is confined to a tiny sliver of the domain and one clamp fixes it.
- Snap the clamp constant to a round value with margin under the tolerance, not to the
  value that only just passes.
