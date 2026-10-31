# Notes

## What the net does

On the domain x0 in [0.1, 5]:

- Units 0, 2, 3, 4 and 6 have a negative slope and a negative bias, so their pre-activation is
  negative for every x0 > 0. They are always 0, so I removed them.
- Units 1 and 5 have kinks at x0 = 0.063 and x0 = 0.022, both below the lower bound, so they are
  always active and act as plain linear terms.
- Unit 7 kinks at x0 = 0.1031, which is just inside the domain (0.3% of its width).

When the linear terms are merged, the exact net is

    0.05909158 * x0 - 0.65885715 + 0.47629807 * relu(x0 - 0.10308627)

That form gives PASS (exact) with max error 6.5e-7. The piece on [0.1, 0.103] is too short to
matter. A least-squares line over the domain fits it with max error 0.0015, about 0.06% of the
output range, which is -0.65 to 1.97.

## Final version (`agent.py`)

    return 0.535 * x0 - 0.708

It gives PASS (snap) with max error 0.002, against a tolerance of 0.052. Nicer constants also pass
but lose much more accuracy:

- `0.54*x0 - 0.71` has max error 0.021.
- `(x0 - 1.3) * 8/15` has max error 0.014.

I kept the 3-decimal fit because no simple fraction is clearly what the net is computing. The net
crosses zero at x0 = 1.323. If the target is a smooth curved function, this net learned only a
straight line for it.

## Heuristics

- **Find dead units first.** On a domain with lo > 0, a unit with slope < 0 and bias < 0 is always
  0. More generally, a unit is dead when its pre-activation is ≤ 0 at both domain endpoints.
- **Find always-active units.** A unit whose kink (-b/w) lies outside the domain, with a
  pre-activation that is positive on the domain, is linear. Fold it into one slope and one
  intercept. After this, a 1-input ReLU net is a short list of (kink, slope change) pairs.
- **Drop kinks near the boundary.** A kink within a few percent of a domain bound changes the
  output only on a tiny interval. Replace the two pieces with one line fitted over the whole
  domain, a least-squares fit rather than either piece's own coefficients. Then check the max
  error.
- **Use the snap tolerance as a budget.** Measure the max error of each candidate rounding, here
  3 decimals, 2 decimals, or a fraction like 8/15. Prefer the simplest one that keeps a clear
  margin. Do not round further when no candidate is clearly meaningful.
- **Keep the exact merged form in the notes** as a fallback when the simplified form drops a
  kink.
