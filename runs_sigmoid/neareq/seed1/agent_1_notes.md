# Notes

## What the network does

All eight hidden sigmoids except two weak ones (h1_0, h1_7) have weights of the form `k*(x1 - x0)`:
the x0 and x1 coefficients are opposite and almost equal in size (-10.26/+10.24, -6.17/+6.23, ...).
So the net is mostly a function of one variable, `diff = x1 - x0`:

- h1_1, h1_3, h1_4: steep steps that switch on at diff ~ +0.21..0.25, each with a large negative output weight.
- h1_5, h1_6: mirror-image steps that switch on at diff ~ -0.21, also large negative output weight.
- h1_2: a step that switches on at diff ~ -0.30, positive weight. Together with the bias it makes the
  middle band positive.
- h1_0, h1_7: shallow, small, nearly constant over the domain. They only tilt the boundary slightly.

Scanning the net's sign along x1 for fixed x0 gave decision boundaries at diff = -0.198..-0.207 and
+0.1995..+0.2017 across the whole domain, which is a band of half-width 0.2 around the diagonal.

## Result

    def net(x0, x1):
        diff = x1 - x0
        return 0.2 - abs(diff)

Class 1 means "x0 and x1 are within 0.2 of each other". `check.py`: PASS (snap), 0 disagreements where
the net is confident. The 1.1% unsure-zone disagreements sit right on the band edge, where the net's
boundary drifts by up to about 0.007 (the small tilt from h1_0/h1_7). PASS (exact) would mean
keeping that sigmoid drift, which has no meaning, so I stopped at the snapped rule.

## Heuristics

- Look for hidden units whose input weights are equal-and-opposite (or proportional across units).
  This means the net depends on one linear combination, such as `x1 - x0`. Rewrite in that variable first.
- For a steep sigmoid `sigmoid(k*(z) + b)`, the switch point is `z = -b/k`. List these and group the units
  that share a direction and threshold into one step. Mirror pairs (`+z` vs `-z`) with similar
  thresholds and same-sign output weights mean an `abs(z)` band.
- Shallow, small-weight units (|k| <~ 2, |output weight| < 1) usually only tilt the boundary a little.
  Try dropping them first.
- For binary outputs, find the boundary directly: scan the sign of the net along lines and read off the
  crossings. Then snap them to a round number (0.2 here) and return a signed margin like
  `0.2 - abs(diff)`. That stays inside the language and is easy to read.
