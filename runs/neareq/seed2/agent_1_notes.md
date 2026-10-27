# Notes

## What the network computes
Class 1 iff the two inputs are nearly equal: `|x0 - x1| < 0.2`.

`agent.py`:
```python
def net(x0, x1):
    return 0.2 - abs(x0 - x1)
```
`python check.py agent.py` -> `PASS (snap)`, 0 disagreements where the net is confident
(all differences lie in the net's own unsure zone near the boundary).

## Analysis / what I changed
- Hidden neurons 0, 4, 5 are all `relu(k*(x0 - x1) - ~0.35)` with k ~ 3.2..3.5, i.e. the same
  hinge `relu(x0 - x1 - ~0.1)` scaled; all have large negative output weights (~ -9..-10).
  Combined: about `-96 * relu(x0 - x1 - 0.1)`.
- Neurons 6, 7 are the mirror: `relu(x1 - x0 - ~0.09)` with combined weight about -77.
- Neuron 1 (`0.134*x0 + 0.024*x1 - 0.477`) is negative on the whole input box -> dead, removed.
- Neurons 2, 3 plus the bias form a small positive "budget" P (about 2..20, ~9 near the diagonal)
  that is always > 0. Inside the band `|x0-x1| < ~0.1` the output is P > 0 -> class 1; outside,
  the steep hinges overpower P. The crossing sits at `|d| ~ 0.1 + P/96 ~ 0.2`.
- I measured the actual boundary by sweeping along the diagonal (x0 = m + d/2, x1 = m - d/2):
  crossings at d in [-0.213, -0.197] and [0.200, 0.213] for m in [-2, 2]. The slight drift with m
  comes from P depending on x0 and is noise around one shared threshold, snapped to 0.2.
- So everything collapses to a symmetric band of half-width 0.2 around the diagonal.
  An exact pass would need reproducing the slight x-dependent tilt of the boundary, which is
  an artifact of the training, not part of the intended function.

## Heuristics
- Normalize each hidden neuron as `relu(scale * (direction . x - threshold))`: neurons with the
  same direction (up to sign/scale) and close thresholds are duplicates; merge them by summing
  `out_weight * scale`.
- Check each neuron's pre-activation range over the input box (interval arithmetic: sum of
  |w|*range); neurons that are never positive are dead, ones never negative are linear.
- Pairs of mirrored hinges `relu(d - t)` and `relu(-d - t)` with large negative weight plus a
  positive bias mean "|d| < t'": a band / closeness test, i.e. `t' - abs(d)`.
- Small-weight, always-positive neurons combined with the bias are often just a "positive
  offset" that shifts the threshold; estimate threshold = hinge_offset + offset / hinge_slope,
  or measure the decision boundary numerically along the relevant direction.
- For binary outputs, find the boundary empirically (sweep along the discovered direction at
  several positions) and snap the crossings to one shared round threshold; small drift across
  the domain is usually training noise.
- Directions with coefficients like (3.50, -3.50) or (-3.97, 4.00) are just +-(x0 - x1);
  look at coefficient ratios, not magnitudes.
- Try the simplest formula first with the checker; only add detail if confident points fail.
