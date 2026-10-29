# Notes

## Result

```python
def net(x0, x1):
    return abs(x1 - x0) < 0.2
```

`check.py`: PASS (snap), 0 disagreements where the net is confident; 5121 disagreements (0.77% of
inputs), all in the unsure zone along the two boundary lines.

The net says class 1 when the two inputs are within 0.2 of each other ("near equal").

## What I changed and why

- **Two steep units carry the decision.** `h1_2 = tanh(-6.888 x0 + 6.885 x1 + 1.324)` and
  `h1_3 = tanh(-6.264 x0 + 6.254 x1 - 1.275)` both have weights of about +-k on (x0, x1), so they
  are steps in `d = x1 - x0`. Their midpoints are at d = -1.324/6.886 = -0.192 and
  d = 1.275/6.259 = +0.204. Output weights +10.93 and -11.71 with opposite signs make a
  "window": high between the two steps, low outside.
- **Six units are nearly constant.** h1_0, h1_1, h1_4..h1_7 have input weights <= 0.16 and
  biases around +-1.9 (or 0.185 with a 0.10 output weight), so over [-2, 2]^2 they barely move.
  Together with the output bias they sum to about -11.06 (a linear fit gives
  -11.06 - 0.084 x0 + 0.090 x1, max error 0.09). I folded them into a constant.
- **Sign table.** With that constant: d << -0.2 gives -10.93 + 11.71 - 11.06 < 0; inside the
  window +10.93 + 11.71 - 11.06 > 0; d >> 0.2 gives +10.93 - 11.71 - 11.06 < 0. So the
  decision is `-0.2 < d < 0.2`.
- **Exact boundaries.** Solving net = 0 numerically along d for x0 in [-2, 2] gives lower
  boundaries -0.1981..-0.1997 and upper 0.1978..0.2049, so both snap to 0.2 (shared threshold,
  symmetric), written as `abs(x1 - x0) < 0.2`. The remaining drift (up to 0.005 at x0 = 2)
  comes from the slightly unequal weights on x0 and x1 in the steep units and the small linear
  tilt of the background units, and lies inside the net's unsure zone.
- I did not try for PASS (exact): the folded background is off by up to 0.09 in score, which
  moves the boundary by about 0.001, so an exact version would have to keep all 8 tanh units.

## Heuristics

- **Split units by steepness.** In a tanh/sigmoid layer, units with |w| * input range much less
  than 1 and |bias| > ~1.5 are saturated or near-linear over the domain. Fold them (with the
  output bias) into a constant, or a linear term if needed. The few steep units carry the logic.
- **Equal-and-opposite input weights mean a difference variable.** A unit with weights (-k, +k)
  depends only on `x1 - x0`; its step sits at `d = -bias / k`.
- **Two steps of opposite output sign on the same variable form a window.** A positive weight on
  an increasing step at d = a and a negative weight on an increasing step at d = b > a give
  "high when a < d < b". When a = -b, write it as `abs(d) < b`.
- **Evaluate the sign table with the folded constant** to decide which regions are class 1,
  rather than reading weight signs alone.
- **Measure the real boundary numerically** (root-find the net along the relevant direction at
  several positions) before snapping. The step midpoint `-bias/k` is only approximate (here
  -0.192/0.204) because the other terms shift the zero crossing; the actual roots were much
  closer to +-0.2.
- **Grid ties.** Checkers that sample on grids hit points exactly on a snapped boundary
  (here d = 0.2). Disagreements there are expected and fall in the unsure zone.
