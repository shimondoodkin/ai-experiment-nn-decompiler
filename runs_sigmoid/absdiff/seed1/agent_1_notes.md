# Notes

## What the network computes

A smoothed `|x0 - x1|` on [-3, 3]^2. Along the difference d = x0 - x1 the output follows |d|
within about ±0.1 for |d| in [0.25, 5]. It bottoms out at about 0.25 at d = 0 and flattens to
about 5.45 at the corners |d| = 6. Moving along x0 + x1 changes the output by at most about 0.04.

## What I changed

- The hidden units come in mirrored pairs on the difference direction (weights about (+a, -a) and
  (-a, +a), with nearly equal biases and output weights):
  - units 5 and 7: slope about 1.7, bias about -1.86, output weight about 2.95. That is a step at |d| ≈ 1.1.
  - units 0 and 6: slope about 1.45, bias about -5.65, output weight about 3.2. That is a step at |d| ≈ 4.
- Units 1–4 sit on the x0 + x1 direction and have small output weights (-0.58 to +0.27). Together
  they change the output by only a few hundredths, so I removed them and folded their average into
  the constant.
- I wrote each pair as `sigmoid(k*(d - c)) + sigmoid(k*(-d - c))`, with the slope k and the center
  c = bias/slope snapped: (1.7, 1.1) and (1.45, 4). The output weights became 3 and 3.2, and a
  grid search gave the constant -0.55.
- Plain `abs(x0 - x1)` fails: it is off by 0.55 at the corners and 0.25 at d = 0. Its wiggle
  error of ±0.11 is also right at the tolerance, so a piecewise-linear |d| was not worth pursuing.
- Result: `PASS (snap)`, max error 0.048 against a tolerance of 0.104. An exact pass is not
  possible without keeping all eight sigmoids.

## Heuristics

- Look for mirrored unit pairs: weight vectors that are near negatives of each other, with similar
  bias and output weight. Their sum `s(w·x - b) + s(-w·x - b)` is an even function of w·x, a
  "bump/ramp" in |w·x|. Several such pairs with increasing centers make a smoothed |w·x|. With
  ReLU the same pattern gives an exact abs.
- Rotate to the natural coordinates (difference x0 - x1 and sum x0 + x1). Then profile the output
  along one coordinate while sweeping the other. If the spread along the sum is small, drop the
  units that point along the sum and fold their mean into the constant.
- For sigmoid units, rewrite `sigmoid(a*u + b)` as `sigmoid(a*(u - c))` with c = -b/a. Snap the
  center c (a boundary in input units) and the slope separately; centers snap more meaningfully
  than raw biases.
- Do not drop the mirrored half of a sigmoid step pair just because it looks "off" in one half of
  the domain. At u = 0 each half contributes w*sigmoid(-b), which can be a large share of the
  constant.
- The snap tolerance is 2% of the output range. Find the ideal formula's residual profile first
  (here |d| had ±0.11 wiggles plus saturation at the corners). If the profile already uses up the
  tolerance, keep the smooth sigmoid form with snapped parameters instead.
- After fixing the structure, a small grid search over snapped candidate values (with the constant
  re-fit for each) is a quick way to pick the nicest numbers that still pass.
