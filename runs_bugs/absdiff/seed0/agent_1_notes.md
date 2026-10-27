# Notes on net (2 inputs -> 8 ReLU -> 1, regression)

## What the network does
Roughly a valley along the diagonal x0 = x1 (an asymmetric |x0 - x1|): slope about 0.75 when
x0 > x1, and steeper (about 0.8 at first, then about 1.1 after a second kink) when x1 > x0.
On top of that are two soft "plateau steps" of height about 0.5: one in x0 (between x0 = -0.7 and
x0 ~ -(x1+1)/6) and one in x1 (between x1 ~ 1/3 and x1 ~ -0.4). Because of those steps the valley
floor is not 0. The net gives 0.8 at the origin and 0.09 at (-3,-3). So it is not a clean
|x0 - x1|, and a clean formula alone fails the snap tolerance (about 0.12 = 2% of the output range).

## What I changed
- Rewrote every unit as `c * relu(x_dom + k * x_other + t)` (divided by the largest input weight),
  which makes the slopes and kink positions readable.
- Dropped neuron 4 (output weight 0.15, inner scale 0.22, so effective weight about 0.03). After
  refitting it costs only 0.02 max error.
- Grouped units into pairs that act together:
  - h2 + h7: two kinks on the x1 > x0 side of the diagonal (`above`, `above_steep`). They could not be
    merged: dropping either one costs about 0.11.
  - h5: the x0 > x1 side (`below`), relu(x0 - x1 + 1/8).
  - h0 - h3: an x0 ramp minus a near-equal x0 ramp = a bounded step (`x0_step`).
  - h1 - h6: the same pattern in -x1 (`x1_step`).
- Snapped the inner weights (0.7, 1/6, 1/8, 1/3, 0.4, 0.25, 0.875, 0.85, 0.075, 0.15). For each one
  I refit the rest with least squares and kept the snap only if the max error stayed small. Setting
  the small x0 cross-terms in the x1-step units to 0 failed (0.13 to 0.23 error), so those stay at
  0.075 and 0.15.
- Snapped the output weights to multiples of 0.05 with a small grid search that minimizes max error
  over a 301x301 grid. The bias came out 0 and was removed.
- Result: `PASS (snap)`, max |program - net| = 0.074 (tolerance 0.121). An exact pass would need the
  original unsnapped weights, including neuron 4.

## Heuristics
- Normalize each hidden unit by its largest input weight. Then the inner weights become slope
  ratios and kink offsets, which are easy to snap, and the scale goes into the output weight.
- Look for pairs of units with almost the same direction and output weights of opposite sign:
  `a*relu(z + t1) - b*relu(z + t2)` is a bounded step / saturating ramp (a clamp), not two features.
- Look for pairs with opposite directions (x0 - x1 and x1 - x0): together they form an |difference|
  valley, often with different slopes on each side.
- Rank units by effective contribution (|output weight| x |inner weight| x input range) and try
  dropping the smallest. Always refit the remaining output weights before you judge a drop.
- Snap one parameter at a time and refit everything else (nonlinear least squares, starting from
  the dump's weights). Keep a snap only if the max error stays well under the tolerance. Small
  cross-terms (about 0.1) can matter when the unit's output weight is large; test them before
  zeroing them.
- Snap output weights last, with a discrete search that minimizes the max error (not L2), since the
  checker uses a max-error tolerance.
- Check how the net behaves on a coarse grid first (diagonal, corners). If an "obvious" formula
  misses by more than the tolerance at some point, keep the correction terms rather than forcing it.
