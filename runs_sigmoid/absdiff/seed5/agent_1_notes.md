# Notes

## What the network does

- The 8 hidden sigmoid units split into two groups by input direction:
  - 5 units with weights (w, -w) on (x0, x1), so they only see `d = x0 - x1`. Three face +d
    (centres d ~ 0.94, 1.21, 3.63) and two face -d (centres d ~ -1.07, -3.75).
  - 3 units with weights (w, w), so they only see `s = x0 + x1`. Their output weights are tiny
    (0.11, -0.035, 0.03): together they vary by only ~0.047 over the domain (about 1% of the output range).
- Along d, the output is a smoothed ramp close to `|x0 - x1|`. It is ~0.25 at d = 0, tracks |d| within
  about ±0.09 for |d| in [1, 5], and bends down to ~5.4 at |d| = 6. The two sides of d differ by at most ~0.035.
  So the net is a soft "absolute difference" of the inputs.
- Plain `abs(x0 - x1)` does not pass: it is off by 0.25 at d = 0 and by 0.6 at the corners, and the
  tolerance is 0.104.

## What I changed

1. Dropped the 3 s-units and folded their mean into the constant. The cost is about ±0.024.
2. Made the model symmetric in d and wrote it in terms of `dist = abs(x0 - x1)`. On each side, the
   units pair up into two steps: one near |d| ~ 1.1 (net weights 1.55+0.38 on +d and 2.31 on -d) and one
   near |d| ~ 3.7 (4.07 on +d and 3.50 on -d).
3. Fitted `c + A*sigmoid(w1*(dist - t1)) + B*sigmoid(w2*(dist - t2))` to the net's envelope over s.
   For each d, I took the min and max of the net over all valid s, then did a minimax fit. After that
   I grid-searched snapped values: w1 = 2.5, t1 = 1.1, w2 = 1.25 (half of w1), t2 = 3.7, A = 1.75,
   B = 3.8, c = 0.07.
4. The result has 2 sigmoids instead of 8, and every number is a short decimal.
   Check: `PASS (snap)`, max |program - net| = 0.049 against a tolerance of 0.104.

Final program:

```python
dist = abs(x0 - x1)
return 0.07 + 1.75 * sigmoid(2.5 * (dist - 1.1)) + 3.8 * sigmoid(1.25 * (dist - 3.7))
```

## Heuristics

- **Group hidden units by weight direction.** Normalise each unit's input-weight vector. Units with
  (w, -w) only see x0 - x1, and units with (w, w) only see x0 + x1. That turns a 2-D net into a sum of 1-D
  functions of a few projections.
- **Measure a group's total swing before keeping it.** Multiply each output weight by the
  sigmoid's actual variation over the domain. A group whose swing is well under the snap tolerance
  becomes its mean, folded into the bias.
- **Look for mirrored units, then use abs.** Units that face +d and -d with similar centres and
  similar summed amplitudes form a symmetric function of d. Rewriting it as f(|d|) halves the unit count.
  Check the asymmetry first, by comparing net(d) with net(-d), so you know how much of the budget it uses.
- **Several sigmoids in a staircase usually approximate a ramp** (here, |d|). Compare the net with the
  simple formula. If the formula alone fails, because of the soft corner at 0 and saturation at the
  domain edge, keep a small number of steps and refit them instead of forcing the ideal formula.
- **Fit to the envelope, not to samples.** For a model in one projection, compute for each value of
  that projection the min and max of the net over the remaining directions. Then do a minimax fit to
  (lo+hi)/2 with error |f - mid| + half-spread. It is fast (1-D) and matches the checker's max-error criterion.
- **Snap in a grid search.** Fix slopes to nice values (2.5, 1.25), then grid-search centres and
  amplitudes in steps of 0.05–0.1. The constant has a closed form: (max(hi-f) + min(lo-f))/2.
  Sigmoid-sum fits have nearly flat valleys, so sequential snap-and-refit wanders. A small grid over
  rounded values works better.
- **Write sigmoid arguments as `w * (x - centre)`**, so the threshold can be read directly.
