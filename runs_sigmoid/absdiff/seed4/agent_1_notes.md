# Notes

## What the network does

The output depends almost only on `d = x0 - x1`. It is a smoothed, slightly lopsided `|x0 - x1|`:
about 0.3 at d = 0, about 0.9 at |d| = 1, about 2 at |d| = 2, about 4.07 at |d| = 4, and it flattens
toward 5.2-5.3 at |d| = 6.

## What I changed

1. **Grouped the hidden units by direction.** Units 3, 5 and 7 have weights `(w, -w)` (0.71/-0.72,
   1.254/-1.254, -2.573/2.578), so they are functions of `d = x0 - x1`. They are the three soft steps
   that make the "V".
2. **Collapsed the five weak units (0, 1, 2, 4, 6).** Their pre-activation slopes are small (|w| <= 0.3)
   and they stay in a near-linear part of the sigmoid. Together they are about `0.115*x0 - 0.117*x1 + 3.59`,
   which is again a function of `d` (about `0.116*d`). They were absorbed into a linear term plus the offset.
   Their remaining curvature (about -0.0045*(x0^2+x1^2)) is the only part that is not a function of `d`.
   It spreads the output by about +-0.035 at a fixed `d`, which no function of `d` alone can remove.
3. **Wrote each step as `sigmoid(s * (d - k))`** so that each step has a slope `s` and a centre `k`:
   centres 2.454 -> 2.4, -3.58 -> -3.6, -1.02 -> -1; slopes 0.718 -> 0.7, 1.262 -> 1.2, 2.589 -> 2.5.
   I flipped the negative-weight unit with `-w*sigmoid(z) = w*sigmoid(-z) - w`, so all three steps now
   add to the output.
4. **Snapped the output weights** to 6, 2.5 and 4.5, the slope to 0.1 and the offset to -0.9.
   I picked these with a grid search over nice values. It minimised the max error against the net's
   per-`d` min/max envelope, sampled on a 241x241 grid.

Result: `PASS (snap)`, max |program - net| = 0.065 against a tolerance of 0.101.
The worst points are near the x1 = -3 edge. An exact pass would need all 8 units.
The unexplained 2D spread alone is about 0.035, and the best possible fit with 3 free steps in d
is about 0.037.

## Heuristics

- **Look for units with weights `(w, -w)` (or a common ratio).** They depend on one combined input such
  as `x0 - x1`. When all the strong units share a direction, reduce the problem to one variable.
- **Weak units are linear plus a constant.** A sigmoid unit with |pre-activation slope| * input range
  well below about 4 stays in its near-linear zone. Fit `a.x + c` to their summed contribution on a grid.
  Check the max residual before you drop them; here linear alone left 0.09, which used most of the
  budget until it was merged with the rest of the fit.
- **Rewrite `sigmoid(w*x + b)` as `sigmoid(s*(x - k))` with `k = -b/w`.** Centres and slopes are easier
  to snap than raw biases. A negative output weight can be turned positive with
  `-a*sigmoid(z) = a*sigmoid(-z) - a`.
- **Fit against the per-coordinate envelope, not the mean.** When the reduced variable leaves some
  spread, bin the outputs by `d` and record min and max per bin. Minimax fitting against `[lo, hi]` gives the true
  max error: for each bin it is max(fit - lo, hi - fit). Get those inequalities the right way round: a sign slip
  made the error look like it was 0.001.
- **Snap in stages.** First snap the nonlinear parameters (centres, slopes) to a grid. Solve the linear
  output weights by LP (minimax). Then round those weights to nice values and re-centre the constant as the
  midpoint of the residual range.
- **Leave margin.** The checker samples edges and corners, where the error was about 0.007 larger than on
  my grid. Aim to use at most about 60% of the snap tolerance.
- **Piecewise-linear is not always simpler.** A smooth V needed 5 or more ReLU knots to reach the
  tolerance, while 3 sigmoid steps in `d` fit it with lots of room.
