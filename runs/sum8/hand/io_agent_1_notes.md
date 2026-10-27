# Notes

The function is `relu((x0 + ... + x7 - 1000) / 50)`: the sum of all eight byte inputs, minus a fixed
offset of 1000, scaled by 1/50 and clipped at zero. (You could also read it as 0.02 per unit above
an average of 125 per input.)

How I found it:
- A linear least-squares fit gave nearly equal coefficients, about 0.011 each, for all eight inputs,
  which pointed to a symmetric sum. 46% of the outputs were exactly 0, which pointed to a ReLU clip.
- Two nonzero rows were enough to get the slope. The sums 1144 and 1069 differ by 75 and their
  outputs differ by 1.50, so the slope is 0.02 = 1/50. The intercept then comes out at exactly 1000.
- I checked it against all 2000 samples: the largest error is 3e-6. `check.py` returns PASS (exact)
  with a largest difference of 4.6e-6.

## Heuristics
- Fit a linear model to the whole dataset first. If the coefficients come out nearly equal, try a
  plain sum of the inputs. A fitted slope that is too small, with a large intercept, means some
  of the outputs are clipped.
- A large share of outputs that are exactly 0 points to `relu` on a single affine expression. To
  get the true slope and intercept, fit only the nonzero rows, or difference pairs of nonzero rows.
- Outputs that sit on a fine grid (here multiples of 0.02) show the slope as a simple fraction of
  integer inputs. Snap it to 1/k and snap the threshold to a round number (here 1000).
- Write the threshold as a subtraction on the named sum, then divide by the constant. This keeps the
  program readable and exact.
