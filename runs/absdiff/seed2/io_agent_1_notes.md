# Notes

## What I did and why

- `spec.json`: two float inputs in [-3, 3], one regression output.
- First rows of `samples.csv` matched `|x0 - x1|` by eye (e.g. 0.82, 2.86 -> 2.04; -1.38, -2.64 -> 1.26).
- Residual `out - |x0 - x1|` over all 2000 samples: mean -0.0001, max 0.0116. The largest
  residuals all sit next to the diagonal x0 == x1, where the function stays slightly above 0
  (about 0.01 to 0.035) instead of reaching 0. That is the rounded bottom of the "V" a small ReLU
  net gives when it approximates abs(). It is not a real feature, so I did not model it.
- Program: `gap = x0 - x1; distance = abs(gap)`, the absolute difference (distance between two
  numbers on a line). Coefficients snapped to exactly +1/-1 with no offset.
- `check.py`: PASS (snap), max |program - net| = 0.0168, with all of it at the diagonal
  (unsure zone). I did not add a fudge term to chase an exact pass. That would mean fitting
  approximation noise.

## Heuristics

- Test the simplest symmetric forms first on the raw rows: x0+x1, x0-x1, |x0-x1|, max, min,
  products. A few rows by eye are often enough to guess, then check the residual on all samples.
- Look at where the largest residuals are, not just how big they are. If they cluster on a kink
  (here x0 == x1), they are the net's smooth approximation of a piecewise-linear function, and the
  clean formula is right.
- A small constant floor near a kink of abs/relu/max (output about 0.01 to 0.03 where it should be 0)
  is a sign of approximation. Snap it to 0 and don't add an offset.
- Name intermediate quantities (`gap`, `distance`) so the formula reads as a concept.
- Stop at PASS (snap) when the only disagreements are in the unsure zone around a kink. Fitting
  them would encode noise, not the function.
