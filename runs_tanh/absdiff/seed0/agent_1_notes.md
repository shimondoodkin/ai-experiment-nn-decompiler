# Notes

## What the network does
Output is a regression value that depends almost only on `diff = x1 - x0`: about 0.45 on the
diagonal x0 == x1, rising smoothly and symmetrically to about 5.1 at |diff| = 6. Along the diagonal
(`total = x0 + x1`) it moves only between 0.27 and 0.48.

## What I changed
- Units 2 and 4 have weights about (-0.38, +0.38) on (x0, x1), biases about -0.79 and +0.78, and output
  weights about +3.8 and -3.87. Together they are one even "dip" in `diff`:
  `tanh(k*diff + b) - tanh(k*diff - b)`. I merged them into one shared slope 0.38, offset 0.78 and
  amplitude 3.9.
- The other six units have very small weights (|w| < 0.09) along x0 + x1. Over [-6, 6] they are close
  to linear, so they become a constant plus a small slope: `5.5 + 0.015 * total`. The constant 5.5
  also takes the saturated parts of the dip pair. Without the slope term the error is 0.16, above the
  0.098 tolerance. With it the error is 0.055.
- Snapped values: 0.38, 0.78, 3.9, 5.5, 0.015. I chose them by a small grid search over nearby round
  values, after a least-squares fit of the form `c - A*dip(k, b) + cs*total` on a dense grid.
- Result: `PASS (snap)`, max |program - net| = 0.055 (tolerance 0.098). Eight tanh units became two.

## Heuristics
- Look at the first-layer weight vectors. Units whose weights point the same way, here (-1, +1),
  depend only on that projection. Rotate the inputs into `diff = x1 - x0` and `total = x0 + x1`
  before you analyse anything.
- Two tanh units with the same slope, opposite biases (+b, -b) and opposite output weights of about
  the same size form an even bump or dip: `A*(tanh(k*u + b) - tanh(k*u - b))`. Merge them with a shared
  k, b and A.
- Units whose input weights are tiny compared with the input range (|w|*range < ~0.5) work as a
  constant plus a linear term. Fold them into one bias and a small linear term. Check whether the
  linear term is needed against the tolerance.
- Print 1-D profiles along the main projections (here diff at total = 0, and total at diff = 0). They
  show the function's shape and how big the leftover variation is, before any fitting.
- Fit a few small parametric forms with least squares, then grid-search rounded values near the fit
  and keep the simplest one that stays well inside the snap tolerance.
