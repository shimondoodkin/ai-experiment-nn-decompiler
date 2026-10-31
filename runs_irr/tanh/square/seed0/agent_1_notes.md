# Notes

## What I changed

- The dump is a 1 -> 8 tanh -> 1 regression net on x0 in [-2, 2].
- Four hidden units (1, 2, 4, 6) have slopes around 0.01-0.02, so on the input box they act as constants.
  They only add to the bias.
- The other four units (0, 3, 5, 7) form two mirrored pairs:
  - 3 and 5 have slopes -1.43 and +1.50 with knees near x0 = -2 and +2;
  - 0 and 7 have slopes +0.96 and +1.08 with knees near x0 = +1.2 and -1.0.
  Together they make a smooth, even, convex bowl.
- Sampling the net on a grid gave 0.0008 at 0, about 1.00 at ±1 and about 3.97-3.98 at ±2. That is x0².
- `agent.py` is `return x0 * x0`. Check result: PASS (snap), max |program - net| = 0.028 (at the box
  edges, where the tanh units saturate a little), tolerance 0.080.
- An exact pass would mean keeping the tanh sum itself. The net is a tanh approximation of x0²,
  so the formula is the readable version.

## Heuristics

- First, print the net on a dense 1-D grid (or along axes). Reading the shape (symmetry, values at
  0, ±1, ±2) often gives the formula faster than reading the weights.
- Hidden units whose slope × input range is much smaller than 1 are effectively constants. Fold them
  into the bias before you reason about the rest.
- Pairs of units with opposite-sign slopes and mirrored offsets, and output weights of the same sign,
  point to an even function (|x|, x², bowls). Matching odd pairs point to an odd function.
- In tanh/sigmoid nets, the largest error of a snapped formula is usually at the box edges, where
  the units start to saturate. Check the corners first.
- Try the simplest polynomial (x*x, |x|, x) before a fitted one. The snap tolerance (2% of the output
  range) often accepts it.
