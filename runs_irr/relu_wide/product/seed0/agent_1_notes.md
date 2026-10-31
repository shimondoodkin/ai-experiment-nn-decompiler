# Agent notes

## Result

`agent.py` is `return x0 * x1`. The check gives `PASS (snap)` on 376,250 empirical inputs. The largest
difference from the net is 0.057, and the snap tolerance is 0.159 (2% of the output range [-4, 4]).
Nothing is beyond the tolerance. The largest differences are at the corners (for example the net gives
-3.943 at (-2, 2) instead of -4), where the net's piecewise-linear fit is slightly flattened.

`PASS (exact)` is not reachable with a readable program: the net is piecewise linear (64 ReLU hinges) and
the product is smooth, so an exact version would just be the dump again.

## How I got there

1. **Looked at the output before the weights.** A 9x9 grid of the net over [-2, 2]^2 showed the
   multiplication table: 0 on both axes, opposite signs in opposite quadrants, and about ±1, ±2, ±3, ±4
   along the diagonals.
2. **Read the weights for confirmation.** Almost every live hidden unit has weights near (a, a) or
   (a, -a), so they are hinges along s = x0 + x1 or d = x0 - x1, with offsets spread over the range.
   That is how a ReLU net builds piecewise-linear versions of s² and d², and x0·x1 = (s² - d²)/4. The
   output weights have matching signs: (+,+) and (-,-) units add, (+,-) and (-,+) units subtract.
3. **Least-squares fit** on 200k uniform points with the basis {x0·x1, x0, x1, 1, x0², x1²}. The
   coefficients came out as 0.9992, -0.0025, 0.0017, 0.0007, 0.0006 and 0.0004. They snap to exactly
   x0·x1, with no linear, constant or square terms. The residual has mean 0.002 and standard deviation
   0.006.
4. **Dead and fixture units.** Units 12, 42 and 63 have weights of about 1e-5 or smaller, and 14, 3, 7,
   10 and 39 contribute almost nothing. They are noise and are all dropped.

## Heuristics

- **Evaluate the net on a coarse grid first.** For 1–2 inputs, a printed 9x9 table often shows the
  function at a glance (here, a multiplication table). That is quicker than reading 64 units.
- **Check the hidden-unit directions.** When most first-layer weight vectors are near (a, a) or (a, -a),
  the net works in s = x0 + x1 and d = x0 - x1. Hinges spread along both diagonals, with the
  output-weight signs set by the direction, mean (s² - d²)/4 = x0·x1. Hinges along one diagonal only,
  with a V shape, mean |x0 - x1| or a similar term.
- **Confirm a guess with least squares over a small basis** (the guessed term plus low-order
  polynomial terms). Snap coefficients within about 1% to integers, and drop terms whose coefficients
  are below about 0.5% of the output scale.
- **A smooth formula will not be `exact` against a ReLU net.** If the check reports `snap` with the
  largest errors at the corners or edges of the domain, that is the piecewise-linear fit running out of
  hinges, not a missing term.
- **Weights at or below about 1e-4 that also have tiny output weights are dead units.** Remove them
  before reading the rest.
