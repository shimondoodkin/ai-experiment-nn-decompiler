# Notes

## What the network does
2 inputs in [-2, 2], 16 ReLU units, regression. It is a convex piecewise-linear bowl: about 0.92 at the
origin, about 2.23 at (±2, 0) and about 3.0 at the corners. The values match `sqrt(1 + x0^2 + x1^2)`
(1.40 at (1,0) vs sqrt 2, 1.735 at (1,1) vs sqrt 3, 3.0 at (2,2)). That formula is off by up to 0.077,
though, mostly in a sharp dip at the origin, where the net has a cone and the formula is flat. So it fails
the check (tolerance 0.042), and the program keeps the net's piecewise-linear shape.

## Changes
1. **Dead units removed.** I evaluated each unit's pre-activation at the 4 box corners (it is affine, so its
   range over the box is set by the corners). Units 1, 3, 7 and 12 are never positive, so they always
   output 0.
2. **Always-on unit folded.** Unit 4 is positive over the whole box, so it is linear and went into the linear
   term and the constant.
3. **Output weights folded in.** All live output weights are positive, so `c*relu(z) = relu(c*z)`. That left
   11 relus plus an affine term. Greedy pruning with a least-squares refit could not drop any of them
   (dropping the best one gives an error of 0.052).
4. **Antiparallel pairs became abs().** Units 2 and 9, 6 and 8, and 10 and 13 have opposite directions and
   about equal offsets:
   - with a negative offset, `relu(a+b) + relu(-a+b)` is `relu(|a| - |b|)`;
   - with a positive offset, it is `b + max(b, |a|)`.
5. **Cone at the origin.** Five units (0, 5, 11, 14, 15) have bias about 0. Using `relu(z) = (z + |z|)/2`
   turns them into `|z|/2` plus a linear part. Their linear parts cancel the remaining affine term
   (residual slope about 0.002), so the whole function is even and the linear term is gone.
6. **Snapping.** Every direction coefficient and offset is rounded to 0.01. The constant is the minimax
   value 0.575.

Result: `PASS (snap)`, max |program - net| = 0.024 (tolerance 0.042). No `PASS (exact)` version: exact
would mean keeping every coefficient to full float precision, which is just the dump again.

## Heuristics
- Get each first-layer unit's range over the input box from the box corners (it is affine). If it is never
  positive, delete the unit. If it is never negative, fold it into a linear term.
- When an output weight is positive, fold it into the relu: `c*relu(z) = relu(c*z)`. Then units can be
  compared by their scaled direction and offset alone.
- Look for antiparallel unit pairs (w, b) and (-w, b'), with b about b':
  - `relu(z+b) + relu(-z+b)` = `b + max(b, |z|)` when b > 0;
  - it is `relu(|z| - |b|)` when b < 0.
- Units with bias about 0 are folds through the origin. With `relu(z) = (z+|z|)/2` their linear parts
  often cancel the leftover affine term, which shows the function is even.
- If every output weight is positive, the net is convex. Check for a radial or norm-like target such as
  `sqrt(1 + r^2)` or `hypot`: compare values at (0,0), (1,0), (1,1) and (2,2) with sqrt(1), sqrt(2), sqrt(3)
  and sqrt(9).
- A smooth formula can fail the check where the net's own piecewise-linear error is biggest, mostly at kinks
  such as a cone at the origin. Find that by mapping net minus formula on a grid. If the error passes the
  tolerance, keep the cleaned-up piecewise-linear form and give the formula in a comment.
- Before snapping, run greedy pruning with a least-squares refit of the output layer to see whether any unit
  is redundant. Then try rounding grids from coarse to fine, and pick the coarsest one that stays well inside
  the tolerance (here 0.01 gives 0.024 against a tolerance of 0.042, and 0.02 already gets close to it).
- Finish with the minimax constant (the midpoint of the max and min residual), not the least-squares one.
