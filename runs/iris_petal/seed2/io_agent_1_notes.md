# Notes

## What the function is

Two float inputs, x0 in [1.0, 6.9] and x1 in [0.1, 2.5], with three outputs and an argmax decision.
These are the ranges of Iris petal length and petal width, so I treated it as an Iris species
classifier (0 = setosa, 1 = versicolor, 2 = virginica). A nearest-sample class map matched that:
class 0 in the lower left, class 2 in the upper right, class 1 in a band between them.

## How I got there

1. A global linear fit of each output left residuals of about 5, so the outputs are piecewise linear
   (a small ReLU net).
2. For each sample I fit a local affine model on its 8 nearest neighbours and kept the fits that were
   exact (residual under 1e-3). Grouping the gradients gave about 9 linear regions.
3. For each class boundary I took the samples with a small margin (|out_a - out_b| < 1) and noted
   which region they fell in. Then I solved out_a = out_b with that region's affine coefficients.
   - Class 0/1: in the main region, x0 + 2.22*x1 = 4.38. Above x1 ≈ 1.27 it bends to
     x0 + 4.2*x1 = 6.9, but only over a small corner.
   - Class 1/2: in the upper region, x0 + 1.09*x1 = 6.76. Below x1 ≈ 0.54 it is x0 + 0.37*x1 = 6.37.
     Class 2 is the union of the two half-planes.
4. The unsnapped version (with both pieces of each boundary) passed `PASS (snap)` with 64 unsure
   disagreements. I then snapped one coefficient at a time and re-ran `check.py`:
   - Setosa: one line, `x0 + 2.25*x1 < 4.4`, was enough. The steep upper piece fell inside the
     unsure zone. `2*x1 < 4.2` and `< 4.5` both failed.
   - Virginica: the shallow lower piece became the pure length cut `x0 > 6.25`. The main line stayed
     at `x0 + 1.1*x1 > 6.75`. Changing it to `x0 + x1` failed, and so did dropping the lower piece.
5. Final result: `PASS (snap)`, with 0 disagreements where the function is confident and 1682 in the
   unsure zone (5.46% of inputs). I did not try for `PASS (exact)`, which would mean reproducing
   about 9 affine output regions. A readable decision rule was the goal.

## Heuristics

- Match the input bounds against well-known datasets first. Here [1, 6.9] x [0.1, 2.5] is Iris petal
  length/width, which also fixes the class meanings and the expected shape of the boundaries.
- Print a coarse ASCII class map (nearest sample on a grid) before fitting anything. It shows how
  many boundaries there are, roughly where they lie, and whether they bend.
- For ReLU nets, fit local affine models on k nearest neighbours and cluster the gradients. That
  recovers the linear regions, and solving out_a = out_b inside the right region gives the exact
  boundary line, with no need for logistic or SVM fits.
- A bent boundary is an intersection (AND) or a union (OR) of half-planes. Compare the two lines on
  both sides of the kink to see which one is tighter.
- Normalise each boundary to `x0 + a*x1 > c` (coefficient 1 on the dominant input) before snapping.
  This makes the coefficients easy to compare and to round.
- Snap one coefficient at a time and re-check. Drop a boundary piece when it covers only a small
  corner, and replace a near-vertical piece with a single-variable threshold (`x0 > 6.25`).
- Keep a short scripted loop of candidate rules to run through the checker. Comparing the
  "confident disagreements" counts quickly shows which simplification goes too far.
