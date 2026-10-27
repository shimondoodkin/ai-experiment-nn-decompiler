# Notes

## What the function is
Two inputs in the ranges of Iris petal length (x0, 1.0-6.9) and petal width (x1, 0.1-2.5), three
argmax outputs. The class map (printed as an ASCII grid from the samples) shows three bands
ordered by petal size: 0 = setosa (small), 1 = versicolor, 2 = virginica (large). Classes 0 and 2
never touch, so only two boundaries matter: 0|1 and 1|2.

## What I did
1. Printed a nearest-sample class grid over (x0, x1): two diagonal boundaries, slightly bent.
2. For each adjacent pair (a, b) took out_a - out_b on samples near the boundary and fitted planes
   with a small RANSAC (3-point fits, inlier tolerance 1e-3). The raw outputs are exactly piecewise
   linear (ReLU net), so the pieces come out with residual ~0.
   - 0|1: x0 + 3.122*x1 = 4.891 (right part) and x0 + 3.245*x1 = 4.974 (left part).
   - 1|2: x0 + 1.091*x1 = 6.781 (most of the range) and x0 + 0.702*x1 = 6.578 (x1 < ~0.5).
   A third far-away piece of out1-out0 (deep inside class 1) does not affect the decision.
3. Decided how the two pieces combine by checking which one is active at each end of the boundary:
   in both cases the class is the "larger" side when the max of the two lines is positive.
4. Snapped version (`agent_snap.py`): `length + 3.2*width < 4.95` -> 0;
   `length + 1.1*width > 6.8 or length + 0.7*width > 6.6` -> 2; else 1. PASS (snap), 0 confident
   disagreements. Cruder snaps (x0 + 3*x1 < 4.8, x0 + x1 > 6.7) FAIL: the unsure zone is only
   ~0.08 in x0 wide (score margin 0.55 / slope ~6.7), so coefficients must be good to ~0.05.
5. Final `agent.py` keeps the fitted 4-decimal coefficients with max(line, line) per boundary:
   PASS (exact). Two-decimal rounding of the same lines already drops to PASS (snap).

## Heuristics
- Recognise the domain from input ranges (1.0-6.9 / 0.1-2.5 with 3 classes = Iris petals) and name
  the variables and classes accordingly.
- Start with an ASCII class map of nearest samples; it shows topology (which classes touch, how
  many boundaries, whether they are straight or bent) before any fitting.
- For argmax outputs, model the pairwise differences out_a - out_b, not the outputs themselves;
  only differences between neighbouring classes matter.
- ReLU nets are piecewise linear: RANSAC plane fits with a tiny tolerance recover each linear piece
  exactly; fitting in bands (e.g. by x1) quickly reveals where a boundary bends.
- A bent boundary between two pieces is max(line1, line2) or min(line1, line2); decide by testing
  which piece is active at each end (equivalently: OR vs AND of the two half-plane conditions).
- Normalise each line by the x0 coefficient to read it as "x0 + k*x1 = t"; this makes snapping and
  comparing pieces easy.
- Estimate the snap tolerance before snapping: unsure-zone margin / gradient of the score gives the
  allowed boundary shift in input units. Snap only as coarsely as that permits.
- Small kinks whose two pieces are within tolerance can be merged into one line (0|1 here);
  kinks larger than the tolerance need both pieces (1|2 here).
- Exact pass needs the fitted coefficients to ~4 decimals; keep a snapped variant alongside.
