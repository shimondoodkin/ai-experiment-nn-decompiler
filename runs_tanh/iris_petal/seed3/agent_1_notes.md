# Notes

## What the network does
2 inputs (x0 in [1, 6.9], x1 in [0.1, 2.5]), 8 tanh hidden units, argmax over 3 classes.
Plotting the decision map on a grid showed three bands: class 0 in the low corner, class 2 in the
high corner, class 1 the wide band between. Class 0 and class 2 never touch inside the box.

## What I changed
- Replaced all 8 tanh units and the 3 output scores with two linear tests.
- Located both boundaries by bisection along x0 for 25 values of x1, then least-squares fitted
  x0 as a line in x1:
  - 0/1 boundary: x0 = 5.136 - 3.700*x1 (max residual 0.001, essentially straight);
  - 1/2 boundary: x0 = 6.794 - 1.085*x1 (max residual 0.024, slightly curved).
- Snapped to `x0 + 3.7*x1 < 5.1 -> 0` and `x0 + 1.1*x1 > 6.8 -> 2`, else 1.
  Tested alternatives with check.py: the unsnapped fit (5.14 / 1.08 / 6.79) has fewer unsure-zone
  disagreements (3658 vs 5134) but both have 0 confident disagreements. Further snapping fails:
  `x0 + x1 > 6.75` (slope 1 instead of 1.1) gives 1326 confident disagreements, and the 0/1
  boundary does not tolerate slopes 3.5 or 4.
- Result: PASS (snap), 0 confident disagreements. Exact is not reachable with straight lines because
  the 1/2 boundary bends slightly.

## Heuristics
- For 2-input argmax nets, first print a coarse ASCII class map over the input box; it shows how
  many regions there are, which pairs of classes share a boundary, and whether boundaries look straight.
- Find each pairwise boundary by bisection along one input for a sweep of the other, then fit a
  line; the max residual tells you whether a single linear test can work.
- Express each boundary as `x0 + a*x1 <op> t` (normalize one coefficient to 1), then snap `a` and `t`
  one at a time to round values, re-checking after each; keep the slope more precise than the
  threshold, since slope errors grow across the box while a threshold shift is uniform.
- With nested regions (class 0 | class 1 | class 2 in bands) an if-chain of half-plane tests
  (return the outer classes first, the middle class last) replaces the argmax completely.
- tanh hidden layers with many near-duplicate units (here pairs 1/2, 4/5, 6/7 with near-equal weights)
  usually implement just a few soft cuts; don't simplify unit by unit, read the decision geometry.
