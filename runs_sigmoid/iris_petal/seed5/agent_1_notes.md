# Notes

## What the network does
2 float inputs -> 8 sigmoid units -> 3 scores -> argmax. Mapping the argmax on a grid shows three
bands separated by two almost straight, roughly parallel lines (both slope down in x1):

- class 0 / class 1 boundary (bisected on 24 x1 values): x0 = 4.43 - 2.68*x1, max deviation from
  a straight line 0.006 -> effectively linear.
- class 1 / class 2 boundary (49 x1 values): x0 = 6.83 - 1.14*x1, max deviation 0.044 (slight
  curvature, strongest near x1 = 0.1 where the boundary is at x0 = 6.76).
- Classes 0 and 2 never touch inside the box, so the decision is an ordered chain 0 < 1 < 2.

## What I changed
Replaced the 8 sigmoids and 3 output sums with two linear tests:

    if x0 + 2.7*x1 < 4.45: return 0
    if x0 + 1.125*x1 > 6.8: return 2
    return 1

Snapping: for each boundary I scanned "nice" slopes (2.5 ... 2.8, 8/3; 1.0 ... 1.25) and
thresholds in 0.05 steps, kept the combos that pass the checker, and chose the simple values in the
middle of the passing range with the fewest unsure-zone disagreements (2.7 / 4.45 and
1.125 = 9/8 / 6.8). Slopes 2.5, 1.0 and 1.2 already fail, so the zone is narrow (unsure zone
is 3.9% of the domain) and the slopes cannot be rounded to integers.

Result: PASS (snap), 0 disagreements where the net is confident, 3641 in the unsure zone (mostly
near x1 ~ 0.1, x0 ~ 6.6-6.7, where the real 1/2 boundary bends away from the straight line).
PASS (exact) is not reachable with a straight-line rule, since the net's boundary is a sigmoid
level curve and not exactly straight.

## Heuristics
- For argmax nets with 2 inputs, first print the argmax on a coarse grid: it shows the region
  topology (which classes border each other, ordered bands vs. wedges) right away.
- Bisect each pairwise boundary (only where the pair is the top two) along one input for many
  values of the other, then fit a line by least squares. The residual tells you if one linear test
  per boundary is enough.
- When the classes are ordered bands, write the program as a decision chain (if A: 0, elif B: 2,
  else 1); the middle class needs no test of its own.
- Write each boundary as `x0 + k*x1 < t` (normalise one coefficient to 1), then snap k and t
  separately: scan a few nice k values and for each the set of passing t, then pick the centre of
  the widest passing range. Check the snap with the checker, not by eye: the allowed slope range
  can be narrow (here +-0.1).
- Sigmoid nets give smooth, slightly curved boundaries. A straight-line rule usually passes snap but
  not exact; the remaining disagreements cluster at the boundary end with the most curvature.
- The many hidden units (8 here) are no hint of complexity: several sigmoids are near-duplicates
  (similar weight directions) that together give one sharp boundary.
