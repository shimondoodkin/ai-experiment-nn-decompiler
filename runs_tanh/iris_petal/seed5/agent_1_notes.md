# Notes

## What the network does
2 float inputs (x0 in [1, 6.9], x1 in [0.1, 2.5]), 8 tanh hidden units, argmax over 3 classes.
Printing the predicted class on a 60x49 grid showed three bands separated by two almost straight
diagonal lines; the 8 hidden units are just a smooth way of drawing those two lines.

## What I changed
- Located each class boundary by scanning x0 at fixed x1 and bisecting the class change,
  then least-squares fitted `x0 = a*x1 + b` to the boundary points.
  - 0|1 boundary: x0 = -2.80*x1 + 4.46 (max residual 0.005, essentially straight).
  - 1|2 boundary: x0 = -1.04*x1 + 6.68 (max residual 0.044, slightly curved).
- Replaced the whole net with two linear tests on the inputs:
  `x0 + 2.8*x1 < 4.5 -> 0`, else `x0 + x1 < 6.6 -> 1`, else `2`.
- Snapping: tried several rounded variants with check.py. Slope 2.8 could not be snapped to 3
  (1003 confident disagreements); slope 1.04 snapped to 1 with threshold 6.6 (6.65 failed,
  104 confident disagreements). Threshold 4.46 snapped to 4.5.
- Result: `PASS (snap)`, 0 disagreements where the net is confident; all differences are in the
  net's unsure zone along the boundaries. An exact pass is not realistic for a tanh net without
  keeping the tanh units themselves.

## Heuristics
- For low-dimensional argmax nets, first print the class map on a grid; the number and shape of
  boundaries tell you the program structure (here: two lines -> an ordered if-chain).
- When classes appear as ordered bands, write the decision as a cascade of threshold tests
  (`if s1 < t1: 0 elif s2 < t2: 1 else 2`) instead of computing all scores and an argmax.
- Find boundaries numerically (scan + bisection along one input at fixed values of the other),
  then fit a line; the residual tells you whether a linear rule is enough.
- Normalize each boundary to coefficient 1 on one input (`x0 + a*x1 < b`) before snapping, so
  only one slope and one threshold need rounding.
- Snap greedily and verify each step: round slope to the nearest simple value, then re-tune the
  threshold; if a snap fails, step back to one decimal place. A slightly curved boundary can still
  snap to a straight line, with the threshold placed to balance the error at both ends.
- Smooth (tanh/sigmoid) hidden layers in small classifiers often just implement near-linear
  boundaries; do not try to simplify unit by unit when the end-to-end decision map is simple.
