# Notes

## What the network does

4 inputs -> 8 tanh units -> 3 class scores, argmax. Looking at the decision regions (not at the
individual units), I found:

- Class 0 and class 2 never touch. Bisecting random segments onto class changes gave only 0|1 and
  1|2 boundary points, no 0|2 points. So the net is an ordered 3-way split: 0 < 1 < 2.
- Both boundaries are nearly flat planes. Solved for x2 (the input with the largest weight),
  they are:
  - 0|1: x2 ~ 0.17 x0 + 0.63 x1 - 0.72 x3 + 0.29;
  - 1|2: x2 ~ 0.35 x0 + 0.69 x1 - 1.70 x3 + 3.63.

  Both bend slightly near the edges of the box (in the x3 and x1 directions), where the tanh units
  saturate. A single plane per boundary gave 460 confident errors. A max-margin LP showed that no
  single plane separates the confident points.
- Each class region on the "outside" is convex. The region below the 0|1 boundary and the region
  above the 1|2 boundary are each the intersection of two half-spaces. Two planes per boundary
  separate the confident points with no errors.

## The program

```
low_a  = x2 + 0.7*x3  - 0.18*x0 - 2/3*x1     class 0 if low_a < 0.1 and low_b < 0.46
low_b  = x2 + 0.75*x3 - x0/6    - 0.6*x1
high_a = x2 + 1.8*x3  - 0.36*x0 - 0.62*x1    class 2 if high_a > 3.87 and high_b > 2.96
high_b = x2 + 1.5*x3  - x0/3    - 0.8*x1     otherwise class 1
```

All 8 tanh units and the 3 score lines are gone. What is left is 4 linear forms with snapped
coefficients and 4 thresholds. Check result: `PASS (snap)`, with 0 disagreements where the net is
confident; every disagreement is in the unsure zone. On 1M fresh uniform points plus 300k more,
there are also 0 confident disagreements. The original threshold 3.88 had one, and lowering it to
3.87 fixed that.

## How I got there

1. I sampled the box and labelled each point with the net's class and margin, using "confident" =
   margin >= the checker's unsure threshold (0.5623).
2. I bisected random segments onto class changes. Fitting a plane (SVD) to each pair's boundary
   points showed which pairs share a boundary and how flat each one is (residuals).
3. I ran a max-margin LP (scipy linprog) per boundary, with the x2 coefficient fixed at 1. It was
   infeasible for one plane. I then fitted two planes by alternating: the class that must satisfy
   both planes is in both LPs, and each point of the other class is assigned to the plane that
   cuts it off. This converged to 0 errors in 2 iterations.
4. Snapping: I did a grid search over "nice" slope values (tenths, 0.05 steps, thirds, sixths)
   for both planes together. I scored each pair of planes by its slack, meaning how far the two
   thresholds can move together with every confident point still correct. I then chose
   thresholds inside the feasible window. The windows are narrow (about 0.02 in x2 units), so the
   thresholds need two decimals.
5. I validated on a large fresh sample, not just the checker's set, and moved one threshold to
   the safe side.

## Heuristics

- For argmax nets, first find which class pairs actually share a boundary: bisect random segments
  onto class changes and tabulate the pairs. If some pair never touches (here 0|2), the classes
  are ordered, and the program becomes a decision list: test the extreme classes, and the middle
  class is the default.
- Fit a plane to each pair's boundary points (smallest singular vector) and write it solved for
  the input with the largest weight. Residuals binned by the other inputs show where the boundary
  bends. For tanh/sigmoid nets, the bends are at the box edges, where units saturate.
- If one plane is not enough, check which side is convex. If the boundary bends away from a
  class, that class's region is an intersection of half-spaces: `a < t1 and b < t2`. Two planes,
  fitted by alternating LPs, are usually enough for a small smooth net.
- Use the checker's unsure-zone threshold to define "confident" points, and require 0 errors on
  those only. That leaves a real gap, so an LP with a positive margin is feasible.
- Snap all the planes of one boundary together, and score each candidate by its joint slack (how
  far all its thresholds can move while it stays correct), not by its error count alone. Then
  pick round thresholds inside the window.
- Validate on a fresh, larger random sample (a different seed) as well as the checker's input
  set. Thresholds at the edge of the window fitted on one sample can fail on corner points of
  another sample. Move them toward the middle of the window.
- The tanh units themselves did not need to be understood one by one. Working on the decision
  geometry was far shorter than pruning or merging units.
