# Notes

## What I did

- The dump is a 2 -> 8 sigmoid -> 3 argmax net. Instead of simplifying the units one by one,
  I printed the class map on a 60x25 grid over the input box. It showed three bands separated by
  two nearly straight lines: class 0 in the low-x0/low-x1 corner, class 2 in the high corner,
  class 1 in between.
- I bisected each boundary in x0 at 49 values of x1 and fitted `x0 = a*x1 + b` by least squares:
  - 0|1: `x0 = -2.619*x1 + 4.368` (max deviation 0.011, so practically a straight line);
  - 1|2: `x0 = -1.086*x1 + 6.666` (max deviation 0.036, slightly curved).
- The fitted lines pass as-is (PASS snap). Then I snapped the coefficients and tried several
  candidates against the checker:
  - `x0 + 2.5*x1 < 4.3` and `x0 + 1.1*x1 < 6.7` pass, with 0 disagreements where the net is
    confident;
  - `x0 + x1 < 6.5 / 6.55 / 6.6` fails at the high-x1 end (the 1|2 slope really is about 1.1);
  - `x0 + 3*x1 < 4.5` fails (the 0|1 slope is about 2.6, not 3).
- Final program: two linear thresholds checked in order (0, then 1, else 2). The lines do not
  cross inside the box, so an if-chain is equivalent to the 3-way argmax. Result: PASS (snap).
  All disagreements (1.8% of the test inputs) are in the net's unsure zone. It cannot be exact,
  because the sigmoid boundaries are slightly curved.

## Heuristics

- For low-dimensional argmax nets (2 inputs), first print the decision map on a coarse grid
  instead of reading weights. Smooth sigmoid nets with many units often just draw a few
  straight cuts.
- To find a boundary, bisect along one input at many values of the other, then fit a line.
  The maximum residual tells you whether one linear threshold is enough.
- If the class regions are ordered bands (class k sits between k-1 and k+1) and the boundaries
  do not cross inside the input box, write the argmax as an if-chain of threshold tests.
- Snap the slope first (to 1, 1.1, 2.5, ...), then refit the intercept for that slope. If a
  rounder slope fails, the failures show up at the far end of the boundary, away from where
  the line pivots.
- For sigmoid nets, an exact pass is usually out of reach with piecewise-linear code. Aim for
  zero disagreements where the net is confident, and accept differences in the unsure zone.
- Coefficient sizes in the hidden layer hint at the boundary slopes. Here, units 2 and 3 have
  x1/x0 weight ratios of about 3-4.6, and units 5-7 about 1-1.4. Treat this only as a hint and
  confirm it with bisection.
