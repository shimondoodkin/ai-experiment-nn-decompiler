# Agent notes

## What the network does

A 2 -> 8 (sigmoid) -> 3 argmax net. Mapping the argmax over a grid showed three bands separated by
two nearly straight, non-crossing boundaries running from top-left to bottom-right. Class 0 is the
low corner (small x0 and x1), class 2 is the high corner, and class 1 is the band in between. Classes 0
and 2 never touch.

## What I changed

- I replaced the 8 sigmoid units and 3 output scores with two linear threshold tests.
- I found each boundary by bisecting along x0 for 49 values of x1 and fitting a line to the points:
  - 0|1: x0 = -1.583*x1 + 3.594. Snapped to x0 + 1.6*x1 = 3.6, which is `5*x0 + 8*x1 < 18`.
  - 1|2: x0 = -1.127*x1 + 6.770. Snapped to x0 + 1.125*x1 = 6.75, which is `8*x0 + 9*x1 > 54`.
    This boundary bends slightly (the local slope is 1.18 near x1=0 and 1.09 near x1=2.5), so a
    straight line cannot match it exactly.
- I tried coarser snaps:
  - slope 1.5 with threshold 3.5 for 0|1: FAIL;
  - slope 1 with thresholds 6.5, 6.6 or 6.7 for 1|2: FAIL;
  - slope 1.2 with threshold 6.8 for 1|2: FAIL;
  - slopes 1.1 or 1.125 with thresholds 6.75 or 6.77, together with 1.6 / 3.6: all PASS (snap).

  I kept 9/8 and 27/4, because they give integer coefficients.
- Result: `PASS (snap)`, with 0 disagreements where the net is confident. All 4180 disagreements are
  in the unsure zone (3.55% of inputs). PASS (exact) would need the sigmoid curvature, which goes
  against the goal of a simple program.

## Heuristics

- For low-dimensional argmax nets, print the class map on a coarse grid first. The topology
  (bands, wedges, islands, which classes touch) suggests the program's shape before any algebra.
- When classes form ordered bands, write a chain of threshold tests in that order
  (`if s0 < t0: 0 elif s1 > t1: 2 else 1`). Only adjacent classes need a boundary.
- Recover each boundary directly: bisect along one input at many values of the other, then fit a
  line. This is more reliable than reading slopes off the hidden-unit weights. Here several hidden
  units were almost parallel and combined into each boundary.
- Snap the slope first and the intercept second. Then clear denominators to get integer
  coefficients (x0 + 1.6*x1 < 3.6 becomes 5*x0 + 8*x1 < 18).
- Sigmoid nets give slightly curved boundaries. A straight line usually stays inside the unsure
  zone, but check the ends of the input range, where the curvature errors add up (failures showed
  up at the corners x1 ~ 0.1 and x1 ~ 2.5).
- Try a few candidate snaps against the checker in one loop. Keep the simplest one that passes,
  and use the number of unsure-zone disagreements to break ties.
