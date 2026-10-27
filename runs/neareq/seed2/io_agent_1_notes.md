# Notes

## What the function does

An "approximately equal" detector: it outputs 1 when `|x0 - x1| < 0.2`, else 0.

## How I got there

- Class 1 is only 11% of samples. A coarse grid of the class over (x0, x1) showed a thin band along
  the diagonal x0 = x1, so the useful feature is the difference `x0 - x1`.
- The raw score against `d = x0 - x1` is a tent: a flat top of about 9 near d = 0, then it falls
  roughly linearly (about -78 per unit) on both sides, and it is symmetric in d. The sum x0 + x1 only
  shifts the score slightly (the plateau runs from about 8.8 to 9.8).
- The class flips at |d| between 0.199 and 0.209 on both sides, so I snapped the threshold to 0.2,
  one shared value for both sides, as `abs(x0 - x1) < 0.2`.
- `check.py`: PASS (snap). All 589 disagreements are in the function's unsure zone, right at the
  |d| = 0.2 edge, where the net's boundary moves by about ±0.01 with x0 + x1.
- I did not try to reproduce the raw score exactly. The small dependence on x0 + x1 looks like
  training noise, not part of the intended function.

## Heuristics

- Start by mapping the class on a coarse grid. A thin band shape points straight at the right
  feature: a diagonal band means x0 - x1 (equality or closeness), an anti-diagonal band means x0 + x1.
- After you pick a candidate feature, sort the samples by it and print the score next to it. A tent
  shape or an even, symmetric shape means `abs(feature)` compared against a tolerance.
- Among the samples, find the largest |feature| in class 1 and the smallest in class 0. The
  threshold sits between them. Snap it to a round value such as 0.2, and use one shared threshold
  for both sides of a symmetric band.
- A blurred boundary (overlap of about 5% of the threshold) that drifts with another variable is
  usually imprecision in the trained net. It belongs in the snap tolerance, not in the program.
- Correlation with products such as x0*x1 can mislead. Here it was high (0.84) only because
  "x0 close to x1" correlates with a positive x0*x1. Check the geometry before trusting a correlation.
