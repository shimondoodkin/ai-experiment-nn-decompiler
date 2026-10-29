# Notes

## What I did

- The dump is a 2 -> 8 (tanh) -> 3 argmax net. Instead of simplifying unit by unit, I mapped the
  decision regions on a grid (scratch/map.py). There are three bands ordered along a "size"
  direction: class 0 at small x0/x1, class 1 in the middle, class 2 at large values. Class 0 and
  class 2 never touch.
- I scanned rows of fixed x1 for class changes and fitted each boundary as x0 = a*x1 + b
  (scratch/bnd.py). Both are almost exactly straight (max residual about 0.001; the quadratic term is
  about 0.002):
  - 0|1: x0 + 1.48*x1 = 3.51
  - 1|2: x0 + 1.06*x1 = 6.72
- The program is two threshold tests on linear scores, applied in order (a decision list).
- Snapping: I tried several rounded versions and measured the largest net margin (top score minus
  second score) where program and net disagree (scratch/margin.py). The checker's unsure zone is 0.515.
  - 1.5 / 3.5 and 1.05 / 6.7: max 0.31. This is the version I chose, comfortably inside the zone.
  - x0 + x1 < 6.65 for the second boundary passes, but only barely (0.46); 6.6 and 6.68 fail.
    Too fragile.
  - x0 + x1 < 6.7 fails: the slope is really about 1.06, and over the x1 range of 2.4 a slope
    error of 0.06 shifts the line by about 0.15 at the edge.
- Result: `PASS (snap)`, with 0 disagreements where the net is confident. `PASS (exact)` is not
  reachable without keeping the tanh units, because the true boundaries are slightly curved.

## Heuristics

- For low-dimensional argmax nets (2 inputs), map the class regions on a grid before touching the
  weights. Eight tanh units that combine into two straight boundaries are far easier to see from the
  output than from the units.
- Find boundary points by scanning one input at fixed values of the other for label changes, then
  fit x_a = slope*x_b + c. If the residual is tiny, a linear threshold is enough. Check a quadratic
  term to confirm there is no curvature.
- Ordered bands (0 | 1 | 2 along one direction, with 0 and 2 never touching) become a decision list:
  `if s1 < t1: 0; elif s2 < t2: 1; else 2`.
- Choose snaps by the largest net margin among disagreements, not just by pass/fail. A snap that
  passes close to the unsure-zone limit is fragile: new counterexamples or another seed would break it.
- A slope error matters more than an intercept error when the other input has a wide range. Snap the
  slope only if slope error × input range is small compared with the boundary's tolerance band.
- Express each boundary as x0 + k*x1 < c (coefficient of one input normalised to 1), so the snapped
  values stay readable.
