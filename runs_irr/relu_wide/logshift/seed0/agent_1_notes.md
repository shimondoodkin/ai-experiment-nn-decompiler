# Notes

## What I did

- The net is 1 input -> 32 ReLU units -> 1 output (regression), domain x0 in [-1.5, 4].
- With one input every hidden unit is a hinge at x = -b/w. I listed the hinges and each unit's
  slope contribution w*v. 14 of the 32 units have their hinge outside the domain, so on [-1.5, 4]
  they are plain lines (or always zero). Several more (units 7, 8, 10, 15, 21, 24, 31) have
  |w*v| < 3e-4 and are just noise.
- I evaluated the net on the segments between in-domain hinges. It is continuous, increasing and
  concave. The slope falls steadily from 1.67 (near x0=-1.5) to about 0.19 (near x0=4).
- Values: -0.685 at -1.5, 0.688 at 0, 1.097 at 1, 1.792 at 4. These are log(0.5), log(2), log(3),
  log(6). The segment slopes also match 1/(x0+2) at each segment midpoint (1/0.6 = 1.67,
  1/5.6 = 0.179).
- So `agent.py` is `log(x0 + 2)`. `check.py` gives PASS (snap) with max |program - net| = 0.008,
  against a snap tolerance of 0.0495. The largest error is at the left edge, where the curve bends
  most and the net's piecewise-linear fit is coarsest.
- I did not try for PASS (exact). That needs every in-domain hinge, so about 18 kinks with
  float coefficients, which is just the dump again and explains nothing.

## Heuristics

- For 1-input ReLU nets, compute each hinge -b/w and drop units whose hinge is outside the input
  box. They are constant-sign there, so they are linear (or zero) and fold into the affine part.
- Drop units with |w_in * w_out| times the domain width far below the tolerance.
- Tabulate the slope on each segment between hinges. Monotone, decreasing slopes mean a concave
  curve. Before fitting splines, test elementary candidates: log(x+c), sqrt(x+c), 1/(x+c), exp.
- Find the shift from the derivative: if slope*(x+c) stays roughly constant across segments, the
  function is k*log(x+c). Then check that the values match k*log(x+c) at a few round points
  (x = 0, 1, the domain ends).
- A domain lower bound like -1.5 with an argument shift of +2 keeps log's argument positive
  (0.5..6). Domain bounds often hint at the shift of a log or sqrt.
- The biggest error between net and formula is where curvature is highest (here near the left
  edge). Check there first when judging whether a closed form fits.
