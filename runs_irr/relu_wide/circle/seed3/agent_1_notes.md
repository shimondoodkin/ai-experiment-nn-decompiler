# Notes

## What the network does

2 inputs in [-3, 3], 64 ReLU units, binary output (class 1 when the score > 0).

- About 18 units have tiny weights and big positive biases (e.g. `relu(-0.02*x0 - 0.25*x1 + 4.37)`):
  inside the box they are always active and almost constant, so together they form the positive
  constant score at the centre (score at the origin is about 134).
- The other ~46 units have negative output weights and bias about -1.5 to -2.8, with input weight
  directions spread around the full circle. Each one switches on once the point moves far enough
  in its direction, and pulls the score down. Many half-planes with directions all around the
  circle and similar offsets = a polygon approximating a disc.
- A radial scan from the origin at 24 angles found exactly one sign change per ray, at radius
  1.984 to 2.037. So the decision region is a disc of radius 2 around the origin.

## Program

`return 4 - (x0*x0 + x1*x1)`: class 1 when x0² + x1² < 4.

Check: PASS (snap), empirical over 440k inputs, 0 disagreements where the net is confident;
all 6794 disagreements are in the net's unsure zone (the polygon vs. the true circle).
PASS (exact) is not reachable with a clean formula: the net's boundary is a 46-sided-ish polygon
with radius wobbling by about ±1.5%, and copying it would mean keeping the network.

## Heuristics

- Split units into "always on inside the box" (small weights, large positive bias; check
  `|w|*range < bias`) and "edge" units. The always-on ones only add a constant (plus a small linear
  tilt); their sum is the score's base level.
- Many units with the same sign of output weight, similar bias/|w| ratio and directions spread
  evenly over 360 degrees mean a radial function: a disc / circle (or a cone `r`). Compute the
  offset |bias|/|w| per unit to estimate the radius.
- For a binary 2-input net, do a radial scan of sign changes from the centre (or a grid scan) to
  read the boundary shape directly before analysing the weights. A near-constant crossing radius
  means `r² < R²`; snap R to the nearest integer / simple fraction and centre to the origin.
- Write the score as `R² - (x0² + x1²)` rather than `sqrt`: it has the same sign, is simpler, and
  avoids sqrt.
- A ReLU net can only make polygonal boundaries, so a curved true boundary will only ever give
  PASS (snap); don't chase exact by adding corners.
