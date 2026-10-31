# Notes

## What the network does
Class 1 when the point (x0, x1) lies inside the disk of radius 2 around the origin:
`x0^2 + x1^2 < 4`. `agent.py` returns `4 - x0*x0 - x1*x1` (positive inside).
Check result: PASS (snap), 0 disagreements where the net is confident; the 5811 disagreements are all
in the net's unsure zone, on the boundary circle.

## How I got there
- 2 -> 16 -> 1 ReLU net, binary output, threshold 0.
- Unit 7 is dead on the whole box (bias -0.38, weights ~0.01): removed.
- Units 4, 6, 8, 10, 13, 15 have large positive biases (+3.7 to +4.5) and small weights, so they are
  always active on [-3,3]^2: together they are a single affine term
  `2.80*x0 + 2.51*x1 + 65.7` (constant includes the output bias).
- The remaining 9 units all have negative output weights and, after normalising each weight vector,
  point in 9 directions spread around the circle (-142, -91, -52, -23, 27, 48, 91, 140, 167 degrees)
  with normalised offsets 0.33-0.67. Output = big constant minus a sum of ReLU ramps in all
  directions, i.e. a convex "bowl" whose positive part is a polygon around the origin: a
  piecewise-linear approximation of a paraboloid / circle.
- On a 601x601 grid, positive points have radius <= 2.03 and negative points radius >= 1.96, and the
  positive area fraction (0.346) matches pi*4/36 = 0.349. So the radius is 2 (r^2 = 4).
- An exact program would have to copy the 9-sided polygon; the circle is the intended function and
  passes the snap check.

## Heuristics
- First classify units by their range on the input box: always-off (drop), always-on (fold into one
  affine term), and switching units. Large bias with small weights means always-on.
- For switching units, normalise each weight vector to unit length and list the angle and the
  normalised offset. Many units with the same output sign, directions spread over 360 degrees and
  similar offsets = a convex piecewise-linear bowl: the decision region is a polygon approximating
  a disk (or ellipse if offsets vary systematically with angle).
- Estimate the radius from the grid: max radius of positive points vs min radius of negative points,
  and the area fraction vs pi*r^2/box_area; snap r^2 to an integer.
- For binary outputs return a smooth score with the right sign (`r^2 - x0^2 - x1^2`); the checker only
  needs the decision to agree outside the unsure zone.
