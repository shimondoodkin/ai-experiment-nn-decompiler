# Notes

## What the network does

The 2-input, 64-unit ReLU net is a binary classifier whose class-1 region is a disk of radius 2
around the origin: `x0^2 + x1^2 < 4`.

How I found it:
- The units fall into two groups. About 20 units have small weights (|w| < 0.3) and large positive
  biases (+2.3 to +4.1). They are always active on the [-3, 3]^2 box, so together they are only a
  near-constant (plus a small linear tilt) that makes the score positive at the centre.
- The other ~40 units have weights of size about 2 to 3 in every direction and negative biases of
  about -2. Each is a ReLU ridge `relu(w·x - b)` that switches on once the point is about distance
  |b|/|w| ≈ 0.7 out along direction w, with large negative output weights. Their normals are spread
  all round the circle, so the sum is a polygonal approximation of a cone that grows with the radius.
- Plotting the sign on a grid showed a round blob. Scanning the boundary along 16 rays gave a
  radius between 1.988 and 2.011 in every direction, so the boundary is a circle of radius 2.
- Units 60 and 62 have output weights of about 0.002 and 0.01; they do nothing.

## Program

`return 4 - (x0*x0 + x1*x1)`, with class 1 when the value is > 0.

`check.py`: PASS (snap), 0 disagreements where the net is confident. The 4930 disagreements are all
in the net's unsure zone, on the polygon-vs-circle boundary (the net's radius wobbles by ±0.01).
PASS (exact) would mean copying the net's 40-sided polygon, which is the noise the snap is meant
to take out, so I did not try for it.

## Heuristics

- Split the hidden units by bias sign and weight size. Units with small weights and large positive
  biases are always on inside the box: fold them into one constant (plus a linear term if needed).
  Units with large weights and biases of similar size pointing in many directions are ridges, and a
  ring of them is a polygon approximation of a radial function (norm / circle).
- For a 2-input binary net, draw the sign on a coarse grid before reading weights. The shape
  (disk, half-plane, band, XOR quadrants) usually gives the formula at once.
- To recognise a circle, scan the zero crossing along many rays from the centre. A near-constant
  crossing distance r means `r^2 - x0^2 - x1^2`. Its spread tells you how much tolerance the snap
  needs, and the centre can be found the same way if it is not the origin.
- Snap the radius to a round value (here 2.0, from crossings at 1.99 to 2.01). With x² written
  as x*x (no `**` in the language), the squared form avoids needing sqrt.
- Units whose output weight is about 1e-3 of the others can be dropped.
