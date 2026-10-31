# Notes

## What the network does

`dump.py` is a 2 -> 16 (tanh) -> 1 binary classifier on x0, x1 in [-3, 3].

- Five hidden units (0, 1, 7, 8, 11) have input weights of about 0.009 and biases of about +-4, so
  over the whole box they sit at tanh(+-4) = +-0.9993. They are constants. Folding them with the
  output bias gives a single constant of -42.896.
- The other eleven units all have weight-vector norm about 2.5 and |bias| about 4.4, so each one
  switches on a line at distance |b|/|w| = 1.69..1.75 from the origin. Their normal directions are
  spread around the full circle (angles -155, -100, -65, -36, -21, 14, 47, 59, 91, 125, 175 deg),
  and the output-weight sign matches the bias sign so every unit pushes the score down when the
  point moves outward. Together they form a smooth, nearly round "inside" region.
- Bisecting the decision boundary along 72 rays from the origin gives radius 1.968..2.009
  (mean 1.991, median 1.990). The score is about +18.7 at the origin and about -26 at the corners.

## The program

    return 4 - (x0 * x0 + x1 * x1)

Class 1 inside the disc of radius 2 around the origin (x0^2 + x1^2 < 4). `PASS (snap)`: there are
no disagreements where the net is confident; the 5964 disagreements are all in the unsure zone, a thin
band around the circle where the eleven tanh edges make the boundary wobble by about 0.02.

Checked alternatives:
- the net with its constant units folded (11 tanh units plus one constant) gives `PASS (exact)`, but it is just
  the dump with fewer terms, not a readable formula;
- r^2 = 3.9 and 3.95 also pass, but r^2 = 4.05 fails with 50 confident disagreements. r = 2 is the round value
  inside the measured range, so I kept it.

## Heuristics

- **Constant units:** a unit whose |w|·(input range) is tiny compared with |b| (here |w| ~ 0.009, |b| ~ 4) is
  saturated over the whole domain. Replace it with v·tanh(b) (or v·sigmoid(b)) and fold it into the output bias.
- **Line distance:** for a tanh/sigmoid unit, the switching line is at distance -b/|w| from the origin along the
  direction w/|w|. When many units share the same distance and |w|, and their normals are spread around 360
  degrees, the net is approximating a circle (or a convex polygon) of that size.
- **Measure the boundary directly:** bisect the net's decision along many rays from a point inside the class region.
  A near-constant radius means a disc. Snap the radius to a round number inside the measured min..max range, and
  check both sides (r^2 slightly larger or smaller) to see which side the net's confident points allow.
- **Edge distance is not the boundary radius:** the decision boundary lies outside the units' line distance (2.0
  against 1.73 here), because the output threshold is crossed only after several units have partly switched.
  Use the bisected boundary, not -b/|w|, for the final constant.
- For a binary output, any score with the same sign works. A polynomial like r^2 - (x0^2 + x1^2) is the simplest
  readable form, even when the net builds it out of many linear edges.
