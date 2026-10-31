# Notes

## Result

`agent.py` returns `4 - (x0*x0 + x1*x1)`: class 1 inside the disk of radius 2 centred at the origin.
`check.py`: PASS (snap), 0 disagreements where the net is confident; 5836 disagreements (2.8% of inputs),
all in the net's unsure zone along the circle.

## What I did

1. Printed an ASCII map of the decision (`raw > 0`) on a 61x61 grid over [-3,3]^2: a round blob centred
   at the origin.
2. Bisected the boundary along 48 rays from the origin: radius 1.97 to 2.03 at every angle, mean about 2.0.
   The net's boundary is a slightly wobbly polygon approximating the circle r = 2.
3. Read the units as half-planes: 12 steep units (|w| about 3 to 5) whose lines lie at distance 1.67 to 1.83
   from the origin with normals spread around the circle, each with a large negative output weight
   (-3 to -12) — "outside this tangent line -> class 0". Two units with positive output weight (h6, h15)
   have lines at distance about -1.8 (origin on their active side), so they are roughly constant +8 near
   the disk and cancel the bias picture. Three units (h3, h7, h14) have tiny slopes and tiny output weights;
   they are constants. Together: an intersection of tangent half-planes = a polygon around a circle.
4. Snapped the polygon to the circle x0^2 + x1^2 < 4 (radius 2, a small integer). Returning `4 - r2` keeps
   the sign convention (threshold 0) and gives a smooth score.

An exact pass would need the net's own wobbly 12-gon; the circle is the intended function.

## Heuristics

- For 2-input binary nets, first draw the decision region on a grid (ASCII is enough). Its shape (disk,
  band, quadrant, XOR pattern) usually names the formula before any weight is read.
- For a blob around a centre, bisect the boundary along rays from the centre. Near-constant radius means a
  circle `x0*x0 + x1*x1 < r*r`; a radius that varies with angle as 1/|cos| means a polygon or box.
- Read each sigmoid unit as a line: normal = (a, c)/|(a, c)|, signed distance = -d/|(a, c)|, steepness =
  |(a, c)|. Many steep units at the same distance with normals spread around the circle, all with large
  negative output weights, are tangent lines of a circle (a polygon approximation of a round region).
- Units with tiny input weights (|(a, c)| < 0.3) and tiny output weights are constants; units whose line
  lies outside the box (or with the whole box on one side) are constants on the domain.
- Snap the radius (or r^2) to a nice number using the median of the bisected radii, not the unit lines'
  distances: the polygon's tangent lines sit inside the circle (here about 1.8 for radius 2).
- Return a signed score (`r*r - x0*x0 - x1*x1`) instead of 1/0, so the program also gives the margin.
