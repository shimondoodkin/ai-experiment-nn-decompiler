# Notes

## What the network does
2 inputs in [-3, 3], binary output, 16 sigmoid units. The units come in two families:
- 11 "fence" units with output weight about -10.5 and bias about -7.5 (weight-vector norm about 3.7-4.5).
  Each fires once the input crosses a line at distance roughly 7.5/4.2 ~ 1.8 from the origin, and
  pushes the score down. Their normals point in directions spread all around the circle.
- 5 "interior" units with output weight +2.8 to +5.3 and bias +5.6 to +7.6, which are on near the
  origin and switch off outside a similar distance.
Together they cut out a convex, nearly round region around the origin: a polygon approximating a disc.

Plotting the sign on a grid shows a disc. Scanning 360 rays from the origin for the sign change
gives boundary radii 1.978-2.019; a least-squares circle fit gives centre (0.009, 0.012), radius 1.998.

## What I changed
Replaced the 16-unit net with the snapped formula `4 - (x0*x0 + x1*x1)`: class 1 inside the circle of
radius 2 centred at the origin. Centre snapped to (0, 0), radius to 2 (r^2 = 4).

`check.py agent.py`: PASS (snap), 0 disagreements where the net is confident, 5830 inside the
unsure zone (2.78% of inputs, all hugging the boundary, where the polygonal net deviates from the
true circle by up to about 0.02 in radius). PASS (exact) is not reachable with a simple formula:
the net's boundary is a soft 16-sided polygon, not a circle.

## Heuristics
- For low-dimensional binary nets, first draw the sign of the output on a grid; the shape (disc,
  band, half-plane, quadrant) usually suggests the formula before any weight analysis.
- Group hidden units by sign of the output weight and sign/size of the bias. Many units with equal
  |bias|/|w| ratio and normals spread over all directions are the facets of a polygon approximating
  a circle of radius |bias|/|w| (here about 1.8 for the facet lines; the actual boundary sits a bit
  further out where the summed penalties cross the output bias).
- Measure the boundary directly: scan rays from a point inside for the sign change, then fit a
  circle (linear least squares on 2cx + 2cy + k = x^2 + y^2). Snap centre and radius to nice values.
- Return a signed score (r^2 - x^2 - y^2) rather than a 0/1 comparison, so the program has a natural
  margin and disagreements fall only where the net itself is near zero.
- Keep scratch scans coarse: evaluating a 16-unit sigmoid net on 360 x 200k points is slow; a few
  thousand radial samples per ray already gives the radius to 1e-3.
