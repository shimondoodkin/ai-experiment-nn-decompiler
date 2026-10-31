# Notes

Result: `return 4 - (x0*x0 + x1*x1)`, i.e. class 1 inside the circle of radius 2 around the origin.
`check.py`: PASS (snap), 0 disagreements where the net is confident; all disagreements are in the unsure zone.

## What I did

- Unit liveness over the box [-3,3]^2: units 2, 4, 12 can never fire (max of the affine part is < 0);
  unit 3 fires only in a tiny strip near x1 = 3 with output weight 0.014 -> irrelevant.
- Units 5 and 6 have bias ~6 and small slopes, so they are always on: they are a linear term
  (~59.6 - 3.1 x0 + 2.06 x1), the "inside" plateau.
- The remaining 10 units all have output weight ~ -3.3..-4.5 and bias ~ -2..-3 with directions spread
  around the circle: each is a "wall" that switches on outside a tangent line. Ten walls around the
  origin = a polygon approximating a disc.
- Plotted the sign on a grid (a round blob), then bisected the decision boundary along 24 rays:
  radius 2.00-2.04 at every angle. Swept radius 1.9..2.1 against 1M samples: r = 2 has zero confident
  disagreements; 1.95 and 2.05 are already slightly worse/zero, 2.0 is the clean value in the middle.
- Exact equality is not possible with a short formula (the net's boundary is a polygon with slightly
  uneven facets), so snap is the target.

## Heuristics

- First check which hidden units are dead or always-on over the input box (max/min of the affine part
  at the box corners). Dead units go; always-on units fold into a linear term.
- When many units share an output-weight sign and similar |bias|, with weight directions spread over
  all angles, the net is building a convex region from tangent half-planes (polygon ~ disc/ellipse).
  Try a quadratic form `c - x0^2 - x1^2` (or with shifted centre / axis scales).
- Find the boundary by bisection along rays from the region's centre; constant radius -> circle,
  snap the radius to a nice number (here 2, so the program is `4 - r^2`).
- Before writing anything, print an ASCII sign map of the output on a grid: for 2-input binary nets
  the shape is usually recognisable immediately.
- Validate a candidate threshold by sweeping it and counting disagreements outside the unsure zone;
  pick the simple value in the middle of the zero-disagreement interval.
