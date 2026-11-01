# Notes

## What the network does

Class 1 exactly when the point lies inside the disc of radius 2 around the origin:
`x0^2 + x1^2 < 4`. `agent.py` returns `4 - (x0*x0 + x1*x1)`, so the decision is `> 0`.
Check: `PASS (snap)`, 0 disagreements where the net is confident; all 5399 disagreements are in
the unsure zone, right on the boundary.

## How I got there

1. Normalised every hidden unit (direction `w/|w|`, offset `-b/|w|`) and read the output weights.
   - 14 units have an output weight of about -3 to -4.7, offset 0.71-1.08 from the origin and
     directions spread all around the circle (every ~20-30 degrees). Each is a penalty
     "outside this half-plane". Together they make a polygonal cone of value
     going down away from a central plateau.
   - Unit 12 (`relu(7.70 - 0.23 x0 - 0.03 x1)`) is always active in the box [-3,3]^2, so it is
     just a linear term: with the output bias it gives a positive plateau of about
     47.4 - 1.34 x0 - 0.17 x1.
   - Unit 5 (`relu(0.237 x1 - 0.884)`) only fires for x1 > 3.72, outside the box: dead.
2. Scanned rays from the origin every 10 degrees for the first zero of the output: radius
   1.976-2.022 at every angle. That is a circle of radius 2, approximated by a 14-sided polygon.
3. Wrote the circle and checked it. The polygonal approximation of the net differs from the true
   circle by at most ~1%, which stays inside the unsure zone.

An exact-pass version (the dump with unit 5 removed and unit 12 written without `relu`) is in
`scratch/exact_pruned.py`; it passes `PASS (exact)` but is just the polygon, not a formula, so I
kept the circle as the answer.

## Heuristics

- **Normalise each ReLU unit** to (unit direction, signed distance from origin). Units with
  similar distance and directions spread evenly around the full circle, all with the same-sign
  output weight, are a polygon approximating a circle/ellipse: look for a quadratic
  boundary instead of trying to simplify the lines.
- **Ray scan for the decision boundary**: for a binary net, find the first sign change along
  rays from a central point at many angles. A constant radius means a circle `r^2 < R^2`;
  radius varying as `1/sqrt(a cos^2 + b sin^2)` means an ellipse; an offset centre shows up as a
  sinusoidal radius.
- **Dead and always-on units**: evaluate each unit's pre-activation range over the input box
  (corners suffice for an affine function). Never positive means delete it; always positive
  means drop the `relu` and fold it into the constant/linear part.
- **Big constant plateau, many negative penalty units** is the ReLU way of writing
  "inside a convex region": the output is roughly `C - sum of distances outside each edge`.
- A binary net only needs the boundary right: a smooth formula that is right outside the
  net's unsure zone beats a faithful but unreadable copy of the polygon.
- Snap the radius to the nearest simple value (here 2, so `R^2 = 4`) and re-check.
