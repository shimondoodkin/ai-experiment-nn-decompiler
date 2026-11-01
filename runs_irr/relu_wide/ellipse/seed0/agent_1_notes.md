# Notes

## What the network does
The dump is a 2 -> 64 -> 1 ReLU net with a binary output (class 1 if score > 0). A 31x31 sign map over
[-3, 3]^2 showed one convex blob of class 1 in the middle and class 0 everywhere else. Most hidden units have
negative output weights and negative biases (they are "walls" that switch on outside the blob and push the
score down); five or six units with large positive biases are always on and just carry the positive constant.
The net is a piecewise-linear polygon approximating a smooth closed curve.

## What I did
1. Bisected the decision boundary along 720 rays from the origin (the origin is inside).
2. Least-squares fit of a general conic `A x0^2 + B x0 x1 + C x1^2 + D x0 + E x1 = 1` to the boundary points.
   Normalised by C: A = 0.2495, B = -0.001, D = -0.245, E = 0.008, constant 0.9325.
3. Snapped: B, E -> 0, A -> 1/4. Completing the square gives centre x0 = 0.49 -> 0.5 and right-hand side
   0.9925 -> 1. So the class-1 region is the ellipse `((x0 - 0.5)/2)^2 + x1^2 < 1`.
4. Wrote it as `4 - (x0 - 0.5)^2 - 4 x1^2` (score > 0 inside). The score scale does not matter for a binary
   output, so I multiplied by 4 to avoid fractions.

Result: `PASS (snap)`, 0 disagreements where the net is confident; all 4000 disagreements are in the unsure
zone right on the boundary, where the polygon and the true ellipse differ slightly. `PASS (exact)` is not
reachable with the smooth formula, since the net is only a 64-facet polygonal approximation.

## Heuristics
- For a binary net on 2 inputs, first draw a coarse sign map of the whole domain; the shape (half-plane, band,
  blob, ring) tells you which formula family to try before reading any weights.
- A ReLU layer where nearly all output weights are negative and biases negative, plus a few units with large
  positive bias that are always active, means "constant minus a sum of walls": a convex region, typically a
  polygon approximating a circle or ellipse.
- To recover a curved boundary, bisect the boundary along rays from an interior point and fit a general conic
  (`A x^2 + B xy + C y^2 + D x + E y = 1`) by least squares. Normalise by the largest quadratic coefficient,
  snap small cross/linear terms to 0, snap ratios to simple fractions, then complete the square to read off a
  centre and radii and snap those too.
- For binary outputs only the sign of the score matters: scale the inequality so all coefficients are integers
  (e.g. multiply `(dx/2)^2 + y^2 < 1` by 4).
- Disagreements that all lie in the unsure zone along the boundary mean the formula is right and the net is a
  piecewise-linear approximation of it; don't try to fit the net's facets.
