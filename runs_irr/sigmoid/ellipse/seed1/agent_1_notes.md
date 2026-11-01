# Notes

## What the network does
2 inputs in [-3, 3], 16 sigmoid units, binary output (class 1 if value > 0).
An ASCII plot of the decision region on a 0.1 grid showed one convex blob, wider in x0 than in x1,
shifted right of the origin: an ellipse.

## How I got the program
1. Bisected the decision boundary along 360 rays from a point inside the blob (0.5, 0).
2. Fitted a general conic `a x0^2 + b x0 x1 + c x1^2 + d x0 + e x1 + f = 0` to the boundary points
   (smallest singular vector). Normalised: `x0^2 + 0.001 x0 x1 + 4.07 x1^2 - 0.99 x0 - 0.001 x1 - 3.79`.
3. Snapped: cross term and x1 term -> 0, x1^2 coefficient -> 4, x0 term -> -1 (centre 0.5).
   The boundary extents (x0 in [-1.51, 2.51], x1 in [-1.00, 1.00]) give semi-axes 2 and 1,
   so the constant is 4 - 0.25 = 3.75 after expanding, i.e. `(x0-0.5)^2 + 4 x1^2 < 4`.
4. The 16 sigmoid units are a polygon-like approximation of this ellipse (each steep unit is one
   edge of it; the two near-constant units h1_0 and h1_8 are just bias). None of them is kept.

Result: `4 - (x0-0.5)^2 - 4 x1^2`, PASS (snap); all disagreements are in the net's unsure zone
(close to the boundary). PASS (exact) is not reachable for a sigmoid net with a quadratic program.

## Heuristics
- For a 2-input binary net, first plot the decision region on a coarse grid; the shape (one convex
  blob, band, half-plane, corners) tells you the formula family before reading any weights.
- Many steep hidden units with weights of similar size and directions spread around the circle,
  all feeding the output with large weights of the same sign, are a polygonal fence around a convex
  region: look for a circle / ellipse, not for per-unit logic.
- Units with tiny input weights (|w| < 0.1) and moderate bias are constants; fold them into the bias.
- To recover a closed boundary: bisect it along rays from an interior point, fit a general conic by
  SVD, then snap coefficients (drop near-zero cross/linear terms, round the ratio of squared terms to
  an integer, read the centre and semi-axes from the boundary extents and snap them to half-steps).
- Return the signed margin (inside positive) scaled to integer coefficients; for binary output only
  the sign matters, so scale freely to make constants nice.
