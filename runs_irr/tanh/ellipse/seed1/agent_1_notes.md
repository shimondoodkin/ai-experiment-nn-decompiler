# Notes

## What the network does

The net (2 inputs, 16 tanh units, binary output) is the inside of an ellipse:

    class 1  iff  ((x0 - 0.5) / 2)^2 + x1^2 < 1

centre (0.5, 0), semi-axes 2 along x0 and 1 along x1. `agent.py` returns `1 - (u*u + v*v)` with
`u = (x0 - 0.5)/2`, `v = x1`. Check: PASS (snap), 0 disagreements where the net is confident;
the 5465 disagreements are all in the net's unsure zone (the net's boundary is a 7-sided smoothed
polygon, so it can only approximate the curve).

## How I got there

1. Nine of the 16 units have weights of about 0.01-0.02 and biases of about ±2.6 to ±3.3, so they are
   saturated tanh units: together they add an almost constant -35.2 (range 0.06 over the box) to the
   output. They are pure bias.
2. The other seven units have weight vectors pointing in seven directions spread around the circle,
   and their output weights all have the sign that makes the region inside them positive: a smoothed
   convex polygon, i.e. a sampled convex curve.
3. An ASCII plot of the decision region on a grid showed one convex blob, wider in x0 (about -1.5 to
   2.5) than in x1 (about -1 to 1).
4. I bisected the decision boundary along 720 rays from the centroid of the positive region and fitted
   a general conic (SVD null vector of [x², xy, y², x, y, 1]). Result, normalised:
   x² + 0.0075·xy + 4.13·y² - 0.99·x - 0.01·y - 3.85 = 0.
5. Snapping: xy and y terms → 0, 4.13 → 4, -0.99 → -1. The constant: completing the square gives
   (x - 0.5)² + 4y² = c + 0.25. The nice choice c + 0.25 = 4 (semi-axes 2 and 1) gives c = 3.75; it
   passes. The raw fitted 3.85 and the "round" 4 both fail at a confident point near x1 = -1.03, so the
   completed-square form is the one to snap, not the raw constant.

## Heuristics

- Saturated units (|w| tiny compared to |b|, |b| ≥ 2.5 for tanh) are constants: fold them into the bias
  before looking at anything else. In this net 9 of 16 units were such dead/constant units.
- A binary net whose live units all face outward from one region (directions spread around 360°, output
  weights all with the inward-positive sign) is a convex region; with few inputs and float inputs, try
  a conic before a polygon.
- To find the curve: bisect the decision boundary along rays from the centroid of the positive region,
  then fit a general conic by the SVD null vector of [x², xy, y², x, y, 1]. Rotation (xy) and offset
  terms near 0 snap to 0.
- Snap the conic in completed-square form (centre, semi-axes) rather than its raw constant: the fitted
  constant is biased by the polygon approximation, and "nice" values live in the centre/axes, not in
  the expanded constant (here 3.75 = 2² - 0.5², not 4 or 3.85).
- Try a couple of nice candidates with the checker; a candidate failing at only a handful of confident
  points along one side points to a slightly wrong constant or centre.
- The returned score scale does not matter for binary output; write the program in the most readable
  normalised form (1 - (u² + v²)).
