# Notes

## What the network is

2 inputs in [-3, 3], 64 ReLU hidden units, binary output (class 1 if score > 0).

The hidden units fall into two groups:

- 17 units with a large positive bias (+2.8 .. +4.0) and tiny weights (|w| < 0.35). On the box
  they are always active, so they are just one affine term; with positive output weights they
  add a roughly constant "inside" score (together with the output bias 2.18).
- 41 units with weight vectors of norm about 2.5-3.5 pointing in all directions and bias about
  -1.5 .. -2.4, each with a negative output weight. Each one switches on once the input moves
  past a line at distance ~0.7 from the origin in its direction, and then pushes the score down.
  Together they form a polygonal "cone" that grows with the distance from the origin.
- A few units (0, 1, 5, 24, 27) have tiny weights and tiny output weights; they are negligible.

Sampling the decision boundary along 24 rays from the origin gives a crossing radius between
1.976 and 2.013 in every direction, so the class-1 region is a disc of radius 2.

## Program

    radius_sq = x0*x0 + x1*x1
    return 4 - radius_sq          # class 1 iff x0^2 + x1^2 < 4

`check.py`: PASS (snap), 0 disagreements where the net is confident; the 5440 disagreements are
all in the net's unsure zone (the net's polygon boundary vs the true circle, max radial gap ~0.024).
An exact match would need the whole 41-facet polygon, which is the network itself, so I stopped at
the formula.

## Heuristics

- Split hidden units by |bias| vs weight norm times the input range: units whose bias keeps them
  always on (or always off) over the box are affine (or zero) and fold into the output constant.
- Many ReLU units with similar bias/weight-norm ratio, weight directions spread around the circle
  and same-sign output weights are a piecewise-linear approximation of a radial function
  (|x| or |x|^2); the boundary is then a circle/sphere centred where those lines are symmetric.
- For binary 2-input nets, print an ASCII map of the decision region first, then measure the
  boundary along rays from the apparent centre; a near-constant crossing radius means a disc,
  and snapping the radius to the nearest integer/half usually lands in the unsure zone.
- For binary outputs, return a smooth score (e.g. r^2 - x^2 - y^2) rather than the comparison, so
  the sign carries the decision and the magnitude still says how confident it is.
- Drop units whose output weight times maximum activation is far below the score margin.
