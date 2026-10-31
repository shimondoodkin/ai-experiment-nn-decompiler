# Notes

## What the network does

Output is binary (class 1 when score > 0). Result: **class 1 iff x0² + x1² < 4**, i.e. the point lies
inside the disk of radius 2 centred at the origin.

`agent.py` returns `4 - (x0*x0 + x1*x1)`. Check: `PASS (snap)`; every disagreement (2.8% of inputs) is
in the net's unsure zone, in a thin ring around r = 2.

## How I got there

1. Classified the 16 hidden sigmoid units by their output weight and their line's distance |b|/|w|
   from the origin:
   - 3 units (h1_0, h1_8, h1_14) have weights ≈ 0 on the inputs and bias ≈ +3.5..4.3: they are constants
     (≈ 0.97–0.99) that fold into the output bias.
   - 8 units with output weight ≈ −11.5 and bias ≈ −8: each fires when the point crosses a line about
     1.75 from the origin, outward in one of 8 directions. Crossing any of them pushes the score strongly negative.
   - 5 units with output weight ≈ +5.4 and bias ≈ +8: on near the origin; each turns off beyond a line
     about 1.7 from it, in another 5 directions.
   - The 13 directions together cover the circle every ~20–35°: a polygon/soft approximation of a disk.
2. At the origin the score is +17.5 (all positive units on, no negative unit on).
3. I bisected the sign change along 36 rays: the boundary radius is 1.973–2.009, mean ≈ 1.99, so I
   snapped it to r = 2 (r² = 4).
4. The polygon wobble cannot be reproduced exactly by a simple formula, and it is all inside the unsure
   zone, so the clean disk is the right refactoring.

## Heuristics

- **Constant units:** a sigmoid unit whose input weights are ≈ 0 (≪ the others) is a constant
  sigmoid(b); fold it into the output bias.
- **Distance-from-origin test:** for every unit compute |b|/‖w‖ (the line's distance from the origin)
  and the unit direction w/‖w‖. If many units share one distance and their directions spread around
  the circle, the net is drawing a disk/ring (polygon approximation of x0²+x1² vs r²). Opposite-sign
  output weights with opposite-sign biases are the same "outside the polygon" test written two ways.
- **Ray bisection:** for a radial shape, bisect the decision boundary along evenly spaced rays from
  the centre (the score's maximum point). Mean radius → snap to a nice value; the spread tells you whether
  it is really a circle (small, unstructured spread) or an ellipse/polygon (systematic spread).
- The radius the net learns sits slightly beyond the per-line distance (1.75 → 2 here), because the
  steep negative units only win once they are well past half-on. Use the measured boundary, not the line
  distances.
- For binary nets, return a smooth score (`r0² − r²`) instead of a hard 0/1 comparison. The decision is the
  same, and the score keeps the margin information.
