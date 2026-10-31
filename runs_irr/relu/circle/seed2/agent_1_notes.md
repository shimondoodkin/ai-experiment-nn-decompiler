# Notes

## What the network does

2 inputs in [-3, 3], 16 ReLU units, binary output (class 1 when the score > 0).

- Unit 11 (`relu(0.18*x0 - 0.011*x1 + 7.94)`) is always active on the box: a near-constant bias
  (about 5.63 * 7.94 ≈ 44.7 added to the output bias 2.94).
- Unit 5 is almost never active (bias -0.35, tiny weights) and its output weight is tiny: dead.
- The other 14 units all have negative output weights and biases of about -2 to -3.9, with weight
  directions spread around the circle. Each one switches on outside a line, and together they cut a
  convex polygon out of the plane: the classic ReLU approximation of a disk.
- I printed the decision map on a grid and measured the boundary radius along 24 directions: it is
  1.99–2.03 everywhere. So the network is `x0² + x1² < 4`.

## Program

```python
def net(x0, x1):
    r2 = x0 * x0 + x1 * x1
    return 4 - r2
```

`python check.py agent.py`: PASS (snap), 0 disagreements where the net is confident. The 5399
disagreements are all in the unsure zone, along the polygon edges that approximate the circle. An exact
pass would need the 14-sided polygon itself, which says nothing more than "disk of radius 2". So I kept
the formula.

## Heuristics

- Draw the binary decision map on a coarse grid first. The shape (disk, band, half-plane, quadrant)
  often gives the formula straight away.
- If many units have negative output weights, negative biases of similar size and directions spread
  evenly around the circle, the net is a polygon approximating a round region. Measure the boundary
  radius along rays from the centre. If it is constant, use `x0*x0 + x1*x1 < r*r` (no `**` needed).
  Snap r to a simple value.
- A unit whose bias is large compared with |w|·range is always on (or always off). Fold it into the
  constant, or drop it.
- Units with tiny input weights *and* tiny output weights are dead weight. Drop them.
- For binary outputs, returning a smooth score (`4 - r2`) is as good as an if/else and shorter. Only
  the sign matters.
- When the boundary is a curve that the net approximates with straight pieces, expect only a snap pass.
  Do not chase exactness by reproducing the polygon.
