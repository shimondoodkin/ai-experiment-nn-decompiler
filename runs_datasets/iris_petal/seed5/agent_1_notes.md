# Notes

## What the network does
2 inputs -> 8 ReLU -> 3 logits, argmax. The input ranges (x0 in [1, 6.9], x1 in [0.1, 2.5]) and the
three classes look like the iris petal length / petal width classifier:
class 0 = small petals (setosa), class 1 = medium (versicolor), class 2 = large (virginica).

## What I changed and why
- Instead of simplifying the 8 hidden units one by one, I mapped the decision regions directly
  (grid print of the class map, then bisection of the class boundary along x0 for many x1 values).
  The map has three bands ordered along x0, separated by two nearly straight boundaries; class 0 and
  class 2 never touch, so a two-test if-chain is enough.
- Boundary 1/2 is straight: x0 = 6.725 - 1.06*x1. Snapping the slope to 1 leaves a drift of
  0.06*x1; the threshold that keeps both ends inside the net's unsure band (half-width ~0.09 in x0)
  is 6.65, giving `x0 + x1 > 6.65`.
- Boundary 0/1 is slightly convex (slope -1.59 at small x1 bending to about -2.1 near x1 = 1.5,
  where it leaves the domain). Fitting one line through it: the slope 7/4 keeps
  x0 + 1.75*x1 within 3.78..3.91 along the whole boundary, so `x0 + 1.75*x1 < 3.85` stays inside the
  unsure band (half-width ~0.11). Slope 2 was too far off (spread 0.39).
- All the hidden-layer arithmetic is gone. Final program:

```python
def net(x0, x1):
    if x0 + x1 > 6.65:
        return 2
    if x0 + 1.75 * x1 < 3.85:
        return 0
    return 1
```

`python check.py agent.py` -> PASS (snap): 0 disagreements where the net is confident; the
remaining disagreements are all inside the net's unsure zone (score < 0.619). PASS (exact) is not
reachable with straight lines because the net's 0/1 boundary is a polyline with several ReLU kinks.

## Heuristics
- For low-dimensional argmax nets, map the decision regions first (coarse grid print) and then
  locate each boundary precisely by bisection along one input; this is far cheaper than algebra on
  hidden units, most of which only shape the logits away from the boundary.
- Measure the width of the "unsure" band (where top-two logit gap < the checker margin) along each
  boundary; it tells you how much slope/threshold error a snapped line can afford.
- For a nearly straight boundary, snap the slope to a simple value (1, 7/4, 2, ...) and pick the
  threshold as the midpoint of the range that a·x0 + b·x1 takes along the true boundary; accept the
  snap if half that range is below the band half-width, otherwise try the next simple slope.
- If two classes never share a boundary, a short if-chain of linear tests reproduces argmax.
- Recognising the dataset (here iris petal measurements) helps name variables and sanity-check the
  class order, but the network's extrapolation outside the training data still has to be matched,
  because the checker samples the whole input box.
