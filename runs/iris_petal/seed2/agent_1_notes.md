# Notes

## What I changed and why

The dump is a 2 -> 8 -> 3 ReLU MLP with argmax output. Inputs are x0 in [1, 6.9] and x1 in [0.1, 2.5].
The ranges look like iris petal length and width, and the three classes look like the three species.
I did not try to simplify the 8 hidden neurons one at a time. For an argmax net only the decision
regions matter, so I read the regions straight off the input plane:

1. Printed a coarse grid of the predicted class over (x0, x1). It shows three bands ordered by
   size: class 0 at small x0/x1, class 1 in the middle and class 2 at large values. They are split
   by two oblique boundaries.
2. Bisected each class transition along x0 at x1 = 0.1, 0.2, ..., 2.5, then looked at how the
   boundary x0 changes with x1:
   - 0|1 boundary: the slope is a steady -2.22, so x0 + 2.22 * x1 is about 4.40 everywhere.
     Snapped to `x0 + 2.25 * x1 < 4.4` (9/4). The simpler `x0 + 2 * x1` and `x0 + 2.2 * x1` both
     failed where the net is confident.
   - 1|2 boundary: this is piecewise linear with a kink near x1 = 0.55. Above the kink the slope is
     -1.09, so x0 + 1.1 * x1 is about 6.77. Below it the slope is shallow (-0.37) and the boundary
     sits at x0 of about 6.2 to 6.33. I wrote this as an OR of two half-planes:
     `x0 + 1.1 * x1 > 6.75 or x0 > 6.3`. `x0 + x1 > 6.6` alone failed at both ends: too lenient
     at small x1 and too eager just above the kink.
3. Wrote the result as a decision list: small -> 0, then large -> 2, otherwise 1.

Result: `check.py` gives PASS (snap) with 0 disagreements where the net is confident. The
remaining disagreements are all inside the net's unsure zone, mostly near the 0|1 line at the
x0 = 1 edge, where the net's boundary bends slightly.

Final program:

```python
def net(x0, x1):
    small = x0 + 2.25 * x1 < 4.4
    if small:
        return 0
    large = x0 + 1.1 * x1 > 6.75 or x0 > 6.3
    if large:
        return 2
    return 1
```

## Heuristics

- For argmax or classification nets with few inputs, skip the weights. Map the decision regions
  on a grid first. The region layout (bands, wedges, nesting) tells you the shape of the program.
- If the classes form ordered bands, write a decision list: peel off one extreme class, then
  the other, and make the middle class the default.
- Find boundaries by bisecting along one axis at many fixed values of the other. Constant
  differences give the slope. A sudden change in slope means a ReLU kink, and that usually
  becomes an `or`/`and` of two half-planes (convex or concave corner) rather than a single line.
- Normalise each boundary line to `x0 + k * x1 <op> c`, then snap k to a simple fraction (1, 1.1,
  9/4 and so on) and c to one decimal. Try the simplest candidate first, but expect to keep a
  second significant digit when the slope is far from an integer. The checker's confident zone
  is narrow enough that 2.2 vs 2.25 can matter.
- The worst counterexamples sit at the domain edges (x1 = lo/hi). That is where a wrong slope
  produces the largest offset, so check the line's fit at both ends of the range.
- Small regions near domain corners, where the net bends at the very edge, can usually be ignored
  because they fall inside the unsure zone.
