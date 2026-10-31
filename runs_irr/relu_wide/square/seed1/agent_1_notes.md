# Agent notes

## What the network does

One input `x0` in [-2, 2], 32 ReLU units, regression output. With one input every unit is a hinge
`relu(w*x0 + b)` with its kink at `x0 = -b/w`, so the net is a piecewise-linear curve in x0.

- Sampling it on a grid gave f(0) ~ 0, f(+-0.5) ~ 0.25, f(+-1) ~ 1.00, f(+-1.5) ~ 2.24, f(+-2) ~ 3.98:
  symmetric and quadratic.
- The kinks inside the domain are spread roughly evenly (about every 0.25-0.3) and almost all add
  slope of about +0.5 each. Constant positive slope change at evenly spaced knots is a discretised
  second derivative of 2, which is what x^2 has.
- Units 4 and 15 are dead (weights ~1e-5 or smaller), and units 0, 1, 30 have their kinks outside or on the edge
  of the domain; units 6, 9, 17, 27 and a few others add only small corrections.

## Result

`agent.py` returns `x0 * x0`.

- `python check.py agent.py`: PASS (snap), max |program - net| = 0.020 (at x0 = 2, where the net
  undershoots to 3.98), snap tolerance 0.080.
- I did not chase PASS (exact): that would mean keeping the ~20 in-domain hinges with their
  trained coefficients, which reproduces the network's approximation error rather than the
  function it represents.

## Heuristics

- For a 1-input ReLU net, first list each unit's kink `-b/w` and its slope change `w*v*sign(w)`
  sorted by position, and sample the output on a coarse grid. That turns the net into a readable
  spline in a few lines of code.
- Drop units whose kink lies outside the input box (they are linear there and fold into the bias
  and slope) and units whose weights are ~1e-5 or smaller (dead).
- Evenly spaced kinks with near-equal positive slope changes mean a convex quadratic: slope change
  per unit length ~ 2a gives a*x^2. Negative changes mean concave. Changes that grow or shrink
  along the axis point to cubic or exponential shapes; one big kink means abs/relu/clamp.
- Check symmetry by comparing f(x) and f(-x) on the grid; a symmetric net with f(0) ~ 0 and f(1) ~ 1
  is very likely x^2 (or |x| if the slope is constant on each side).
- Before writing a formula, compute the max error of the candidate against the net over a dense grid
  and compare it with the snap tolerance (2% of the output range). The largest error is usually at the
  domain edges, where the network has the fewest knots.
