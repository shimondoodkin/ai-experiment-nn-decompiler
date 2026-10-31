# Notes

## What I did

- The net is 1 input -> 32 ReLU -> 1, so it is a piecewise-linear function of x0. I computed every kink
  (-b/w) inside [-2, 2] and the slope of each segment between kinks (scratch/an.py).
- The function is convex and its slopes rise steadily: about 0.18 near x=-1.7, 0.98 near x=0, 2.7 near
  x=1, 6.8 near x=2. Its values are f(-2)=0.142, f(0)=1.009, f(1)=2.732, f(2)=7.391, which match
  e^-2=0.135, e^0=1, e^1=2.718 and e^2=7.389. Each segment's slope is close to e^x at its midpoint.
  So the function is the exponential.
- `agent.py` is `return exp(x0)`. The checker gives PASS (snap) over 262,593 inputs, with max
  |program - net| = 0.025 (near x=1.756, where a chord of the convex curve sags the most) against a
  snap tolerance of 0.145.
- An exact pass would mean keeping about 23 kinks, which is the dump itself in another form. The one
  formula is the readable answer. Its error is the net's own approximation error, not the formula's.

## Heuristics

- For a 1-input ReLU net, list the kinks (-b/w) inside the domain and the slope of each segment. The
  slope sequence is the derivative, sampled. Monotone rising slopes mean a convex function. Slopes that
  are proportional to the function's value mean an exponential. Slopes that grow linearly mean a square.
- Before fitting, evaluate the net at a few landmark inputs (0, ±1, the domain ends) and compare the
  values with known constants (1, e, e^2, pi, sqrt 2, ln 2). f(0)≈1 together with f(1)≈2.72 points
  straight at exp.
- Kinks whose -b/w falls outside the input box are just linear terms (or zero) on the domain. Kinks
  that fall very close together (for example 1.657, 1.668 and 1.672 here) are one kink split across
  several units. Both can be dropped or merged when you write a compact version.
- When the net looks like a smooth function, try the smooth closed form first (the language has exp,
  log, sqrt, tanh). Its error against the net is usually the net's chord error, which stays well
  inside the snap tolerance.
- The worst disagreement between a smooth formula and a ReLU net sits in the middle of the widest
  segment where the curvature is highest. Check there first.
