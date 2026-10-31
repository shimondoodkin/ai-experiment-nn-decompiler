# Notes

## What the network computes

The dump is 1 input -> 8 tanh units -> 1 output, for regression on x0 in [-2, 2].

- I sampled it on a grid. It is positive, increasing and convex: 0.119 at -2, 0.992 at 0, 2.708 at 1 and 7.335 at 2.
- Each step of 0.1 multiplies the value by about 1.105. That pointed to exp(x0), since exp(0.1) = 1.105 and the
  values at 0 and 1 are close to 1 and e.
- I compared the net with exp(x0) over 4001 points. The residual is within 0.017 on [-2, 1.9]. It reaches -0.054
  at x0 = 2, where the tanh units saturate and the net falls a little below exp. The snap tolerance is 0.144
  (2% of the output range).
- Polynomial fits needed degree 4 (max error 0.037) to come close. exp(x0) is simpler and is the intended shape.

The units:
- Five units (0, 1, 2, 5, 6) have slopes of about 0.06 and biases of about ±1.1 to 1.5. On [-2, 2] they are almost
  linear, and together they add a nearly constant offset and a small linear and curved term.
- Units 3, 4 and 7 (slopes 0.81, -1.20 and 1.86) bend the curve. Unit 7 (centre x0 ≈ 2.08) gives the steep rise
  near the top, unit 4 (centre ≈ 1.61) the middle, and unit 3 (centre ≈ 0.74) the lower part.

## Result

`agent.py` is `return exp(x0)`. The check gives PASS (snap): max |program - net| = 0.054 and no input is beyond the
tolerance. An exact pass would need all 8 tanh units, which would just be the dump again.

## Heuristics

- For a 1-input regression net, sample it on a fine grid first and look at the shape before simplifying units. Check
  ratios between equal steps (constant ratio = exponential), constant differences (linear) and second differences
  (quadratic), and look at the values at 0, 1 and the endpoints for 1, e, pi, sqrt 2 and so on.
- Check the residual against the candidate formula over the whole domain. The largest errors are usually at the domain
  edges, where tanh/sigmoid units saturate and the net cannot follow a fast-growing function.
- Tanh units with |slope × domain half-width| well below 1 and a sizeable bias act as near-linear terms. Several of
  them together form a constant plus a gentle linear and curved term, and are not features of their own.
- The bend points of the steep units, at -bias/slope, show where the target's curvature changes. For a convex
  increasing target they bunch toward the high end.
- When a closed-form function from the allowed set (exp, log, sqrt, abs, min, max) passes the snap check, prefer it
  over a fitted polynomial or a pruned net. It is shorter, and it is probably the function the net was trained on.
