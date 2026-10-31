# Notes

## What I found

- The dump is 2 inputs -> 64 ReLU units -> 1 linear output, regression, inputs in [-3, 3]^2.
- Almost all hidden biases are close to 0 (|b| < 0.05), and the units with large negative biases
  have tiny output weights. Units 12, 14, 44 and 63 have weights around 1e-8 and do nothing. So the net
  is close to a sum of `a_i * relu(w_i . x)`, which is positively homogeneous: f(t*x) = t*f(x).
- Probing: on circles of radius r the output is r + 0.005 in every direction (0.5 -> 0.50, 1 -> 1.00,
  2 -> 2.01, 3 -> 3.01), and the corners (+-3, +-3) give 4.26, close to 3*sqrt(2) = 4.243. Many
  ReLU units with spread-out directions approximate a round cone.
- So the function is the Euclidean norm, `sqrt(x0*x0 + x1*x1)`.

## Result

`agent.py`: `return sqrt(x0 * x0 + x1 * x1)`, which gives **PASS (snap)**. Max |program - net| = 0.030,
at the origin, where the net puts its output bias plus small offsets (0.03 instead of 0). Snap
tolerance is 0.085. PASS (exact) is not possible with a closed formula: the net is a polygonal
(piecewise-linear) approximation of the cone, and the formula is the smooth cone it approximates.

## Heuristics

- Look at the biases first. If nearly all ReLU biases are about 0 (and the units with large biases have
  tiny output weights), the net is positively homogeneous of degree 1. Then the function is fixed by its
  values on the unit circle, so probe f(r*cos t, r*sin t) for several r and t.
- If f is linear in r and the same in every direction, it is the Euclidean norm. If it changes with
  direction, compare with |x0|+|x1| (corners at 6) and max(|x0|,|x1|) (corners at 3), or a weighted norm.
- Many ReLU units whose weight directions are spread around the circle, all with positive output weights,
  are the sign of a convex, rounded function such as a norm. A few sharp directions mean abs/max.
- Drop units whose weights are around 1e-7 or whose output weight times the largest activation on the
  input box is far below the tolerance.
- Probe the net on a coarse grid and on rays before you snap individual weights. When a known formula
  fits the probes, try it in the checker first: it can replace dozens of units in one step.
- A small constant left at the origin (here 0.03) is the net's fitting error. Leave it out unless the
  checker needs it.
