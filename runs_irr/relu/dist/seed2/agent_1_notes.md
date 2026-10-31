# Notes

## What I did

- The dump is 2 -> 16 ReLU -> 1. Every hidden bias is close to 0, and 15 of the 16 output weights are
  positive. A sum of positive ReLUs of lines through the origin is a convex, positively homogeneous
  function: a polygonal norm.
- I evaluated the net on a grid and along rays (angles every 30 degrees, radius 0..3). Along every ray
  the output is radius * 1.00 (within about 1%), at every angle. So the net is a 16-sided polygon
  approximation of the Euclidean norm.
- `agent.py`: `sqrt(x0*x0 + x1*x1)`.
- Check: `PASS (snap)`, empirical, 318805 inputs. Max |program - net| = 0.057, at the origin, where the
  net has a residual offset of 0.057 (output bias 0.053 plus the small hidden biases). The snap
  tolerance is 0.084. Away from the origin the error is under about 0.02.
- Exact pass is not reachable with a closed formula: the net is piecewise linear, the norm is not.
  The clean formula is the intended function, the polygon facets are training noise.

## Heuristics

- All hidden biases near 0 means the hidden layer is positively homogeneous: probe along rays from the
  origin. If output/radius is constant per direction, the function is homogeneous of degree 1.
- If additionally the ratio is the same in every direction, it is the Euclidean norm
  `sqrt(x0*x0 + x1*x1)`. If it varies with direction like |cos| + |sin|, try the L1 norm; if like
  max(|cos|, |sin|), try the L-infinity norm (`max(abs(x0), abs(x1))`).
- Mostly positive output weights over ReLUs of near-origin lines spread around all directions = convex
  function built from tangent planes (a norm, a distance, or a quadratic bowl). Check symmetry under
  x -> -x and swapping inputs before guessing a formula.
- A small constant at the origin (output bias plus leftover hidden biases) is a fitting artefact; drop
  it when it is within the snap tolerance rather than adding a constant to the formula.
- Sample the net in numpy from `weights.npz` directly (keys W0, b0, W1, b1) to explore quickly, then
  confirm with `check.py`.
