# Notes

## What I changed and why

The dump has two hidden ReLU units and a linear output:

- h1_0 = relu(1.008*x0 - 0.986*x1 + 0.006)  -> snapped to relu(x0 - x1)
- h1_1 = relu(-0.972*x0 + 0.971*x1 + 0.008) -> snapped to relu(x1 - x0)
- out  = 1.006*h1_0 + 1.014*h1_1 + 0.0009   -> snapped to h1_0 + h1_1

All weights are within ~3% of +-1 and all biases are below 0.01, so they snap to +-1 and 0.
The two units are mirror images (w and -w), and relu(d) + relu(-d) = |d|, so the whole
network is `abs(x0 - x1)`: the absolute difference (distance) between the two inputs.

`check.py` gives PASS (snap). The largest deviation (~0.08) is at the corners (|x0-x1| = 6),
where the trained weights are slightly off 1 (e.g. 1.006*0.972 slope on one side); that is
a training artifact, not part of the intended function.

## Heuristics

- Snap weights within a few percent of +-1 / small integers, and biases that are tiny relative
  to the input range, to exact values first; then look for structure.
- A pair of ReLU units with opposite weight vectors (w, -w) and equal output weights is an
  absolute value: c*relu(z) + c*relu(-z) = c*|z|. With unequal output weights a, b it is
  a*relu(z) - b*relu(-z), i.e. a piecewise-linear "leaky" form; with a = -b it is linear.
- Weight vectors like (1, -1) indicate a difference x0 - x1; (1, 1) a sum.
- Residual error from snapping grows with input magnitude (slope error times range), so expect
  the biggest mismatches at the corners of the input box; judge snaps by relative slope error.
- Drop output biases that are far below the checker tolerance.
