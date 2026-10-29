# Agent notes

## What the network does
Two units carry the decision: h1_3 = tanh(6.25*(x1 - x0) - 1.256) and h1_7 = tanh(6.37*(x1 - x0) + 1.264).
Both have input weights of nearly equal size and opposite sign on x0 and x1, so they depend only on
d = x1 - x0. Their thresholds are d = +0.20 and d = -0.20, and the output weights (-11.85, +11.67) are
nearly equal and opposite, so together they form a bump: high when -0.2 < d < 0.2, low outside.

The other six units have tiny slopes (|w| <= 0.11) and large biases. On [-2, 2]^2 they stay near
tanh(+-1.8..2.2), roughly +-0.95, so they act as constants. With the output bias they sum to about
-11.5. That offset moves the output from ~+8.5 inside the band to ~-11.5 outside.

A boundary scan along d for x0 in {-2..2} put the crossings at |d| from 0.199 to 0.207. The spread comes
from the small slopes of the "constant" units. Snapped threshold: 0.2.

## Result
`return 0.2 - abs(x1 - x0)`: class 1 iff |x1 - x0| < 0.2 (a "nearly equal" test).
check.py: PASS (snap), 0 disagreements where the net is confident. All 9349 disagreements are in
the unsure zone along the band edges.

## Heuristics
- Find units whose weights on two inputs are nearly equal and opposite (w, -w). They depend on the
  difference of those inputs, so rewrite them in terms of d = xi - xj.
- Units with small weights relative to their bias (|w|*range << |b|) are constants on the input box.
  Fold them, with the output bias, into one offset.
- A pair of steep sigmoid/tanh units on the same direction, with output weights of opposite sign and
  thresholds at -t and +t, is a window/bump: a test of the form |d| < t. The offset decides whether
  the inside or the outside of the window is class 1.
- To find thresholds, scan the real net along the found direction at several offsets and snap the
  average crossing to a simple value (0.2 here). Residual drift of a few thousandths along the band
  usually comes from the near-constant units and is safe to drop for a snap pass.
- For binary outputs, returning a signed margin (t - |d|) keeps the same decision and is easy to read.
