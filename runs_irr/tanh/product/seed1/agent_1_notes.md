# Notes

## What the network is
- 16 tanh units, 2 inputs in [-2, 2], regression output.
- Every unit's input weights are (a, a) or (a, -a) up to about 5%: 7 units read only
  s = x0 + x1 (units 0, 7, 8, 10, 11, 13, 15) and 9 read only d = x0 - x1 (the rest).
  So the net is f(s) + g(d) + c.
- Evaluating the two groups as 1-D curves (scratch/an.py) gave f(s) ~ s^2/4 and
  g(d) ~ -d^2/4 over s, d in [-4, 4] (e.g. f(2) = 1.01, f(4) = 3.96; g(2) = -0.99,
  g(4) = -3.92), and the constant f(0) + g(0) + b2 = 0.0002.
- (s^2 - d^2)/4 = x0 * x1 (polarization identity).

## Result
- `agent.py`: `return x0 * x1`.
- check: PASS (snap), max |program - net| = 0.072 against a tolerance of 0.158; the largest
  errors are at the corners (|x0| = |x1| = 2), where the tanh sums flatten slightly
  (net 3.93 vs 4).
- PASS (exact) is not reachable with a clean formula: the net's own approximation error of
  the squares (~2%) is far above atol. I did not keep the tanh units, since the formula is
  the meaningful program.

## Heuristics
- Group hidden units by direction of their input weight vector. With 2 inputs, directions
  (1, 1) and (1, -1) mean the net is a sum of 1-D functions of x0 + x1 and x0 - x1.
- Evaluate each group as a 1-D curve (subtract its value at 0) and compare with simple
  shapes: s^2/4, |s|, relu, step, linear. Check symmetric points (f(v) vs f(-v)) for even
  or odd functions.
- f(x0 + x1) - f(x0 - x1) with f quadratic is a product: a tanh net with equal and
  opposite (a, a) / (a, -a) units with opposite-sign quadratic groups is multiplication.
- Many tanh units with output weights of mixed sign and biases spread over the range are a
  spline of a smooth curve; fit the curve, not the units.
- Check the net's own constant (sum of group values at 0 plus the output bias); here it
  was ~0, so no constant term.
- Largest residuals of a snapped polynomial sit at the box corners, where tanh saturates;
  that is the net's error, not the formula's.
