# Notes

## What I did
- `spec.json`: 8 float inputs in [0, 1], one regression output.
- About half the outputs (49.7%) were exactly 0, and the rest were positive. That pointed to a ReLU around a
  linear function.
- A least-squares fit on the rows with out > 0 gave `out = 1.0*x2 - 1.0*x5 + 0.0005*(x0+x1+x3+x4+x6+x7) + 0`
  (residual about 1e-7). So the function is `relu(x2 - x5 + (sum of the other six)/2000)`.
- `agent.py` (submitted): `relu(x2 - x5)`, the positive part of the difference ("how much x2 exceeds x5").
  The six 0.0005 coefficients are snapped to 0, which gives a max error of 0.003 against a tolerance of 0.02.
  Result: `PASS (snap)`.
- `agent_exact.py` (for reference): keeps the leak term `(x0+x1+x3+x4+x6+x7)/2000`. Result: `PASS (exact)`,
  max error 1.5e-7. The six inputs share one tiny coefficient, which looks like noise or a perturbation
  rather than part of the logic.

## Heuristics
- Check what fraction of outputs are exactly 0 (or sit on another floor). A large fraction means a
  ReLU/clamp. Fit a linear model only on the active rows (out > floor), then check that the plain
  `relu(fit)` reproduces every row, including the zeros.
- A linear fit with residual near machine precision means a single active piece. There is no need to search
  for more regions.
- Coefficients that are all the same tiny value (here 5e-4) on the "irrelevant" inputs are a uniform leak or
  perturbation. Snap them to 0 for the readable version, and keep them as one shared term (`sum/2000`) if
  exactness matters.
- Before snapping, estimate the worst-case error: the coefficient times the number of inputs times the
  input range (0.0005*6 = 0.003). Compare it with the snap tolerance.
- Give the result a name by its shape: `relu(a - b)` is "how much a exceeds b" (a hinge or positive
  difference).
