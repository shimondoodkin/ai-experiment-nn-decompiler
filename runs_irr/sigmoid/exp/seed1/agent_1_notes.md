# Agent notes

## What I did

- The dump is a 1-input, 8-unit sigmoid layer feeding a linear output, on x0 in [-2, 2].
- Tabulated the net on a 0.1 grid (scratch/look.py). The values were 0.1256 at -2, 0.9967 at 0,
  2.7192 at 1 and 7.333 at 2. That is e^x0 (e^1 = 2.7183, e^2 = 7.389). The output is convex,
  positive and grows by about a factor e for each unit step in x0.
- The structure agrees: there is one gentle unit and a pile of steeper units, all centred near the
  top edge (x0 ≈ 1.85 to 2.08). Unit 4 is decreasing, but sigmoid(-z) = 1 - sigmoid(z), so it is one
  more rising step near x0 ≈ 2.06 plus a constant. Together they make a rising, convex curve.
  The net only matches exp in that the sum of S-curves bends upwards near the edge.
- `agent.py` is `return exp(x0)`. Check result: PASS (snap), max |program - net| = 0.056 at
  x0 = 2, against a snap tolerance of 0.144 (2% of the output range). The largest errors are at the
  right edge: the sigmoid sum flattens there and comes in below e^2. PASS (exact) is not possible
  with a closed formula. It would need the 8 sigmoids back.

## Heuristics

- For 1-input regression, tabulate the net on a coarse grid first, then compare it with a short
  list of candidate functions (x, x^2, |x|, exp, log, sqrt, sin-like shapes). Key values give
  the answer away: f(0) ≈ 1, f(1) ≈ e and f(-1) ≈ 1/e mean exp.
- Fold a decreasing sigmoid into a rising one with sigmoid(-z) = 1 - sigmoid(z). Its weight
  changes sign and the constant moves into the bias. Units with nearly equal centres b/w then merge.
- If many unit centres sit at one edge of the domain, the net is building curvature there (a
  convex or concave tail). Expect a smooth nonlinear target, not a step function.
- Find the largest error of the closed formula. It usually sits at a domain edge where the
  sigmoids saturate. Check it against the snap tolerance before trying to fit coefficients:
  the plain formula may already pass.
