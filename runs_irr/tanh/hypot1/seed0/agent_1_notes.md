# Notes

## What the network does

2 inputs in [-2, 2], 16 tanh units, regression. Tabulating it on a grid showed a bowl that is
symmetric in x0 and x1 and in their signs: about 1.03 at the origin, 1.40 at (1, 0), 1.73 at (1, 1),
2.21–2.27 at (±2, 0), and 2.92–2.95 at the corners. These values match sqrt(1 + x0² + x1²)
(1, 1.414, 1.732, 2.236, 3). That is the distance from (x0, x1, 0) to (0, 0, -1), or hypot(1, x0, x1).

## What I changed

- I replaced the 16 tanh units with `sqrt(1 + x0*x0 + x1*x1)`.
- The plain formula fails: max |error| is 0.080 and the snap tolerance is 0.0385. All of the excess
  is in the four corners, where the net sags about 0.05–0.08 below the formula. Away from the corners
  the residual is a ±0.035 ripple with no structure (tanh fitting noise).
- I added the corner sag as `0.08 * relu(|x0| + |x1| - 3.4)`. A minimax fit gave k = 0.083,
  t = 3.44, and I snapped them to 0.08 and 3.4. With the snapped values the max error is 0.0374,
  which is the floor set by the ripple, so the snapping costs nothing.
- I also tried an affine-scaled `a*sqrt(c + R) + d`, a radial clamp `min(..., m)` and an |x0·x1|
  corner term. None of them did meaningfully better (best 0.0365), so I kept the clean form.
- Result: `PASS (snap)`, max |program − net| = 0.0374 (tolerance 0.0385). Exact is out of reach for a
  closed formula: the net's own ripple around the formula is ±0.035.

## Heuristics

- Tabulate the net on a coarse grid first (9×9 is enough for 2 inputs) and look at symmetries
  (x↔y, sign flips) and values at the origin, the axes and the diagonals. Compare those values with
  sqrt of small integers, squares and similar landmarks to guess the closed form.
- Many tanh units with tiny weights (|w| ≈ 0.05) and biases near ±1 work in the near-linear part of
  tanh. Together they add a smooth low-order polynomial along one direction. They are a strong hint
  that the function is smooth (a polynomial or root), not piecewise.
- Fit polynomials of rising degree to see whether the function is low order. Here degree 2 left 0.34
  and degree 4 left 0.06, which points to a non-polynomial such as a sqrt.
- Once a formula is close, map the residual (net − formula) on a grid. Fix any structured part
  (here the corners, where the training data runs out and the net bends inwards) with a small
  relu correction in an L1 or |x·y| coordinate, and treat the rest as the net's fitting noise.
- Fit the correction by minimax (Nelder–Mead on an L^p norm with p rising to 80) rather than least
  squares, because the check is a max-error test. Then try nearby round values and keep any that
  don't raise the max error.
- If the tolerance floor comes from unstructured ripple, stop; no clean formula can beat it.
