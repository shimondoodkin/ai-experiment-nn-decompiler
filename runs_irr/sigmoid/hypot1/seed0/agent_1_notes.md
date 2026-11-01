# Notes

## What the network computes

A 2 -> 16 sigmoid -> 1 regression net on x0, x1 in [-2, 2]. Evaluating it on a 9x9 grid showed a
bowl symmetric in both inputs and under swapping them: f(0,0) = 1.02, f(1,0) = 1.41, f(1,1) = 1.72,
f(2,0) = 2.21, f(2,2) = 2.95. These match sqrt(1 + x0^2 + x1^2) (1, 1.414, 1.732, 2.236, 3), i.e.
the Euclidean norm of the vector (1, x0, x1).

## What I changed

- Replaced all 16 sigmoid units with a single closed form, `sqrt(1 + 0.98 * (x0*x0 + x1*x1))`.
- The exact `sqrt(1 + r2)` FAILs only at the box corners (max error 0.058 at (±2, ±2) vs snap
  tolerance 0.039): the net under-shoots there because its sigmoids saturate near the edge of the
  training box. Residuals binned by r^2 were about ±0.01 for r^2 < 6 and dropped to -0.03..-0.05 for
  r^2 > 7.
- Tried one-parameter corrections by minimax fit: `p*sqrt(1+r2)` (p=0.992, err 0.0345),
  `sqrt(1+r2)+c` (0.040), `min(sqrt(1+r2), c)` (0.049), `sqrt(1+p*r2)` (p=0.982, err 0.0335).
  Snapped p to 0.98 -> PASS (snap), max error 0.035. `0.99*sqrt(1+r2)` just missed (0.0393).
- The remaining error (~±0.03 at a fixed radius) is non-radial noise from the 16 units, so no
  simple formula is expected to reach exact.

Result: `PASS (snap)`, max |program - net| = 0.035 (tolerance 0.039).

## Heuristics

- Before reading weights, evaluate the net on a coarse grid and look for symmetries (even in each
  input, symmetric under swapping inputs). Even + swap-symmetric strongly suggests a function of
  r^2 = x0^2 + x1^2.
- For radial functions, compare the grid values against a short list of candidates: r, r^2,
  sqrt(1+r^2), exp(-r^2), 1/(1+r^2), log(1+r^2). Matching f(0), f(1,0), f(1,1), f(2,0) is usually
  enough to pick one.
- Many sigmoid units with small weights and both signs of bias are often just a smooth basis for
  one function; do not try to simplify unit by unit when a global formula fits.
- Nets trained on a box are least accurate at the corners (fewest neighbours, saturating units).
  If the ideal formula fails only at the corners, bin residuals by radius to confirm, then try a
  single-parameter correction (scale inside or outside the formula, offset, cap) fitted by minimax
  (max error, not least squares, since the checker uses the max), and snap the parameter to a
  round value that still passes.
- Prefer putting the correction factor inside the root/argument (sqrt(1 + k r^2)) when it gives a
  lower max error than an outside scale; it bends the curve at large r while leaving f(0) unchanged.
