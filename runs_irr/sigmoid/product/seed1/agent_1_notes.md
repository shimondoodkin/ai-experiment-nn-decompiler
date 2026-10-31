# Agent notes

## What the network computes

`net(x0, x1) = x0 * x1` on [-2, 2]^2 (regression). `check.py`: PASS (snap), max |program - net| = 0.107
against a snap tolerance of 0.157. The largest errors are at the four corners (net gives 3.89-3.91 where the
product is 4), where the sigmoid sums flatten out.

## How I found it

1. Read the hidden-layer weights. Every unit's weight vector is close to (±1, ±1) times a scale (0.73-1.74),
   so there are only two directions: s = x0 + x1 (units 0, 7, 8, 11, 12, 13) and d = x1 - x0 (units 1-6, 9,
   10, 14, 15). The net is therefore g(s) + k(d), a sum of two 1-D curves.
2. Printed the net on a 9x9 grid. The table was antisymmetric in each input, zero on both axes, and equal to
   the product of the row and column labels to about 0.01 inside the box. g(s) = s^2/4 and k(d) = -d^2/4
   give (s^2 - d^2)/4 = x0*x1, which explains the two-direction structure.
3. Checked `x0 * x1`: PASS (snap) on the first try.

## What I removed

All 16 sigmoid units, their weights and the output bias (-0.0056, which is noise). PASS (exact), with atol
1e-5, would need essentially the whole sigmoid net back, so the formula is the useful version.

## Heuristics

- **Group hidden units by weight direction first.** Normalise each unit's (w0, w1) and cluster. If the units
  fall on only a few directions (here x0+x1 and x0-x1), the net is a sum of 1-D ridge functions along those
  directions. Recognise each ridge separately.
- **Directions x0+x1 and x0-x1 with opposite-sign quadratic ridges mean a product:**
  x*y = ((x+y)^2 - (x-y)^2)/4. Sigmoid/ReLU nets have no multiply, so this polarization identity is how they
  build one. Expect it whenever the units sit on the two diagonals.
- **Tabulate on a small integer/half-integer grid before fitting anything.** A 9x9 table often shows the
  formula directly (signs, zeros on axes, symmetry, values that are products or sums of the labels).
- **Expect errors at the box corners.** Sigmoid ridge sums saturate at the edges of the training box, so the
  net undershoots a convex/growing target there. If the residual is largest at the corners and small inside,
  that points to the clean formula, not a correction term.
- **Drop the output bias when it is about 0.1% of the output range** and the formula is zero where the bias
  would matter (here f(0, 0) = 0).
- **Don't chase PASS (exact) for a smooth regression net with many sigmoids.** With a tight atol, exactness
  requires every unit. A clean formula within snap tolerance is the better result.
