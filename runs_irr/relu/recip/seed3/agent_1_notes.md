# Notes

## What the network computes

A 1 -> 8 -> 1 ReLU net, regression on x0 in [0.5, 4]. The output falls from 1.914 at x0 = 0.5
to 0.240 at x0 = 4, convex and decreasing, so it looks like a coarse piecewise-linear fit of
something like 1/x (it is not 1/x itself: at x0 = 0.971 the net gives 0.975, 1/x gives 1.030).

## What I changed

1. **Dead units removed.** For each unit I computed the knot -b/w and checked whether it is ever
   active on [0.5, 4]. Units 0 (active only for x < 0.19), 2 (x < -0.23), 4 (never: w ~ 0, b < 0) and
   5 (x < -0.22) are always zero on the box, so they go.
2. **Folded each remaining unit into a knot form.** `c * relu(w*x + b)` = `c*|w| * relu(±(x - k))` with
   k = -b/w. This gives four terms with knots 0.5009, 0.714, 0.971, 1.871 and folded coefficients.
3. **Rounded the folded constants** to 5-6 significant digits, still `PASS (exact)`
   (max |program - net| = 6e-6).
4. Named the hinges. `agent.py` is the exact version.

The `relu(x - 0.5009)` unit is linear on almost all of the box (its knot is 0.0009 above the lower
bound). Replacing it with a plain linear term costs about 4e-4, which is within snap tolerance but
breaks exactness, so I kept the relu.

## Snapped alternative (PASS snap, not submitted)

The snap tolerance is 0.0335 (2% of the output range). A minimax LP fit with nice knots showed:
- 1 knot: best error 0.11 (fails);
- 2 knots (1, 2): error 0.029 after the fit and 0.0295 after snapping (passes, but with little margin);
- 3 knots (0.7, 1, 1.875): error 0.0197 after snapping:

      return 0.7 - 0.11*x + 0.3*relu(0.7 - x) + 1.3*relu(1 - x) + 0.4*relu(1.875 - x)

  `PASS (snap)`, max error 0.0197 (the worst point is at x0 = 4). It is in `scratch/agent_snap.py`.
I submitted the exact version because exact is ranked higher. It is about as short (4 hinges and a
constant against 3 hinges, a line and a constant).

## Heuristics

- **Dead-unit test on a box:** for 1-D (or per-input) affine units, compute the knot -b/w and the
  sign of w. If the active side does not meet the input interval, drop the unit. With w ~ 0, check
  only the sign of the bias. Here half the hidden layer was dead.
- **Fold `c*relu(w*x+b)` into `(c*|w|) * relu(±(x - k))`.** This leaves one coefficient per hinge,
  with the knot shown directly. It makes the piecewise-linear structure (knots, slope changes)
  readable and exact.
- **A knot just inside the box edge makes a unit almost linear.** It can be swapped for a linear
  term within snap tolerance, but the swap breaks exactness. Keep it when aiming for exact.
- **Read the segment slopes.** The slope changes at each knot give the convexity and monotonicity
  at a glance, and they suggest the target family (here decreasing and convex, 1/x-like).
- **To decide whether snapping is safe, use a minimax LP over a dense grid on candidate nice knot
  sets** before snapping coefficients. Least squares underestimates how few knots will do, because
  its max error is larger than the minimax optimum.
- **Snap the constant last,** at the midpoint of the residual range, and round it to a grid
  (0.05 here). The constant is usually what sets the final max error.
- **Exact rounding:** folded coefficients survive rounding to about 5-6 significant digits within
  atol 1e-5 / rtol 1e-4. Check this rather than assuming it.
