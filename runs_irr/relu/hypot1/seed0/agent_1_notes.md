# Notes

## What the network computes

A 2 -> 16 -> 1 ReLU regression net on [-2, 2]^2. Every output weight is positive, so the net is a
convex sum of hinges. Grid samples give f(0,0) = 1, f(+-1,+-1) = 1.73 = sqrt(3), f(2,0) = 2.24 = sqrt(5),
and f(2,2) = 3. That is `y = sqrt(1 + x0*x0 + x1*x1)`.

That formula on its own **fails** the checker. Its max error against the net is 0.055, and the snap
tolerance is 0.041. The net has local piecewise-linear dips, for example near (0.5, 0.19), where it sits
0.055 below the formula. Its residual against the formula runs from -0.055 to +0.042. That spread
(0.097) is wider than twice the tolerance, so no rescaled formula (a*sqrt(b + c*x0^2 + d*x1^2) + k,
minimax 0.055) and no formula plus a few of the net's own hinges (0.054) can pass. The submitted
program therefore keeps the hinge form. The formula is given in the header comment.

## What I changed

1. **Dead units removed (exact).** Units 0, 11 and 12 never become positive on the box: the maximum of
   w.x + b over the corners is below 0. Removing them leaves 13 units, which PASS (exact).
2. **Output weights folded in.** Every output weight v is > 0, so v*relu(z) = relu(v*z). Each hinge then
   enters the output with weight +1, and the output is `const + sum of hinges`.
3. **Greedy pruning with full refit.** I removed the unit whose removal hurt least, refit every
   parameter (least squares, then iteratively reweighted towards minimax), and repeated. Max errors on
   a 401^2 grid: 12 units 0.019, 11 units 0.027, 10 units 0.036, 9 units 0.053 (infeasible).
4. **Snapping.** Rounding to a 0.01 grid and nudging one coefficient at a time:
   - 12 and 11 units PASS (0.026 and 0.032);
   - 10 units on a 0.01 grid just fails (0.0415).

   On a 0.005 grid, 10 units PASS (snap) at max error 0.0368. That is the submitted `agent.py`.
5. **Naming.** Each hinge is named by the direction it rises towards (sw, west, north, south, east).
   Two pairs share a direction with only the crease moved: `sw_a`/`sw_b` and, roughly,
   `east_a`/`east_c`.

Result: `PASS (snap)`, max |program - net| = 0.0368 (tolerance 0.0409). The program has 10 hinges
instead of 16 units, and every coefficient is a multiple of 0.005.

## Heuristics

- **Find dead units first.** For a first-layer ReLU unit, evaluate w.x + b at the box corners. If the
  maximum is <= 0, the unit is dead and can be removed exactly.
- **Check the output weights' signs.** If they are all positive (on a single hidden layer), the net is
  convex. Fold each weight into its unit, relu(v*z) = v*relu(z), so the output is `const + sum(hinges)`.
  That removes a multiplier per unit.
- **Sample a coarse grid and look at the special points**: the origin, the axes and the corners. Values
  like 1, sqrt(3), sqrt(5), 3 point straight at `sqrt(1 + x0^2 + x1^2)`. Try the formula against the
  checker before anything else.
- **Compare the formula's residual spread with the tolerance.** When the formula fails, look at
  max - min of (net - formula). If it exceeds twice the snap tolerance, no affine rescaling of the
  formula can pass, and the net's own approximation dips must be kept. Go straight to a reduced
  piecewise-linear program. Do not spend time on formula variants.
- **Prune with a full refit, not output-only refit.** Linear refit of the output weights loses
  feasibility at 11-12 units. Refitting all parameters (least squares, then IRLS reweighting by |err|
  towards minimax) reaches 10 units.
- **Snap after the fit, then nudge.** Round to a grid (0.01, else 0.005). Then do coordinate descent of
  +-1 grid step on the max error, measured on a dense grid plus random points. Plain rounding of 30+
  coefficients adds about 0.015 of error; nudging recovers most of it.
- **Supply an analytic Jacobian.** A numeric Jacobian on a dense grid made the refit loop too slow.
  The ReLU Jacobian is just the mask times (x0, x1, 1).
