# Notes

## What the network computes
1 float input x0 in [0, 200], regression output from 0 to about 61, 8 sigmoid hidden units.
The slope of the net gives the shape: about 0.10 near 0, about 0.2 over 30-60, about 0.4 over
90-180, and a bend down to about 0.16 at the very top end (x0 > 190).

So it is a continuous piecewise-linear "progressive rate" function (like tax brackets):
marginal rate 0.1 up to 20, 0.2 from 20 to 80, 0.4 above 80.

## What I changed
- Replaced the 8 sigmoids with three ReLU terms: `0.1*x0 + 0.1*relu(x0-20) + 0.2*relu(x0-80) - 0.2`.
- Rates snapped to 0.1 / 0.2 / 0.4 (each one doubles); knots snapped to 20 and 80. Many knot pairs
  fit about equally well (the max error is set by the top end), so I chose round ones.
- The constant -0.2 is a centring offset. Without it the error at x0=200 is too large (1.34 > 1.21).
- The bend after 190 looks like the sigmoids saturating at the domain edge, so I left it out.
  Adding `-0.2*relu(x0-190)` cuts the max error to about 0.99, at the cost of one more term.
- Check: PASS (snap), max |program - net| = 1.188 against a tolerance of 1.215 (2% of range).
  The worst point is near x0 = 181, where the sigmoid net overshoots and then bends down.

## Heuristics
- For a 1-input regression net, sample the net's numeric derivative on a grid first. Plateaus in the
  slope are linear segments; the jumps between them are ReLU knots. With sigmoid hidden units, each
  unit is a soft step in slope. Center = -b/w, and the slope change is about amp*w/4 per unit.
- Fit free-knot continuous piecewise-linear models (minimise the max error) with 2, 3, 4 knots.
  Then snap the slopes to a clean pattern (0.1/0.2/0.4) and grid-search round knots plus a constant
  offset against the max error.
- Sigmoid nets smooth corners and droop at the domain edges. The worst error is usually at a
  smoothed corner or at the boundary, so centre the program with a small constant offset (the
  midpoint of min and max residual) rather than forcing f(0)=0.
- If several knot pairs give the same max error, the knots are not identified by the net. Pick the
  roundest values.
- A final slope change right at the domain edge (here at 190 of 200) is likely edge saturation, not
  part of the function. Keep it only if the check fails without it.
