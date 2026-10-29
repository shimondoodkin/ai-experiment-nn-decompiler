# Notes

## What the net does
1 input x0 in [0, 200], 8 sigmoid units, regression output (range about 0 .. 60.7).
Sampling the net densely and looking at its numerical slope showed a smooth piecewise-linear shape:

- slope ~0 at x0=0 rising to 0.2 by x0~30 (a smoothed kink near x0=10; y(0)~0);
- slope 0.2 on ~30..65 (y ~ 0.2*x0 - 2);
- slope 0.4 on ~95..175 (y ~ 0.4*x0 - 18); the two lines meet at x0 = 80;
- slope falls back to ~0.2 near the top of the domain (190..200).

So the net is a "bracket schedule": 0.2 per unit above 10, plus another 0.2 per unit above 80,
with the extra rate switched off above ~190 (probably a sigmoid-saturation edge effect, but it
is needed to stay within tolerance at x0=200).

## What I changed
- Replaced the 8 sigmoid units with three ReLU hinges:
  `0.2*relu(x0-10) + 0.2*(relu(x0-80) - relu(x0-190))`.
- Unit-level observations (not kept as code): h1 (center at x0 = -299) is a near-constant 1
  over the domain; the other units are soft steps centred at 33, 93, 95, 120, 142, 174, 178;
  pairs of opposite-sign soft steps (e.g. 93/95, 174/178) build the slope changes, i.e. the
  sigmoids together approximate integrated steps (ramps).
- Knots snapped to multiples of 10, slopes to 0.2, intercept 0 (the knot at 10 absorbs it).
- Checked: `PASS (snap)`, max |program - net| = 0.905 (tolerance 1.214), worst at the
  smoothed kink x0=10. Dropping the 190 knot fails (error 1.34 at x0=200). Exact pass would
  require keeping the sigmoids (net is smooth, no simple exact form).

## Heuristics
- For a 1-input regression net, plot/sample the output and its numerical derivative first:
  plateaus in the slope reveal a piecewise-linear target; slope transitions mark knots.
- Sigmoid nets fit kinks by smooth ramps; write the line for each plateau (y = a*x + b),
  intersect consecutive lines to get the knot, then snap knots and slopes to round values.
- A unit whose centre (-b/w) lies far outside the input domain is a constant; fold it into the bias.
- Expect edge effects near domain bounds (saturation makes the slope sag); an extra hinge near
  the edge may be needed to stay within tolerance even if the "true" rule has no such knot.
- Do a small grid search over snapped candidates (knots on a 5/10 grid, slopes in multiples
  of 0.1/0.2) scoring max error against the net, and keep the simplest under the snap tolerance
  with margin.
- The worst error of a hinge approximation to a sigmoid net sits at the knot where the net's
  transition is widest (lowest |w| of the units involved); check those first.
