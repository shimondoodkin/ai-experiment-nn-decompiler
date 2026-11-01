# Notes

## What the network is

1 input (x0 in [0.1, 5]) -> 32 ReLU units -> 1 regression output.

- Of the 32 hidden units, 16 are never active on [0.1, 5] (negative slope and negative bias, so
  `w*x + b < 0` for all x > 0, or a tiny positive slope whose knot lies past 5). They contribute
  nothing and were dropped.
- Two units (7 and 22) are always active, so they are just a linear term.
- The remaining 14 units put knots at about 0.100, 0.103, 0.201, 0.204, 0.368, 0.637, 0.650,
  0.670, 1.037, 1.620, 2.432, 2.852, 2.981, 3.528. The function is therefore piecewise linear,
  increasing, and concave (slopes go from ~7 near 0.1 down to ~0.23 near 5), with knots dense near
  the left end: the shape of a logarithm.
- Sampled values: net(1) = 0.0068, net(2) = 0.685 (ln 2 = 0.693), net(5) = 1.614
  (ln 5 = 1.609), net(0.1) = -2.269 (ln 0.1 = -2.303). So the net is a PWL fit of `ln(x0)`.

## What I changed

Replaced the whole network by `return log(x0)`.

- Max |log(x0) - net| over the domain = 0.0335, at the left end x0 = 0.1, where the fit's
  first linear piece cannot follow the steep curvature of log. The snap tolerance is 2% of the
  output range (0.0777), so the check is `PASS (snap)` with no input beyond the tolerance.
- An `exact` pass would need the 14-knot piecewise-linear program itself; I preferred the formula
  the knots approximate, since it is one node and is what the network is trying to compute.

## Heuristics

- For a 1-input ReLU net, first classify each hidden unit on the input box: never active (drop),
  always active (fold into a linear term), or a knot inside the box (keep). Here half the units
  were dead.
- Sample the function on a fine grid and look at the values at recognisable points (x = 1, 2,
  the bounds). net(1) ~ 0 together with a concave increasing shape and knots crowded near the
  small end of a positive domain strongly suggests `log(x)`; compare `log`, `sqrt`, `1/x`, `x^2`,
  `exp` candidates by their max error against the net.
- Knot density follows curvature: knots crowd where |f''| is large. For log that is near the low
  bound; for exp near the high bound; for 1/x near 0.
- Compare the candidate's max error to the snap tolerance (2% of the output range) before
  bothering with an exact piecewise-linear rewrite; the largest errors of a PWL fit to a smooth
  function sit at the domain edges, where the fit has the fewest constraints.
- Positive-domain lower bounds like 0.1 (instead of 0) are a hint that the target is undefined
  or singular at 0 (log, 1/x).
