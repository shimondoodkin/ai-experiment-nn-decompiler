# Notes

## What the network computes

1 input x0 in [0, 4], 8 ReLU units, regression output. Each unit is `relu(w*x0 + b)` with its kink at
x = -b/w. Only three kinks fall inside [0, 4]:

- unit 7 turns on at x = 0.4057 (slope change +0.022, tiny);
- unit 4 turns off at x = 0.4083 (slope change -0.763);
- unit 5 turns on at x = 1.6495 (slope change -0.207).

Units 1 and 6 are dead on the whole domain (their kinks are at negative x and they are active only
for x below them). Units 0, 2 and 3 are always active, so they are just linear terms that fold into
the starting line.

Exact rewrite (PASS exact, max error 3.7e-7, kept in `scratch/exact.py`):

    0.173553 + 1.248363*x + 0.022361*relu(x - 0.405677) - 0.762986*relu(x - 0.408346) - 0.207385*relu(x - 1.649524)

## Simplification in agent.py

- The two kinks at 0.4057 and 0.4083 are 0.003 apart: merged into one kink at 0.4 with the net slope
  change (-0.74 -> -0.75).
- Slopes snapped: 1.248 -> 1.25, 1.248-0.741=0.508 -> 0.5, 0.300 -> 0.3 (so the changes are -0.75 and -0.2).
- Intercept 0.1736 -> 0.175, kink 1.6495 -> 1.65.

Result: `0.175 + 1.25*x - 0.75*relu(x - 0.4) - 0.2*relu(x - 1.65)`, a concave
piecewise-linear curve that rises from 0.17 to 2.02 with decreasing slope. PASS (snap), max error
0.0145 (tolerance 0.037), largest at x = 4. The shape (concave, steep near 0, about 2 at x = 4,
about 1 at x = 1) looks like a piecewise-linear approximation of sqrt(x), but sqrt(x) itself is off by
0.17 at x = 0 and sqrt(x + 0.03) by 0.064, so neither passes; the net's version is kept.

## Heuristics

- For a 1-input ReLU net, compute every unit's kink -b/w first. Units whose kink is outside the
  domain are either dead (drop them) or always active (fold w*v and b*v into one linear term).
- Rewrite the net as `c + s*x + sum d_k * relu(x - k)` with d_k = v*|w| signed by the side the unit is
  active on (a unit that switches off as x grows contributes -v*w at its kink). This form is exact
  and makes the structure obvious.
- Kinks closer together than about 1% of the domain can be merged into one kink with the summed
  slope change; the error is |d| * gap, tiny for small slope changes.
- Snap segment slopes (not the raw slope changes) to simple values, then derive the slope changes
  from them; snap the intercept last and check the worst error at the domain ends.
- Compute the snap tolerance (2% of output range) before choosing how coarse to snap; here it allowed
  rounding kinks and slopes to 0.05 steps.
- Test whether the curve is a known function (sqrt, log, x^2) by comparing at the ends and a few
  points; if it fails by more than the tolerance, keep the piecewise-linear form.
