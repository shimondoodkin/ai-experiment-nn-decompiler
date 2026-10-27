# Notes

## What the network computes
1 input (x0 in [0, 200]) -> 8 ReLU units -> 1 linear output. With one input every hidden unit is
just a hinge at x0 = -b/w, so the whole net is a 1-D piecewise-linear curve.

- Dead units: h1_1 and h1_6 have negative slope and negative bias, so they are 0 on the whole
  domain [0, 200]. Removed.
- Hinges of the remaining units: 0.89 (h1_5), 1.62 (h1_7), 2.00 (h1_0), 2.06 (h1_3), 78.0 (h1_2),
  196.4 (h1_4). The four hinges in 0.9..2.1 cluster around x0 = 2 and carry tiny slope changes
  except h1_0/h1_3, so they were merged into one kink at x0 = 2.
- Resulting segments (slope): x0<2: about -0.13; 2..78: 0.185; 78..196: 0.400; 196..200: 0.314.
- The curve is convex except for the last little kink at 196 (an edge artifact near the domain
  boundary, error 0.19 if ignored), so it is written as a max of three lines:
  `max(-0.5 - x0/8, 0.185*x0 - 1.12, 0.4*x0 - 18)`.
  Intercepts were snapped (-0.488 -> -0.5, -17.96 -> -18, slope -0.13 -> -1/8); the lines still
  cross at x0 = 2 and x0 ~ 78.

Result: `PASS (snap)`, max |error| = 0.19 against a snap tolerance of 1.25 (worst at x0 = 200,
where the dropped kink would have mattered). Exact pass would require keeping all raw
coefficients, which defeats the purpose.

## Heuristics
- For a 1-input net, compute each unit's hinge point -b/w and its contribution to the slope
  (output weight * input weight); the net is then a list of (kink, slope delta) pairs.
- Drop units that are inactive over the whole input range (sign of w and b both keep the
  pre-activation negative on [lo, hi]).
- Merge hinges that are close together into one kink; drop hinges whose slope delta is tiny.
- Hinges very close to the domain edge with small effect are usually fitting artifacts; check the
  error from removing them against the tolerance.
- If the slopes are increasing along the axis the function is convex: write it as max(line, ...)
  (decreasing slopes -> min). This removes all relu terms and makes intercepts readable.
- Snap slopes/intercepts to simple values (1/8, 0.4, integers) and re-verify; the check's snap
  tolerance (2% of output range here) tells how much freedom there is.
