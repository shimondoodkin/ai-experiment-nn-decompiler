# Notes

## What I changed

- The net is 1 input -> 8 ReLU units -> 1 output, so it is a piecewise-linear curve in x0. I worked out
  where each unit switches on or off (x0 = -b/w) and compared that with the domain [-1.5, 4]:
  - units 0, 4 and 6 switch at x0 < -2.4: they are always off in the domain, so they were removed;
  - unit 5 switches at x0 = 5.01: always off, removed;
  - unit 1 switches at x0 = -3.56: always on, so it is just a line;
  - units 2, 7 and 3 give the real kinks, at x0 = -0.688, -0.677 and 0.964.
- The two kinks at -0.688 and -0.677 are 0.011 apart, so I merged them into one kink.
- Slopes per piece are 1.157, (0.347 on the short piece), 0.482 and 0.228, decreasing, so the curve
  is concave. A concave piecewise-linear curve is the minimum of its lines, so the program is
  `min(1.14 + 1.16*x0, 0.67 + 0.48*x0, 0.91 + 0.23*x0)`, with coefficients rounded to two decimals.
- Rounder slopes (0.5 and 0.25) fail: the error grows linearly to x0 = 4 and reaches 0.07, past the
  0.0485 snap tolerance. A single kink is not enough either (best error about 0.14).
- Result: PASS (snap), max |program - net| = 0.0041 (tolerance 0.0485).
- The shape looks like a log curve (roughly log(x0 + 2)), but the net is about 0.09 below that at
  x0 = -1.5, so I kept the net's own piecewise-linear shape rather than the formula.

## Heuristics

- For a 1-input ReLU net, compute every unit's kink x = -b/w first. Units whose kink lies outside the
  input range are either always off (drop them) or always on (fold them into one line).
- Kinks closer together than about 1% of the input range can be merged into one; refit the lines
  after merging.
- If the slopes decrease left to right, write the curve as min(line1, line2, ...); if they increase,
  as max(...). This is shorter and easier to read than a sum of relu terms.
- Fit the lines by least squares on a dense grid, then round coefficients and check the max error.
  Slope rounding costs error proportional to the distance from the kink, so the end of the domain is
  where a too-round slope fails; round slopes less than intercepts.
- A smooth-looking curve (here like a log) is tempting, but check the formula against the net, not
  the guess; the net's ends often differ from the true function by more than the snap tolerance.
