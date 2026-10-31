# Notes

## What the network computes

One input x0 in [0.1, 5], regression. The 8 hidden ReLUs reduce to a concave piecewise-linear
curve with three pieces (it rises steeply, then more slowly: a log-like shape, but log(x) itself
misses the net by 0.29, far beyond the 0.0735 snap tolerance, so it is not a log).

## What I changed

1. **Dead units removed.** h1_1, h1_2 and h1_6 have negative slope and a bias <= ~0, so they are 0
   for every x0 >= 0.1 (h1_2's kink is at 2.3e-5).
2. **Always-on units merged.** h1_0 and h1_3 turn on at x0 ~ 0.098, below the domain's lower bound
   0.1, so they are plain lines; together with the output bias they give the base line
   0.76826 x0 - 0.87734.
3. **Kinks.** The remaining units are kinks at x0 = 0.5375 (h1_4, turned off above), 0.5434 (h1_7)
   and 1.8699 (h1_5). Writing each as `c * relu(x0 - k)` gives an exact program
   (max error 6.6e-7, `PASS (exact)`):
   ```
   y = 0.76826297*x0 - 0.87733797 - 2.7506135*relu(0.53748502 - x0)
       + 0.12135876*relu(x0 - 0.54339552) - 0.58574785*relu(x0 - 1.8699162)
   ```
4. **Snapped.** The kinks at 0.5375 and 0.5434 are only 0.006 apart, so I merged them into one. Then the
   slopes are decreasing (3.519, 0.890, 0.304), the curve is concave, and it is the minimum of its three
   segment lines: 3.5188 x0 - 2.3558, 0.8896 x0 - 0.9433, 0.3039 x0 + 0.1520. Rounding to
   `min(3.5x - 2.35, 0.9x - 0.95, 0.3x + 0.16)` gives max error 0.0125 (tolerance 0.0735),
   `PASS (snap)`. The lines meet at x0 = 0.538 and x0 = 1.85, close to the net's kinks.
   (Using +0.15 for the last line also passes, error 0.021.)

`agent.py` is the snapped min-of-three-lines version. The exact relu version is above and in
`scratch/exact.py`.

## Heuristics

- For 1-input nets, first compute each unit's kink k = -b/w and its sign of slope; any unit whose
  active side lies entirely outside [lo, hi] is either dead (drop it) or always on (fold it into a
  base line). Kinks just outside the domain (here 0.098 vs lo 0.1) count as always on.
- Combine all output weights times unit slopes per segment to get the explicit piecewise-linear
  function; list each segment's line (slope, intercept).
- If segment slopes decrease monotonically the function is concave and equals `min` of the segment
  lines (increasing: convex, `max`). This removes every relu and the knots become implicit.
- Merge kinks that are much closer than the domain width (here 0.006 out of 4.9); the tiny segment
  between them costs far less than the snap tolerance.
- Snap line coefficients to one or two decimals and check the max error against 2% of the output
  range; adjust an intercept by 0.01 if one segment carries most of the error.
- Before settling on piecewise lines, least-squares fit candidate closed forms (a*log(x)+b,
  a*sqrt(x)+b) and compare the max error with the tolerance; a log-like shape is not proof of a log.
