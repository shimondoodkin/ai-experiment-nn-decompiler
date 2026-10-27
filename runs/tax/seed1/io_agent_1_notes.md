# Notes

## What I did
- One float input on [0, 200], regression output. Sorted the samples by x0 and printed the slope
  between neighbouring samples; the slope only changes at a few places, so the function is
  piecewise linear (a ReLU net).
- Least-squares line fits per segment (max residual about 1e-5, so the segments are truly linear):
  - x0 in [0, ~2]: a small dip, slope about -0.13 (three nearly identical pieces, kinks near 0.89 and 1.62),
    bottoming out at about -0.757 near x0 = 2.03.
  - x0 in [~2, 78]: out = 0.18475*x0 - 1.1369 (zero at x0 = 6.15).
  - x0 in [78, 196.4]: out = 0.40040*x0 - 17.964 (the lines meet at x0 = 78.03).
  - x0 in [196.4, 200]: slope 0.314, an edge effect near the top of the domain.
- Snapped: slope 0.185 with zero at 6, kink at 78, extra slope 0.215 above it (so the rate becomes 0.4),
  and a floor at -0.75 that stands in for the small dip near 0. The last bend at 196.4 and the
  three-piece dip are left out as artifacts of the training; both stay well inside the snap tolerance.
- `agent.py`: `max(0.185*(x0 - 6) + 0.215*relu(x0 - 78), -0.75)`. Result: PASS (snap), max
  deviation 0.31 (at x0 = 200, from the dropped top bend) against a tolerance of 1.25.
- `exact_try.py` (scratch file): the unsnapped piecewise fit, max deviation 3e-4. It still gets
  PASS (snap) only, because the tiny pieces around x0 = 2 are not fully pinned down.
- How to read it: a two-rate (tiered) linear charge. Nothing below about 6, then 0.185 per unit,
  and 0.4 per unit above 78.

## Heuristics
- For a 1-D input, sort by x and print the slopes between neighbouring samples where they change. A ReLU net
  shows up as flat slope runs, and the kink positions can be read off directly.
- Fit each run with least squares and put kinks where the neighbouring lines meet. That is more
  precise than the sample spacing.
- Write a piecewise-linear function as base line + c*relu(x - k) terms. Each kink becomes one readable
  "extra rate above k" term.
- Short pieces at the domain edges (here near 0 and above 196) are often artifacts of the training.
  Drop them or replace them with a simple floor/cap if the snap tolerance allows. Check the size of
  the deviation before deciding.
- Snap kink positions to integers (78.03 -> 78, zero crossing 6.15 -> 6) and slopes to round
  decimals. Then check that the extra error stays well below the snap tolerance, which the checker
  prints.
- Do not use min/max of all segment lines blindly. An edge segment line can win far from its own
  region (here the top segment took over near x0 = 0). Use relu terms or if-branches instead.
