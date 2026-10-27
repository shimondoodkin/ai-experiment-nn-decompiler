# Notes

## What the function is
One float input x0 in [-2, 2], binary output (out > 0). Sorting the samples by x0 and looking at
finite-difference slopes shows a clipped "tent" (trapezoid):

- x0 < -0.42: rising line, slope 76.47, out = 76.469*x0 + 38.270 (zero at x0 = -0.5005)
- -0.42 < x0 < 0.38: flat plateau, out = 6.0861 (a ceiling / clamp)
- x0 > 0.38: falling line, slope -53.00, out = -53.003*x0 + 26.668 (zero at x0 = +0.5031)

So out = min(6.086, rise, fall) up to tiny rounding at the two plateau corners, and the decision is
positive only between the two zero crossings: "x0 is near zero", i.e. |x0| < 1/2.

## What I wrote
- `agent.py` (submitted): `abs(x0) < 0.5`. The crossings -0.5005 and +0.5031 were snapped to the
  symmetric half-step +-1/2, so one shared threshold for both sides. Check: PASS (snap); the only
  disagreements are 196 points in 0.5 < x0 < 0.5031, all inside the function's unsure zone.
- `agent_raw.py` (reference only): the unsnapped score `min(6.086123, 76.46928*x0 + 38.27016,
  -53.00346*x0 + 26.66798)`. Check: PASS (exact). It shows the trapezoid model is complete; the
  slopes 76.47 / 53.00 and the plateau 6.086 are training artefacts that do not affect the decision
  beyond where the line crosses zero, so they are not kept in the readable version.

## Heuristics
- For a 1-D input, sort by the input and print where the finite-difference slope changes; the
  breakpoints and linear pieces fall out directly (a few spurious slopes come from near-duplicate x).
- For a binary output only the zero crossings of the score matter; fit each linear piece with
  least squares, solve for its root, and snap the root (here -0.5005 and 0.5031 -> -0.5 and 0.5).
- If two crossings are roughly symmetric around 0, express them as one threshold on abs(x).
- Flat runs with an exactly repeated output value indicate a clamp (min/max) — a ceiling of a
  sum of ReLUs; the plateau value is irrelevant to the decision if it is on the positive side.
- Large, asymmetric slopes (76 vs 53) are just the steepness of the learned boundary; do not
  try to snap them when only the decision is required.
- Use check.py's "unsure zone" report to confirm that all remaining disagreements are confined to a
  thin sliver next to the snapped threshold (here 0.5..0.503) rather than spread out.
- Keep an unsnapped score version around to confirm the structural model is exact before snapping.
