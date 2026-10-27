# Notes

## What the function does
One float input x0 in [-2, 2], binary output (out > 0). Sorting the samples by x0 showed the raw
score is a clipped "tent" (a trapezoid):

- left flank: out = 25.68 + 50.58*x0 (exact linear fit, residual < 1e-5), zero crossing at x0 = -0.508
- plateau:    out = 5.522 (constant) for about -0.40 < x0 < 0.39
- right flank: out = 24.90 - 50.16*x0 (exact linear fit), zero crossing at x0 = 0.496

So score = min(5.522, 25.68 + 50.58*x0, 24.90 - 50.16*x0), and the decision is "x0 is inside a
window around 0". The two crossings (-0.508, +0.496) are symmetric up to noise, so they share one
threshold, snapped to 0.5.

## Final program (agent.py)
`inside = abs(x0) < 0.5` -> PASS (snap). The only disagreements are in the narrow band
|x0| in [0.496, 0.508], all inside the function's unsure zone.

A variant with the raw fitted coefficients (agent_exact_try.py) gives PASS (exact), but it is just the
unsnapped network weights; the snapped window is the readable answer.

## Heuristics
- With a 1-D input, sort the samples by x and print the score along with class changes; the shape
  (plateau, linear flanks, kinks) is visible directly.
- Fit lines per segment; a residual of ~1e-5 confirms a pure ReLU piece. A constant segment means a
  min/max clip (saturated ReLU).
- A binary decision only depends on the zero crossings of the score, not on slopes or plateau height:
  find the crossings, then snap them.
- Two crossings symmetric around 0 (here -0.508 and +0.496) -> one shared threshold with abs():
  `abs(x) < t`, t snapped to a simple fraction (0.5).
- Prefer the readable snapped rule when it passes in snap mode; raw fitted coefficients give exact
  agreement but hide the meaning.
