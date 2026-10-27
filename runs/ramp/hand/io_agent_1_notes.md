# Notes

The samples show `out = 0` for every x0 < 0, `out = x0` (to float32 precision, ~3e-8) for
0 <= x0 <= 1, and `out = 1` for every x0 > 1. So the function is a clamp of x0 to [0, 1]
(a saturating identity / "hard sigmoid"). Written as `min(max(x0, 0), 1)`; the knees snap
exactly to 0 and 1. `check.py` reports PASS (exact), max error 2.98e-08 (float32 rounding).

## Heuristics

- For a 1-D input, sort the samples and look at the output around candidate breakpoints;
  flat runs plus a slope-1 segment point straight at clamp / relu forms.
- Test the simplest guess right away by computing the max residual over all samples. A
  residual near float32 epsilon (~1e-7 relative) means the formula is exact, and the network
  just runs in float32.
- Constant output regions at round values (0, 1) suggest saturation. Write them as
  `min(max(x, lo), hi)` instead of piecewise if/else.
- Put the breakpoints on round numbers (0, 1) when the data near the knees allows it.
