# Notes

Function: a triangular bump (hat function) centered at x0 = 3 with half-width 1 and height 1:
`out = relu(1 - |x0 - 3|)`. It is 0 outside [2, 4], rises linearly from 0 at 2 to 1 at 3, and
falls back to 0 at 4. Typical use: a "near 3" fuzzy membership / tent indicator.

How found: the first few samples already showed `out = 1 - |x0 - 3|` exactly (e.g. x0=3.0957 ->
0.9043, x0=2.349 -> 0.349) and 0 far away. Verified over all 2000 samples (max error 1.2e-7, i.e.
float32 rounding), then `check.py` gave PASS (exact).

## Heuristics

- For a 1-D regression, first look at a handful of sorted samples: zero regions plus linear pieces
  with slope +-1 suggest a relu/abs composition; read off the kink locations and snap them to
  integers.
- A piece where out + x or out - x is constant identifies slope +-1 and its intercept directly.
- A symmetric rise/fall around a point c with zeros outside is `relu(h - |x - c|)` (hat/tent);
  check height at the center and the zero crossings to get h and c.
- Verify the closed form against every sample (max abs error at float32 rounding level ~1e-7 means
  exact) before running the checker.
