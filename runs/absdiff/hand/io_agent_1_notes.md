# Notes

## What I did
- Read `spec.json`: regression, two float inputs on [-3, 3], one output.
- Eyeballed the first rows of `samples.csv`: the output was always non-negative and looked like the gap
  between the two inputs (e.g. 0.82, 2.86 -> 2.04; -2.75, 2.51 -> 5.26).
- Tested the hypothesis `out = |x0 - x1|` on all 2000 samples: max residual 4.4e-7, mean ~3e-9, i.e.
  float32 rounding only. No offset, scale, or extra kinks were needed.
- `agent.py`: `distance = abs(x0 - x1)`. `check.py` reports `PASS (exact)`, max error 4.7e-7.

## Heuristics
- Before fitting anything, read a dozen rows and guess a familiar closed form (sum, difference, abs,
  min/max, clamp); test the guess against every sample and look at the residuals.
- A non-negative output that grows with the gap between two inputs suggests `abs(a - b)`; sign and
  symmetry checks (swap inputs, negate both) quickly confirm or rule this out.
- Residuals around 1e-7 relative to outputs of size ~1-5 are float32 noise; treat the formula as exact
  and stop rather than adding correction terms.
- Look at the largest residuals, not only the mean: they show where a missing kink, clamp or offset is.
- Keep the final program minimal and give the intermediate a meaningful name.
