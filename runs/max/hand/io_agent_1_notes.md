# Notes

The function is `max(x0, x1)` (an elementwise maximum of the two inputs).

What I did: the first sample had out == x1 with x1 > x0, so I checked a few candidate closed forms
(max, min, x1, |x0 - x1|) against all 2000 samples. `max(x0, x1)` matched every sample with max
error 3.6e-7, which is float32 rounding, so no coefficient snapping or region splitting was needed.
`check.py` reports `PASS (exact)` (max |program - net| = 3.6e-7 over 240405 inputs).

## Heuristics

- Look at a few raw rows first: if the output equals one of the inputs exactly, suspect a selector
  (max/min/relu/abs) rather than a weighted sum.
- Test a short list of cheap closed-form candidates (identity, max, min, abs difference, sum) against
  all samples and look at max error and fraction within tolerance before fitting anything.
- A residual around 1e-7 relative to values of order 1 is float32 noise from the network; treat it as
  exact and stop.
- max(a, b) is equivalent to a + relu(b - a); a 2-input net with a ReLU on a difference often encodes it.
