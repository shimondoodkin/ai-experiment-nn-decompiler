# Notes

## What I did
- Spec: two float inputs in [-3, 3], one regression output.
- The first sample (x0=0.82, x1=2.86, out=2.04) suggested `out = |x0 - x1|`.
- Checked that against all 2000 samples. Away from the diagonal (|x0-x1| > 0.05) the error is at most about 1e-4.
  Near the diagonal the function bottoms out at a floor of about 0.0205 instead of reaching 0, so the
  max error there is about 0.02.
- That floor is a small artifact of the trained net (the ReLU pair that forms |.| leaves a little
  bias), not real behavior. So I snapped it to 0 and kept the clean formula
  `distance = abs(x0 - x1)`.
- `python check.py agent.py` gives `PASS (snap)`: max diff 0.0205, snap tolerance 0.12, and every
  difference is in the unsure zone along x0 == x1. I did not chase `PASS (exact)`, because that would
  mean encoding the 0.0205 floor (e.g. `max(abs(x0 - x1), 0.0205)`), which is not a meaningful
  constant.

## Heuristics
- Before fitting anything, test a few obvious closed forms (sum, difference, |difference|, min, max,
  product) on a handful of rows by eye.
- Sort samples by residual. If the largest residuals all sit on one structure (here the diagonal
  x0 == x1) and share one constant output value, that is a floor or kink artifact of the net, not
  separate logic.
- Look at the residual away from those structures: if it is ~1e-4 everywhere else, the formula is right
  and the rest is training noise.
- A floor or offset that is small compared to the output range (here 0.02 against a range of 0 to 6)
  should snap to 0. Abs, min and max built from ReLUs often leave such floors at the kink.
- Prefer the readable snapped formula over chasing exactness with artifact constants.
