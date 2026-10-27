# Notes

## What I changed and why

- Input domain is x0 in [0, 200]. Neurons 0, 2, 3, 4, 6 have a negative slope and a negative
  bias, so their pre-activation is < 0 for every x0 >= 0: they are always 0 and were removed.
- The three live neurons are ramps `w*relu(a*x0 + b)`. Rewrote each as `(w*a) * relu(x0 - k)`
  with kink `k = -b/a`: kinks 2.858, 21.554, 79.890; slopes 0.1220, 0.0769, 0.2010.
- Cumulative slopes are 0.122, 0.199, 0.400 -> snapped to 1/8, 1/5, 2/5 (increments 0.125,
  0.075, 0.2). Kinks snapped to 3, 21.5 (half-step), 80. Output bias 0.066 snapped to 0
  (it is ~0.1% of the 62-unit output range).
- Result: `0.125*relu(x0-3) + 0.075*relu(x0-21.5) + 0.2*relu(x0-80)`, a convex
  progressive-rate (tax-bracket-like) ramp. Max error 0.084 vs snap tolerance 1.24 -> PASS (snap).
  PASS (exact) would require keeping the raw constants, which defeats the purpose.

## Heuristics

- Use the input bounds to find dead ReLUs: if sign(a*x+b) is fixed over the whole domain the
  unit is either always 0 (drop) or always linear (fold into the output affine part).
- In 1-D, normalise each unit to `slope * relu(x - kink)`; kinks = -b/a, slopes = w_out*a.
  The function is then a piecewise-linear spline; read it as segments.
- Snap cumulative slopes (the slope of each segment) to simple fractions rather than snapping
  the individual increments; the segment slopes are what the original rule most likely used.
- Snap kinks to integers or half-steps; small output biases relative to the output range snap to 0.
- Check the snap tolerance (fraction of output range) first to know how aggressive snapping can be,
  and verify numerically on a dense grid before running the checker.
