# Agent notes

## What the network does

One float input x0 in [-2, 2], regression output, 8 tanh units.

- Sampling the net at the integers gave 0.126, 0.370, 0.996, 2.722, 7.348 for x0 = -2..2,
  which is e^-2, e^-1, e^0, e^1, e^2 to within about 0.04.
- `agent.py` is `return exp(x0)`. Its largest gap from the net is 0.041 (at x0 = 2, where the net
  undershoots the exponential's steep end). The snap tolerance is 0.144, so it passes as `PASS (snap)`.

## How the net builds it

- Five units (1, 2, 3, 4, 6) have slopes of only about ±0.055 and biases of ±1.3..1.8. Their
  arguments move by less than 0.12 over the whole box, so each one is a near-constant plus a
  near-linear term. Together they amount to 6.077 + 0.070·x0 − 0.0034·x0² (max error 3e-4). That is the
  "constant" part of the net: it puts the curve's offset at about 1 near x0 = 0.
- Three units are real bends: tanh(1.67x0 − 3.49) (centre 2.08), tanh(0.98x0 − 1.43) (centre 1.45)
  and tanh(0.64x0 − 0.62) (centre 0.98). Each takes over further right with a larger weight
  (1.35, 1.57, 2.94), so the slope grows with x0. That is a convex, growing curve, an exponential
  pieced together from sigmoids.

An exact version (within 1e-5) would need all three bend units kept as they are, plus the
near-linear fold. That is no more readable than the dump, so I kept the formula.

## Heuristics

- **Sample at integer and half-integer inputs and compare against known functions** (exp, x², |x|,
  sqrt, log, sin) before analysing units. For 1-input regression, the values at the integers often
  name the function directly (here 0.37 / 1 / 2.72 / 7.39 = powers of e).
- **Units whose argument barely moves over the input box are polynomial terms.** If
  |slope|·(hi − lo)/2 is small (< about 0.2) and the bias is not near 0, tanh/sigmoid is almost linear
  there. Fold all such units into one constant + linear (+ small quadratic) term by least squares.
- **A run of tanh units with increasing centres and increasing output weights builds a convex,
  growing function.** Suspect exp (or a power) when the steep end of the box gets the largest weights.
- **The net undershoots at the edge of the box where the target is steepest.** Fitting the true formula
  leaves its largest error there. Compare that error with the snap tolerance (2% of the output range)
  rather than trying to match the net's edge error.
- **Prefer the closed form over a cleaned-up copy of the units** when the closed form passes snap and
  an exact match would need nearly every unit.
