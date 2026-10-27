# Notes

## What the network does
Binary classifier on one input x0 in [-2, 2]: class 1 iff |x0| < 1/2 (a band around zero).

## What I changed and why
- `h1_0 = relu(-0.0075*x0 - 0.385)`: pre-activation is negative for every x0 in [-2, 2]
  (max is -0.37), so the neuron is always 0. Removed (dead neuron).
- The output is `6.086 - 6.87*relu(7.72*x0 - 3.00) - 4.83*relu(-6.0*x0 - 2.55) - 6.99*relu(-6.8*x0 - 2.85)`:
  a constant positive bias minus ramps that switch on to the right (h1_1) and to the left
  (h1_2, h1_3). The two left ramps have nearly the same kink (-0.4245 vs -0.4187) and just
  add up to one steeper ramp.
- Since only the sign matters, I solved for the zero crossings numerically:
  x0 = -0.5005 and x0 = 0.5031. Both snap to ±1/2, symmetric, so the decision is
  `0.5 - abs(x0) > 0`.
- Result: `PASS (snap)`; the only disagreements are in the thin slivers between ±0.5 and the
  network's true crossings (inside the unsure zone). `PASS (exact)` would require keeping the
  unsnapped thresholds -0.5005 / 0.5031, which carry no meaning.

## Heuristics
- First check each hidden neuron's pre-activation range over the input box; neurons that are
  never positive are dead and can be deleted (and always-positive ones are linear).
- For a binary output, the magnitude of the score is irrelevant: find the decision boundary
  (zero crossings of the piecewise-linear score) and re-express it directly, rather than
  snapping individual weights.
- In 1-D, a piecewise-linear net is fully described by its kinks; between kinks solve the
  linear equation for the zero crossing.
- Several ReLUs with nearly equal kink points and the same sign of slope act as one ramp; merge them.
- A positive bias minus ramps going both ways = an interval/band; if the two boundaries are
  roughly opposite, express it with abs() (|x - c| < r).
- Snap boundaries to the nearest simple value (here ±0.5, i.e. a half-step) and accept
  disagreements that fall only inside the network's low-confidence zone.
