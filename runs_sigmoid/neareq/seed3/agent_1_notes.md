# Agent notes

## Result
`agent.py`: `return 0.2 - abs(x1 - x0)`. Class 1 iff |x1 - x0| < 0.2. `check.py`: PASS (snap), with 0
disagreements where the net is confident. The 6419 disagreements are all in the unsure zone (0.99% of inputs).

## What I changed and why
- **Steep units:** h1_0, h1_3 and h1_5 have weights of about ±10..11 on x0 and x1, with opposite signs and
  equal size, so each one is a soft step in d = x1 - x0. Dividing the bias by the slope puts the steps at
  d ~ -0.21 (h1_0 and h1_3, together +19.5) and d ~ +0.20 (h1_5, -18.2). The two rising steps sit almost
  on top of each other, so I merged them.
- **Slow units:** h1_1, h1_2, h1_4, h1_6 and h1_7 have small weights and large biases (|b| ~ 3-3.8), so
  they stay saturated over the whole box. With the output bias they add up to about -10, varying only by
  ±0.17 (a least-squares linear fit gives -9.99 + 0.048 x0 - 0.038 x1). They are a constant offset.
- **Net shape:** the output is about -10 below the band, +9.5 inside it and -8.7 above it. That is an
  indicator of |d| < 0.2.
- **Measured boundaries:** I scanned the class boundary along d for 13 values of (x0+x1)/2. The lower edge
  is at d = -0.198..-0.203 and the upper edge at +0.198..+0.207. Both snap to ±0.2, a symmetric band.
- **Exact pass not reached:** keeping the three steep sigmoids and fitting the slow units with a quadratic
  still did not reach PASS (exact). It halved the unsure-zone disagreements, but the boundary shifts by
  ~0.005 across the box, and the slow units cause that drift. I kept the simple formula.

## Heuristics
- Look for hidden units whose input weights are large, equal in size and opposite in sign. They are
  steps in a difference (x1 - x0), placed at -bias/slope. Group units with the same direction and close
  thresholds into one step, and add their output weights.
- Units with small input weights and a large |bias| are saturated on the whole input box. Treat them as
  constants (sigmoid(b) ~ 1 or 0) and fold them into the output bias. Bound how much they vary over the
  box, to confirm they cannot move a decision.
- For binary outputs, add up the plateau values between the steps (below, between, above). A
  "+big then -big" pair with a negative base gives a band indicator: `t - abs(d - c)` when the band is
  symmetric about c.
- Measure the actual decision boundary numerically along the main direction, at several points across
  the orthogonal direction, and snap to the shared nice value (here ±0.2). Do not rely only on
  -bias/slope of each unit, because the summed steps and offsets move the crossing point.
- When the only gap to an exact pass is slow drift from near-saturated units, accept PASS (snap). A
  readable formula is worth more than matching the drift of the boundary.
