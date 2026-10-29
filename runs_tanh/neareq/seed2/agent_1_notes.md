# Notes

## What the network computes

Class 1 iff `|x1 - x0| < 0.2`: the two inputs are nearly equal. `agent.py` returns `0.2 - abs(x1 - x0)`.
It passes `check.py` as `PASS (snap)`: 0 disagreements where the net is confident, and 5784 inside the
net's unsure zone (about 1% of inputs, all within about 0.01 of the decision line).

## How I got there

- Grouped the 8 tanh units by weight size:
  - Units 4, 5 and 6 have large, nearly opposite weights on x0 and x1 (about ±6.76, ±3.67 and ±5.76).
    Each is a soft step in `d = x1 - x0`, with switch points at d = -0.19, +0.26 and +0.22.
    Their output weights (-11.7, -4.5, -10.5) together make a bump: about +22 near d = 0 and about ±3 far away.
  - Units 0, 1, 2, 3 and 7 have small input weights (< 0.31) and biases around ±2.1, so they stay near
    saturation (tanh ≈ ±0.97) over the whole domain. Together they act as a nearly constant offset
    with a slight tilt, which pushes the far tails below 0.
- Scanned the decision boundary along lines with fixed x0. The crossings sit at d ≈ -0.207..-0.200
  and d ≈ +0.199..+0.211, drifting slightly (about 0.004 per unit of x0) because of the slow units.
  That drift is noise, so I snapped both edges to ±0.2, giving one shared threshold on |d|.
- `PASS (exact)` would require reproducing the net's slightly tilted, slightly curved boundary to
  float precision, which means keeping essentially all 8 units. I did not keep them, because the
  snapped formula is the meaningful one.

## Heuristics

- For tanh/sigmoid nets, sort units by input-weight size. Units with small input weights and large
  |bias| saturate across the domain, so treat them as constants plus a small tilt, not as features.
- When a unit's input weights are nearly equal and opposite (w, -w), it is a function of the
  difference x1 - x0. Several such units with switch points on both sides of 0 and opposite-signed output
  weights form a band or window detector: `|x1 - x0| < c`.
- For binary outputs, scan the sign change of the dump along a few 1-D slices. The crossings give the
  thresholds directly. Then snap them to a round shared value (here ±0.2) and write the score as
  `c - abs(...)`, so the sign carries the decision.
- Expect a few disagreements in the unsure zone after snapping. They are fine for snap, but exact needs the full net.
