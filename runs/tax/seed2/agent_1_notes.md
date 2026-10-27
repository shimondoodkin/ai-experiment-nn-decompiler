# Notes

## What the network computes
A 1 -> 8 -> 1 ReLU net on x0 in [0, 200] is a piecewise-linear function of one variable. I listed
each hidden unit's kink (x = -b/w) and whether it is active inside the domain, then summed
`w_out * w_in` per segment to get the slope on each interval.

- h1_1 (-0.238*x - 0.885) is never active for x >= 0: removed.
- h1_6 kinks at x = 208.7 > 200, so it is linear on the whole domain (folds into slope/offset).
- Kinks at 0.08, 0.55 and 0.94 (h1_5, h1_0, h1_4) only bend the curve slightly near 0 (output
  about -0.1 there): fitting noise around a zero start.
- Kinks at 20.4 and 21.6 (h1_3, h1_2) are one breakpoint: snapped to 20.
- Kink at 79.7 (h1_7) snapped to 80.

Segment slopes: about 0.12 on [1, 20], 0.20 on [21.6, 79.7], 0.40 above 80, which is a
progressive tier schedule with rates 0.1 / 0.2 / 0.4 (doubling) and breakpoints 20 and 80.
Values check out: net(20)=2.16 (2), net(80)=14.16 (14), net(200)=62.3 (62).

## Result
`f = 0.1*min(x,20) + 0.2*min(relu(x-20),60) + 0.4*relu(x-80)`
(equivalently `0.1*x + 0.1*relu(x-20) + 0.2*relu(x-80)`).
check.py: PASS (snap), max error 0.30 against a tolerance of 1.25 (largest at x=200, where the
net's top slope is 0.401 instead of 0.4).

## Heuristics
- For a 1-D input, turn the net into a list of kinks (-b/w per hidden unit) and segment slopes;
  that is the whole function.
- Drop units whose kink lies outside the input range on the inactive side (always zero), and fold
  units whose kink lies outside on the active side into the linear term.
- Merge kinks that are close together (20.4 and 21.6) into one round breakpoint; treat a cluster of
  kinks near the domain edge with tiny net effect as fitting noise around the boundary.
- Read slopes as rates: look for simple ratios between segments (here each doubles) and round to
  simple decimals.
- Check key points (breakpoints, ends) against round numbers to confirm the intercept (here 0).
- Write the result as tiers (min/relu clamps) so the business meaning (tiered pricing/tax) shows.
- The snap tolerance is a fraction of the output range; a small slope error times a long
  interval is where the largest deviation shows up (the domain end).
