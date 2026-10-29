# Notes

## What the network computes

The dump is 1 input -> 8 sigmoid units -> 1 linear output on x0 in [0, 200]. Each sigmoid is a soft
step at centre c = -b/w (36, 90, 101, 128, 150, 179, plus one at -18 that is almost constant on
the domain and one at 4.8 with a negligible weight of 0.08). The sum of these steps is a smooth
increasing curve, not a staircase, so I tabulated the output and its derivative instead of
reading the units one by one.

The derivative shows three flat plateaus: about 0.1 on [0, 15], about 0.2 on [35, 70], and about 0.4
on [100, 180], with smooth transitions in between. Then it falls towards 0.14 at x0 = 200. So the
function is a convex piecewise-linear curve with rates 0.1 / 0.2 / 0.4, like progressive tax brackets:

- 0.1*x0 on [0, 20]
- 0.2*x0 - 2 on [20, 80] (continuous at 20: both give 2)
- 0.4*x0 - 18 above 80 (continuous at 80: both give 14)

Lines fitted on the plateaus gave intercepts of about 0, -2 and -17.9, so I snapped them to 0, -2 and -18.
The knots then fall at round values, 20 and 80. Written as `max(...)` of the three lines, it needs
no branches, because the curve is convex.

Without a cap the error at x0 = 200 is 1.57, which is above the 1.21 snap tolerance (2% of the
output range of about 60.4). The net flattens near the top of the domain, so I added `min(..., 60)`.
The top line hits 60 at x0 = 195. This brings the max error down to 0.70, at x0 = 182, where the soft
sigmoid corner rounds the kink.

## Result

`agent.py`: `min(max(0.1*x0, 0.2*x0 - 2, 0.4*x0 - 18), 60)`. Check: PASS (snap), max |diff| 0.704,
tolerance 1.209. PASS (exact) is not possible: a sum of sigmoids never has true kinks or plateaus.

## Heuristics

- For 1-input sigmoid regression nets, tabulate the output and its numerical derivative across the
  domain. Plateaus in the derivative are the linear pieces. Their slopes are often simple
  numbers (0.1, 0.2, 0.4).
- A sigmoid's centre is -b/w and its width is about 4/|w|. A decreasing sigmoid with a negative
  output weight is an increasing step plus a constant. Units centred well outside the domain are
  constants, or near-linear pieces at the domain edge. Units with tiny output weights can be dropped.
- When the derivative rises step by step, the curve is convex: write it as `max` of lines and no
  branches are needed. When it falls step by step, use `min`.
- Fit each line on its plateau and snap the intercept. Then check that adjacent lines meet at round
  knots (here 20 and 80). Continuity at round knots is good evidence that the snap is right.
- A slope that fades at the edge of the domain often hides a cap or saturation. Try `min(..., round
  value)` before trying extra segments.
- Measure the error against the snap tolerance (2% of the output range) before and after each
  simplification. The worst error usually sits at a soft corner, which a sigmoid net can only
  approximate.
