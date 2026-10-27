# Notes

## What the network does
1 input -> 4 ReLU -> 1 binary output. It's a "small magnitude" detector: class 1 iff |x0| < 1/2.

## What I changed and why
- **Dead neurons removed.** h1_0 = relu(0.426*x0 - 1.030) only fires for x0 > 2.42, and
  h1_2 = relu(-0.060*x0 - 0.336) only fires for x0 < -5.59. Neither can fire on the domain [-2, 2],
  so both contribute 0 and were deleted.
- **Mirror pair recognised.** h1_1 = relu(-8.03*x0 - 3.20) and h1_3 = relu(8.00*x0 - 3.09) have
  nearly opposite slopes and similar biases; both have output weight about -6.3. Together they are
  about -6.3*relu(8*|x0| - 3.15): a symmetric "V" penalty around 0 subtracted from the bias +5.52.
- **Decision boundaries solved.** With output = 5.52 - 6.3*(8|x0| - 3.15) on the active side, the
  zero crossings are x0 = -0.5077 (left) and x0 = +0.4965 (right). Both snap to +/-1/2, one shared
  threshold.
- **Simplified to a formula.** Only the sign of the output matters, so the whole net is
  `0.5 - abs(x0)` (positive -> class 1). The scale is irrelevant for a binary output.
- Result: `PASS (snap)`. All disagreements are within ~0.008 of +/-0.5 (the net's unsure zone).
  Exact agreement would require keeping the asymmetric unsnapped thresholds (-0.5077, 0.4965),
  which is training noise rather than meaning, so I kept the snapped version.

## Heuristics
- First compute each hidden unit's activation threshold (-b/w for 1-D inputs, or the bias range
  over the input box) and delete units that are never (or always, then linearise) active on the
  domain. Small output weights (|w| << other weights) are another hint of a vestigial unit.
- Look for pairs of units with opposite-sign slopes of similar magnitude and similar biases and
  equal output weights: relu(a*x - b) + relu(-a*x - b) = relu(a*|x| - b) when b >= 0, i.e. an
  abs-value feature.
- For binary outputs, don't snap individual weights first; solve for where the output crosses the
  threshold, then snap those crossing points (here +/-0.5), and rewrite as a simple signed margin
  (e.g. `c - abs(x)`). Overall scale of the output is irrelevant.
- When two boundaries are nearly symmetric (-0.508 vs 0.496), merge them into one shared threshold
  and use abs().
- Ignore large magnitudes inside units (slopes of 8): steep ReLUs are just the network's way to
  make a sharp step; what matters is where the step happens.
