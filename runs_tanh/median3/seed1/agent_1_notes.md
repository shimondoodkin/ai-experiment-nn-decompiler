# Notes: 3 inputs -> 16 -> 16 -> 1 tanh regression net

## What the net computes
The **median of x0, x1, x2**. Slices made this obvious: with two inputs fixed at -2 and 2, the
output followed the third input exactly between them and flattened to -2 / 2 outside. A least-squares
fit of the net against `median(x)` over 200k uniform points gave slope 0.997, intercept -0.009, RMS
error 0.032.

The median is written branch-free as `x0 + x1 + x2 - min(...) - max(...)`.

## Why pure median is not enough
`median` alone fails the snap check (max error 0.24 vs tolerance 0.117). The residual has two
systematic parts, both smoothing artefacts of tanh units:

1. **Rounded corners at ties.** Where the median is close to the largest input the net acts as a soft
   min and comes out low (about -0.14 at an exact tie); where it is close to the smallest input the
   net comes out high. Binning the residual by the gap to each neighbour showed a bump of width about
   0.3–0.4. Modelled as `0.4 * (relu(0.4 - gap_low)^2 - relu(0.4 - gap_high)^2)` (fitted coefficient
   0.395, width 0.4).
2. **Saturation at the box extremes.** When lo is near -3 and hi near +3 the net shrinks the median
   toward 0 by up to ~5%. Regression on candidate features (`med*spread`, `med^3`, `med*relu(spread-s)`)
   showed only `median * relu(spread - 5)` mattered (fitted 1/60). A constant offset of -0.01 (the
   mean residual) completes it.

Result: `PASS (snap)`, max error 0.112 against tolerance 0.117. Variations (no offset, no shrink,
width 0.3 or 0.5, shrink 1/40 or threshold 4.5) all failed, so all three corrections are needed for
this particular net. The core semantics is just the median; the rest describes the net's error, not
the task.

## Heuristics
- Before reading weights, probe a small grid of slices (fix all inputs but one at a few values,
  sweep the last). Order-statistic functions (min, max, median, clamp) show up as piecewise-linear
  slices with slope 0 or 1 whose breakpoints equal the other inputs' values.
- Fit `y` against the candidate formula (slope/intercept/RMS) before refining; slope ~1 and
  intercept ~0 confirm the hypothesis.
- Median of three in a min/max language: `a + b + c - min(a,b,c) - max(a,b,c)`.
- Smooth-activation (tanh/sigmoid) nets that approximate a piecewise-linear function deviate most at
  its kinks. Bin the residual by the distance to each kink; a soft-min/soft-max correction
  `c * relu(w - gap)^2` (signed by which side the kink bends) usually captures it.
- Second, check residuals against "how close to the input box corners" features (spread, |x| near
  the bound): tanh nets often saturate and shrink the output at the extremes of the training domain.
  A hinge on spread (`relu(spread - s)`) localises the fix to the corners.
- Use least squares to pick candidate features, then tune the few snapped constants against the
  checker's max error, since acceptance is a max-error criterion, not RMS.
- Add the mean residual as a constant offset when the tolerance margin is tight.
