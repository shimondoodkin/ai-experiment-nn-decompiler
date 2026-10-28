# Notes

## What the network does
2 inputs -> 8 sigmoid units -> 3 logits -> argmax. Mapping the argmax on a grid showed three
regions in diagonal bands: class 0 at small x0/x1, class 1 in the middle, class 2 at large x0/x1.
Both class boundaries are almost straight lines (bisected at 13 values of x1 each):

- 0|1 boundary: x0 = 3.97 - 2.14*x1 (crosses x0 = 1 near x1 = 1.39, so above that x1 there is no class 0)
- 1|2 boundary: x0 = 6.81 - 1.12*x1 (very slight curvature, < 0.03 in x0)

The unsure zone (top-two logit margin < 0.43) is only about +-0.07 wide in x0 around each
boundary, so the snapped lines had to stay inside that band over the whole x1 range.

## What I changed
Replaced the 8 sigmoids and 3 logit sums by two linear tests:

    if x0 + 2.2*x1 < 4:   class 0
    elif x0 + 1.1*x1 < 6.8: class 1
    else: class 2

Coefficients tried: slope 2 or 1 (too far from 2.14 / 1.12, drifts outside the band at the ends
of the x1 range), then 2.2 / 1.1 with round intercepts 4 / 6.8, which pass. 2.125 / 1.125 also
pass with fewer disagreements, but 2.2 and 1.1 are simpler (and 2.2 = 2 * 1.1).
The rule order is safe: whenever x0 + 2.2*x1 < 4, also x0 + 1.1*x1 < 4 < 6.8.
Result: PASS (snap), 0 disagreements where the net is confident. An exact version would need the
full sigmoid mixture, so I did not pursue it.

The input ranges (x0 in [1, 6.9], x1 in [0.1, 2.5]) and the three ordered classes look like a
measurement-based 3-class classification with two linear cuts in a 2-D feature plane.

## Heuristics
- For small argmax nets with 2 inputs, first print the argmax on a coarse grid; the region shape
  (bands, wedges, islands) tells you the program structure before reading any weights.
- Bisect each class boundary along one input at many values of the other and fit a line; if the
  residual is small, a single linear test per boundary replaces the whole hidden layer.
- Compute the unsure-zone band (where top-two logit margin < threshold) along each boundary in
  input units; it is the real budget for snapping. Choose the simplest coefficient whose line stays
  inside the band at both ends of the input range (ends are where slope errors show up).
- Write the boundary as `x0 + k*x1 < c` (normalising the coefficient of one input to 1) so k and c
  can be snapped independently to short decimals.
- For ordered classes, a cascade `if s1 < a: 0 elif s2 < b: 1 else: 2` replaces argmax; check that
  the first test implies the second so the cascade cannot skip a class inconsistently.
- Sigmoid nets give smooth, nearly linear boundaries when the units are spread out; their curvature
  is usually far below the unsure-zone width, so piecewise rules are not needed.
