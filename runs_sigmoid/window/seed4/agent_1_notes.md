# Notes

## What the net does
1 input x0 in [-2, 2], binary output, 4 sigmoid units.
- h1_0 (rising step at x0 = 6.548/13.332 = 0.491, weight -13.08) and h1_1 (rising step at
  x0 = -6.372/12.436 = -0.512, weight +7.21) form a bump: the score goes up near x0 = -0.5
  and comes down near x0 = +0.5.
- h1_2 and h1_3 are two near-duplicate falling steps (at x0 = -0.529 and -0.526, weights
  -3.93 and -4.46). Together they act as one step of weight -8.38 that pushes the score further
  down for x0 < -0.5. That only deepens the "class 0" region on the left and does not move the
  decision boundary much.
- Scanning the net's score on a fine grid gives exactly two sign changes, at x0 = -0.4968 and
  x0 = +0.4983. The score is positive (up to 6.77) between them and negative (down to -8.74)
  outside.

## What I changed
- Replaced the whole network with the band rule `class 1 iff |x0| < 1/2`, written as the
  margin `0.5 - abs(x0)` (positive inside the band).
- Snapped the two boundaries (-0.4968, +0.4983) to the symmetric half-step ±0.5.
- Result: PASS (snap). The only disagreements (0.97% of inputs) are in the 0.003-wide slivers
  between the net's boundaries and ±0.5, where the net's score is within the unsure zone.
  Keeping the raw thresholds would give an exact match, but it would only fit training noise.

## Heuristics
- For 1-input binary nets, skip reading the units one at a time: scan the score densely, find
  the sign changes, and snap each one. The units only matter for explaining the shape.
- Two zero crossings at roughly ±c usually mean an `abs(x) < c` (or `abs(x) > c`) band. Snap
  to a symmetric threshold and write it as the margin `c - abs(x)`, which is shorter than an
  if/else.
- Sigmoid units with almost the same step location (-b/w) and the same weight sign are one
  step split in two. Add their weights.
- A sigmoid step with a large weight that sits far from every zero crossing (only making
  confident regions more confident) does not affect the decision and can be dropped.
- The step location of sigmoid(w*x + b) is -b/w. Compare it with the score's actual zero
  crossings: the boundary falls where the weighted steps balance, not at any single unit's
  step.
- Before snapping a threshold, check that the net's score at the snapped value is inside the
  unsure zone (here about -0.13 and -0.08 against a 0.44 zone).
