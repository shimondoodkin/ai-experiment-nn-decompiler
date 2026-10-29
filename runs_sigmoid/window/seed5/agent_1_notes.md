# Notes

## What the network does
Four sigmoid units, each a steep step (slopes ~10-11.4) at a threshold of -bias/weight:
- h1_0: on for x0 < -0.491 (output weight -8.54)
- h1_3: on for x0 > -0.493 (output weight +9.06)
- h1_1: on for x0 > 0.504 (output weight -8.60)
- h1_2: on for x0 > 0.5045 (output weight -9.44)

Region values of the output score: x0 < -0.5 -> about -8.7 (class 0); -0.5 < x0 < 0.5 -> about +8.9
(class 1); x0 > 0.5 -> about -9.1 (class 0). h1_0 and h1_3 are complementary steps at the same point
(one up, one down), and h1_1/h1_2 are two near-duplicate steps at 0.5 that merge into one.
The exact decision boundaries (by a fine scan) are x0 = -0.4942 and x0 = 0.5020.

## What I changed
Snapped both boundaries to +-1/2 and wrote the band as one formula:

    return 0.5 - abs(x0)        # class 1 iff |x0| < 1/2

`check.py`: PASS (snap), 0 disagreements where the net is confident; all disagreements are in the
net's unsure zone near +-0.5. (A version with the raw thresholds,
`(x0 > -0.494201) and (x0 < 0.501953)`, gives PASS (exact), but its thresholds are just the
network's noise around 1/2, so I kept the snapped one.)

## Heuristics
- For a sigmoid unit with a large |weight|, treat it as a step at x = -bias/weight; the direction is
  the sign of the weight.
- Evaluate the output score region by region between the step points: it is piecewise constant, so a
  small table of region values gives the decision directly.
- Steps at almost the same point with the same direction merge (sum their output weights); a pair of
  opposite-direction steps at the same point with opposite-sign output weights is one "switch".
- For binary 1-input nets, find the actual zero crossings of the score with a dense scan, then snap
  them to simple values (here +-1/2). Symmetric boundaries -a and +a become `a - abs(x)`.
- Return a signed margin (like `0.5 - abs(x0)`) rather than a 0/1 comparison, so the score has the
  same sign convention as the net (> threshold means class 1).
