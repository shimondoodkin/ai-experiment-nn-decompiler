# Notes

## What the network does
One float input x0 in [-2, 2], binary output, 4 sigmoid hidden units.

- All four hidden units are steep sigmoids (slope ~10.5-12.3) with centres at
  x0 = 5.336/10.604 = 0.503, 6.191/12.324 = 0.502, -5.248/10.608 = -0.495 (decreasing unit,
  "x0 < -0.495") and -5.209/10.547 = -0.494.
- Units 0 and 1 are duplicates (both "x0 > ~0.5", negative output weight ~-9): together they push the
  score far below zero for x0 > 0.5.
- Unit 2 ("x0 < -0.5", weight -9) pushes the score down for x0 < -0.5.
- Unit 3 ("x0 > -0.5", weight +8.3) plus bias 0.63 makes the score positive in between.
- Scanning the net on a fine grid gives exactly two decision boundaries: x0 = -0.501 and x0 = 0.5025;
  class 1 inside, class 0 outside.

## Program
`return abs(x0) < 0.5` -- the two boundaries snapped to one shared threshold, +-0.5 (a half step).
Check: PASS (snap), 0 disagreements where the net is confident; all differences are in the net's
unsure zone around the boundaries (the net's own boundaries sit at -0.501 / +0.5025).

## Heuristics
- For sigmoid units, the centre is -b/w and the steepness is |w|; with |w| >~ 10 over a range of 4,
  treat each unit as a step function at its centre, with direction given by the sign of w.
- Units with nearly equal centres and same direction are duplicates; sum their output weights.
- For 1-input binary nets, the fastest route is to scan the net on a dense grid, list the sign
  changes of the score, and snap them; symmetric boundaries (-a, +a) become `abs(x) < a`.
- A decreasing step "x < -a" is 1 - "x > -a"; recognising this pairs units 2 and 3 into a single
  boundary, which explains why the output weights look like +-9 with a small bias.
- Snap boundaries to half-steps (0.5) when the net's boundaries are within ~0.5% of them; the
  checker's unsure zone absorbs the residual offset.
