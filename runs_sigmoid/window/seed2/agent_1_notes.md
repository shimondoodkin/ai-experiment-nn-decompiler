# Notes

## What the network does

One float input x0 in [-2, 2], binary output (class 1 when the score > 0). The 4 sigmoid units are
two pairs of steep steps:

- h1_0 = sigmoid(-12.54 x0 + 6.49): a falling step at x0 = 6.49/12.54 = 0.518
- h1_2 = sigmoid(12.07 x0 - 6.27): a rising step at x0 = 0.519, weight -8.0
- h1_3 = sigmoid(-12.99 x0 - 6.42): a falling step at x0 = -0.494, weight -13.2
- h1_1 = sigmoid(3.84 x0 - 2.54): a shallow ramp with a small weight (-0.71); it only shifts the
  score a little and does not create a crossing.

Scanning the score over [-2, 2] gives exactly two sign changes, found by bisection at
x0 = -0.4990145 and x0 = 0.5034656. The score is negative outside (about -6.4 at the left end,
-8.4 at the right end) and positive inside (+6.75 at x0 = 0). So the net is a band (window)
detector: class 1 iff -0.499 < x0 < 0.503.

## What I changed

- Replaced the 4 sigmoids with a single interval test.
- Snapped both edges to +-1/2 (they are within 0.001 and 0.0035 of it), which makes the band
  symmetric: `abs(x0) < 0.5`.
- Return +1 / -1 so the decision `raw > 0` gives the class.

Check: `PASS (snap)`, 0 disagreements where the net is confident; all disagreements are in the
thin slivers between 0.5 and 0.50347 and between -0.49901 and -0.5, inside the net's unsure zone.

An unsnapped variant `return 1 if -0.49901453 < x0 < 0.50346557 else -1` gives `PASS (exact)`;
I kept the snapped `abs(x0) < 0.5` as the final version because it is the readable rule the net
approximates.

## Heuristics

- For a 1-input binary net, do not decode the units first: scan the score densely over the input
  range, list the sign changes, and bisect each one to full precision. The decision is then an
  interval rule whose number of intervals equals the number of crossings + 1.
- Read each sigmoid unit as a step at x = -b/w with sharpness |w|. Steep units with large output
  weights carry the decision; a shallow unit with a small output weight is a bias correction and
  can be dropped.
- Two steps of opposite direction with opposite-sign output weights form a bump (window). Rising
  step with negative weight and falling step with negative weight around a positive bias = "inside
  the band" detector.
- When the two edges of a band are near +-c, snap them to a symmetric pair and write `abs(x) < c`.
  Prefer simple fractions (1/2) for edges on a [-2, 2] float range.
- If a snapped threshold fails, keep the bisected value with ~8 significant digits: that usually
  gives PASS (exact) for binary outputs, since disagreements then only sit within float noise.
- Return +-1 instead of the raw score for a binary rule; only the sign matters.
