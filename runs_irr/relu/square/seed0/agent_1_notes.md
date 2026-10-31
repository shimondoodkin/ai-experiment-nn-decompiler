# Notes

## What the network does

1 input on [-2, 2], 8 ReLU units, regression.

- Dead units: h1_0, h1_4, h1_5, h1_6 have pre-activations that stay negative over the whole box
  (their knots are at x0 = -11.9, -2.48, +2.47, -20.8). Removed.
- Live units, written as slope * relu(+-(x0 - knot)) after folding the output weight in:
  - h1_1: 1.585 * relu(x0 - 0.402)
  - h1_7: 1.593 * relu(x0 - 1.199)
  - h1_3: 1.598 * relu(-0.400 - x0)
  - h1_2: 1.605 * relu(-1.199 - x0)
- The left and right pairs mirror each other: knots +-0.4 and +-1.2, all slopes ~1.6. So the net is
  an even function of x0: `0.05 + 1.6*relu(|x0| - 0.4) + 1.6*relu(|x0| - 1.2)`.
- Interpretation: a piecewise-linear bowl. With knots 0.4, 1.2, 2 the chord slopes of x0^2 are
  (1.44-0.16)/0.8 = 1.6 and (4-1.44)/0.8 = 3.2, exactly the net's slopes, so it looks like a
  coarse approximation of x0^2 (flat ~0.05 in the middle instead of the parabola's dip).

## Changes

- Dropped the 4 dead units.
- Snapped knots to +-0.4, +-1.2, all four slopes to 1.6, merged mirror pairs via abs(x0).
- Bias 0.0555 -> 0.05 (also tried 0: passes, max error 0.057; 0.05 gives the smallest error, 0.026).
- Result: PASS (snap), max |program - net| = 0.026 vs tolerance 0.077. Not exact (snapping slopes
  costs ~0.03 at the box edges).

## Heuristics

- For each unit compute the knot -b/w and test it against the input box; units whose
  pre-activation has one sign over the whole box are constants (0 if always negative) or linear terms.
- Fold the output weight into the unit: out_w * relu(w*x + b) = (out_w*|w|) * relu(+-(x - knot)).
  Comparing the folded slopes, not the raw weights, shows which units are copies of each other.
- Look for mirror pairs (knots at +-k, same folded slope, opposite input sign): they combine into
  `s * relu(abs(x) - k)`, i.e. the function is even.
- Snap knots to round values (here tenths) and equal slopes to one shared value; check whether the
  slopes/knots coincide with chords of a simple function (x^2, |x|) to name what the net approximates.
- Snap the constant last and try a couple of candidates (0, rounded value); pick the one that
  minimises max error rather than the simplest, if both pass.
