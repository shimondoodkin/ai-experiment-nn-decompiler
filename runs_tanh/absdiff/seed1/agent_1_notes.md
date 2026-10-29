# Notes

## What the network computes

A smooth version of `|x0 - x1|`. The output is about 0.18 at x0 = x1, about |d| for 1 <= |d| <= 5,
and it flattens to about 5.64 at the corners (|d| = 6), where training data is rare.

## What I changed

- **Near-constant units (h1_1..h1_4).** Their input weights are at most 0.04 and their biases are
  1.0 to 1.9, so over [-3, 3]^2 each stays near its value at the origin. The output weights times
  `tanh(bias)` plus the output bias sum to about 6.30. All four units fold into one constant.
- **Difference units (h1_0, h1_5, h1_6, h1_7).** Each has weights (+w, -w) on (x0, x1), so the net
  depends only on d = x0 - x1. They come in mirrored pairs with opposite output weights:
  - h1_0 / h1_6: slope about 0.415, biases -1.506 / +1.525, weights +2.63 / -2.67;
  - h1_5 / h1_7: slope about 1.04, biases -0.644 / +0.625, weights +1.15 / -1.19.

  Each pair is `A * (tanh(s*d - b) - tanh(s*d + b))`. That is an even dip in d: about -2A*tanh(b)
  at d = 0, going to 0 as |d| grows. Two dips subtracted from a constant give the V shape of |d|.
- **Snapping.** Slopes go to 5/12 and 1, offsets to 1.5 and 0.6, and the constants are refitted by
  least squares and rounded (6.2, 2.6, 1.2). I checked several snap sets. 5/12 for the wide pair
  mattered: 0.4 or 0.45 cost much more accuracy.
- **Simpler forms fail.** A single tanh pair (max error 0.36), and `c + a*d*tanh(k*d) + q*d^2`
  (0.38), are both far beyond the 0.11 tolerance.

Result: `PASS (snap)`, max |program - net| = 0.064 against a tolerance of 0.110, worst at the
corners x0 = -x1 = ±3. An exact pass would need the original weights.

## Heuristics

- Check whether the hidden units' input weights come in (+w, -w) patterns or other shared
  directions. If every unit that matters has the same direction, the net is a 1-D function of one
  projection (here x0 - x1). Rewrite the program in that variable and probe the net along it.
- Units whose input weights are tiny compared with their bias (|w| * input range << |bias|) are
  constants over the domain. Fold `w_out * act(bias)` into the output bias. Their combined
  variation can still reach a few hundredths, so check the total against the tolerance.
- Two tanh (or sigmoid) units with the same slope, opposite biases (±b) and opposite output weights
  form an even bump or dip: `A * (tanh(s*d - b) - tanh(s*d + b))`. Merge them, averaging s, b and |A|.
- After choosing structure and snapping the nonlinear parameters (slopes, offsets), refit the linear
  parameters (output weights, constant) with least squares on random inputs. Then round them.
  This recovers most of the error the snapping introduced.
- The largest errors tend to be at the domain corners, where the net extrapolates. Compare
  candidates by max error over a sample that includes the corners, not by RMS.
