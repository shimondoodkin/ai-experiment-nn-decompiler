# Notes

## What I changed and why

The dump is a 2 -> 4 -> 1 ReLU network on two boolean inputs with a binary output (class 1 if > 0).
With only 4 possible inputs I evaluated the network on the whole truth table:

| x0 | x1 | score (approx) | class |
|----|----|----------------|-------|
| 0  | 0  | -12.7          | 0     |
| 1  | 0  | +11.9          | 1     |
| 0  | 1  | +13.2          | 1     |
| 1  | 1  | -13.1          | 0     |

That is XOR. The hidden units are recognisable pieces of it:
- h1_0 ~ 3.2*relu(x0 - x1): "x0 and not x1".
- h1_2, h1_3 ~ 2.8*relu(x1 - x0), 4.8*relu(x1 - x0): duplicate "x1 and not x0" detectors (redundant; merge into one).
- h1_1 ~ 3*relu(1 - x0 + x1): a bias-carrying unit that pushes the (0,0) and (1,1) cases negative.
- Output bias -1.87 is just a margin term.

Snapped form would be roughly `relu(x0 - x1) + relu(x1 - x0) - something`, i.e. |x0 - x1|, which for
booleans is exactly `x0 != x1`. Since the output is binary, only the sign matters, so the whole net
collapses to `return x0 != x1` (1 for class 1, 0 for class 0, and 0 is not > 0). Checker: PASS (exact),
exhaustive over all 4 inputs.

## Heuristics

- When the input space is small (few booleans / small integer ranges), enumerate it fully and read off
  the truth table first; that often identifies the function directly (XOR, AND, majority, ...).
- For a binary output only the sign matters: drop scale factors and replace the score with a boolean
  expression that has the same sign pattern (comparisons yield 1/0, and 0 maps to class 0).
- Hidden units with weights (+a, -a) on two inputs and ~0 bias are relu(x_i - x_j) "i and not j"
  detectors; several units with proportional weight vectors are duplicates and can be merged by
  summing their output weights.
- relu(a - b) + relu(b - a) = |a - b|; on booleans |a - b| == (a != b).
- A unit with a large positive bias that is always active acts as an affine term (bias + linear
  correction), not a real nonlinearity; fold it into the output bias/linear part.
- Tiny biases (1e-5 .. 0.04) relative to weights of ~3 are noise: snap them to 0.
