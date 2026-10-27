# Notes

## What the network does
Two boolean inputs, one hidden layer of 4 ReLUs, binary output (class 1 if score > 0).
Evaluating all 4 input combinations of the dump:

| x0 | x1 | score  | class |
|----|----|--------|-------|
| 0  | 0  | -11.1  | 0     |
| 1  | 0  | +9.59  | 1     |
| 0  | 1  | +11.9  | 1     |
| 1  | 1  | -11.4  | 0     |

That is XOR, so `agent.py` is simply `return x0 != x1` (comparison gives 1/0). PASS (exact), exhaustive.

## What the hidden units were doing (for the record)
- h1_0 ~ relu(2*x0 + 4*x1 - 2): active when x1 = 1 (penalty for "x1 and maybe x0").
- h1_1 ~ 3.4 * relu(1 - x0 - x1): NOR detector, fires only at (0,0), pushes score negative.
- h1_2: weights ~0.2-0.27 with negative bias; it is 0 on every boolean input -> dead, removed.
- h1_3 ~ 2.83 * relu(x1 - x0): "x1 and not x0", boosts (0,1) back positive.
- Bias 9.59 makes (1,0) positive by default.
Note h1_0 at (1,0) and h1_1 at (1,0) sit exactly at the ReLU kink (pre-activation ~0), which
is why exact coefficients don't matter once snapped.

## Heuristics
- With a small discrete input domain (here 2 bools = 4 points), enumerate every input and read
  off the truth table first; the decision rule is often a textbook boolean function (XOR, AND, majority).
- For binary/argmax outputs only the sign/ordering matters; the score magnitude can be dropped, so a
  comparison expression (returning 1/0) is a valid replacement for a whole weighted sum.
- Units whose pre-activation is negative on every reachable input are dead; delete them.
- Biases that equal minus one weight (e.g. 2.08*x0 - 2.08) put the ReLU kink exactly on an input
  corner; snap them to integer thresholds like relu(x0 + ... - 1).
- Recognize standard ReLU motifs: relu(1 - a - b) = NOR, relu(a + b - 1) = AND, relu(b - a) = b AND NOT a
  for booleans.
- Verify with the checker after each simplification; prefer the simplest formula that passes exact.
