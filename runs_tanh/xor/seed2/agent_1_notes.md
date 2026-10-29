# Agent notes

## What the network is

Two boolean inputs, 4 tanh hidden units, binary output (class 1 if score > 0).
Because the domain has only 4 points, I evaluated the dump on all of them (scratch/tt.py):

| x0 | x1 | score  | class |
|----|----|--------|-------|
| 0  | 0  | -13.00 | 0     |
| 0  | 1  |  +9.22 | 1     |
| 1  | 0  | +10.70 | 1     |
| 1  | 1  |  -9.28 | 0     |

This is XOR. Every margin is about 9 or more, so the decision is not fragile anywhere.

## What I changed

- Replaced the 4 tanh units and the output layer with `x0 != x1` (the comparison gives 1/0;
  1 > 0 gives class 1 and 0 > 0 gives class 0).
- All weights, biases and hidden units were dropped. On a boolean domain the per-unit
  coefficients do not matter, only the sign of the score at each corner.

Result: `PASS (exact)` on the exhaustive check (4 inputs).

## Heuristics

- When all inputs are boolean or small integers and the domain is tiny (here 2^2 points),
  enumerate the dump's output on every point before reading the weights. The truth table is the
  program; the hidden layer is only one way of producing it.
- For binary outputs, look at the sign pattern of the score on the corners and match it to a
  named boolean function (AND, OR, XOR, majority, threshold count). XOR on bools is `a != b`.
- Check the margins: if every corner's score is far from the threshold, a pure boolean
  expression will pass exactly, with no need to keep scale factors.
- A binary program can return a comparison directly (1/0), as long as the threshold is 0 and
  "true" maps to the positive class.
- For tanh/sigmoid nets with large input weights (|w| about 3 to 5 on 0/1 inputs), each unit acts
  close to a step function; but reading unit-by-unit is unnecessary when the input space can be
  enumerated.
