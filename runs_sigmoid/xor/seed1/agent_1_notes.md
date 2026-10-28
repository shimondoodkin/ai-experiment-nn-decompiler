# Notes

## What the network computes

Two boolean inputs, 4 sigmoid hidden units, binary output (class 1 if score > 0).
Because the domain has only 4 points, I evaluated the network on all of them:

| x0 | x1 | score  | class |
|----|----|--------|-------|
| 0  | 0  | -8.58  | 0 |
| 0  | 1  | +10.60 | 1 |
| 1  | 0  | +7.88  | 1 |
| 1  | 1  | -8.61  | 0 |

That is XOR, so the program is `return x0 != x1`. Checker: PASS (exact), exhaustive over 4 inputs,
smallest margin |score| is about 7.9, far from the unsure zone (0.52).

## How the hidden units read

- Units 0, 1, 3 all have weights about (-a, +a) and a negative bias (about -0.7a): they fire only
  for x1=1, x0=0 ("x1 and not x0"). They are three near-duplicates; merged they add about +21.
- Unit 2 has weights about (-10, +10) and a positive bias +4.9: it is off only for x0=1, x1=0
  ("not (x0 and not x1)"). With output weight -16.8 and bias +8.0 it gives +8 for (1,0) and
  -8.8 elsewhere.
- Sum: (0,1) gets the duplicated units, (1,0) gets the bias freed by unit 2, and the two equal
  inputs stay negative. So it is (x1 and not x0) or (x0 and not x1) = XOR.

I dropped all the weights rather than snapping them, since on a boolean domain the truth table
is the full function.

## Heuristics

- When all inputs are boolean (or the domain is tiny), evaluate the net exhaustively and read the
  truth table first; recognise standard boolean functions (AND, OR, XOR, majority, ...) and write
  that directly.
- For binary output, a comparison that yields 1/0 (`x0 != x1`) is a valid score with threshold 0;
  no need to keep the real-valued score.
- Sigmoid units with weights (-a, +a) and bias b: the sign of b decides the gate. b < 0 with
  |b| < a means "x1 and not x0"; b > 0 with b < a means "not (x0 and not x1)", i.e. x1 or not x0.
- Several hidden units with proportional weight vectors and similar bias/weight ratios are
  duplicates of one gate; merge them by summing their output weights.
- Check the margin (smallest |score| over the domain) against the unsure zone to know how
  robust the snapped program is.
