# Notes

## What the network does
Two boolean inputs, binary output (class 1 iff score > 0). The domain has only 4 points, so I
evaluated the dump on all of them (scratch/tt.py):

| x0 | x1 | score |
|----|----|-------|
| 0 | 0 | -9.22 |
| 0 | 1 | +7.72 |
| 1 | 0 | +10.96 |
| 1 | 1 | -8.34 |

That is XOR. Every margin is large (|score| > 7), so there is no ambiguous point.

## What I changed
- Replaced the 4 sigmoid units and output layer with `return x0 != x1`. A comparison yields 1/0, and
  1 > 0 / 0 > 0 gives the same class as the net on every input.
- Checker: `PASS (exact)` (exhaustive, 4 inputs).
- How the units read: h1_0 is about AND(x0, x1), weighted -16; h1_1 and h1_3 are both about NOR, a
  duplicated pair with combined weight -17; h1_2 is about x0 AND NOT x1, weight +5.8, and is redundant
  given the bias of +7.8. So the score is about 7.8 - 16*AND - 17*NOR, which is positive exactly when
  the inputs differ.

## Heuristics
- With all-boolean inputs (or any small discrete domain), enumerate the full truth table first.
  The truth table is the function. Reading the weights is only needed to explain it.
- For binary outputs, look at the score margins. If all of them are far from the threshold, a pure
  boolean expression gives an exact match. Keep only the sign, not the magnitudes.
- Recognise standard boolean functions from the truth table (XOR -> `a != b`, XNOR -> `a == b`,
  AND -> `a and b`, OR -> `a or b`, a single input or its negation). The language's comparisons
  return 1/0, which already sit on the right side of a threshold of 0.
- A sigmoid unit with large weights on boolean inputs acts as a logic gate. Read its gate from the
  sign of its pre-activation at each corner. Two units that are the same gate (here, two NOR units)
  are duplicates; add their output weights together.
- A unit that is active only where the bias alone already decides the class (here, x0 AND NOT x1) is
  redundant; drop it.
