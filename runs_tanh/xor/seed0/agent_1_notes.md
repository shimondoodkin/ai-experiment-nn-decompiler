# Notes

## What the network computes

Two boolean inputs, four tanh hidden units, one binary output (class 1 if score > 0).
The domain has only 4 points, so I evaluated the net on all of them (scratch/eval.py):

| x0 | x1 | score  | class |
|----|----|--------|-------|
| 0  | 0  | -10.82 | 0     |
| 0  | 1  | +10.50 | 1     |
| 1  | 0  | +10.74 | 1     |
| 1  | 1  | -10.93 | 0     |

That is XOR: class 1 exactly when the inputs differ. All four scores are far from the threshold
(|score| > 10), so the decision is robust.

Reading the hidden units the same way (weights about ±4, biases about ±2 or -5):
- h1_0 ~ "not (x0 and not x1)" (low only at x0=1, x1=0), h1_2 ~ its mirror (low only at x0=0, x1=1);
- h1_1 ~ NOR (high only at 0,0); h1_3 ~ AND (high only at 1,1).
With all output weights negative, the score is high only where NOR and AND are both off and the two
"not-only-one" units sit at their low end, i.e. the two "exactly one" points. Four units for a
two-unit job: the net is redundant (each pair of mirror units would suffice).

## What I changed

Replaced the whole network with `return x0 != x1` (comparisons give 1/0, and 1 > 0, 0 > 0 is false).
Checked: `PASS (exact)` on the exhaustive 4-point domain, no disagreements.

## Heuristics

- For all-boolean inputs with few inputs (2^n small), skip the weights: evaluate the net on the
  whole truth table and read the boolean function off it. Then write the simplest boolean
  expression (here `x0 != x1`).
- Check the margin at each truth-table point; if every |score| is far above the threshold, the
  boolean rewrite is safe and no "unsure zone" handling is needed.
- For a binary output, returning a 0/1 comparison works as the raw score when the threshold is 0.
- Saturated tanh units with weights ~±4 and biases ~±2 on 0/1 inputs act as logic gates: the sign
  of the pre-activation at each corner gives the gate (bias ~ +2 with one +4 and one -4 is an
  "implies"-type gate; bias ~ +1.6 with two -4 is NOR; bias ~ -5 with two +3 is AND).
- Mirror-image units (same magnitudes, swapped input signs) and AND/NOR pairs usually signal a
  symmetric function such as XOR/XNOR; expect redundancy beyond the minimal 2 hidden units.
