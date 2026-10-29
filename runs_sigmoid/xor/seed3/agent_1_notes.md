# Notes

## What I did

- Both inputs are bool, so the domain has only 4 points. I evaluated the dumped net on all of them
  (scratch/tt.py):

  | x0 | x1 | score     | class |
  |----|----|-----------|-------|
  | 0  | 0  | -9.9986   | 0     |
  | 0  | 1  | +0.000059 | 1     |
  | 1  | 0  | +9.1096   | 1     |
  | 1  | 1  | +0.000252 | 1     |

- The decision is class 1 unless both inputs are 0, i.e. logical OR. `agent.py` is `return x0 or x1`.
  The checker reports PASS (exact), exhaustive over the 4 inputs.
- Caveat: the two cases with x1 = 1 have scores of about 1e-4. That is inside the checker's unsure
  zone (|score| < 0.49), so the net is barely deciding there. The confident part of the net is
  only "x0=1, x1=0 -> 1; x0=0, x1=0 -> 0". `return x0` would also pass in snap mode, since it
  disagrees only at (0,1), which is in the unsure zone. OR is the version that matches every
  decision, so that is the one I kept. The sigmoid units all fall away: none of them is needed to
  express the decision.

## Heuristics

- For all-bool inputs with a small domain (2^n ≤ a few thousand), enumerate the truth table first
  and write the program from it. Don't try to simplify the weights unit by unit.
- Look at score magnitudes as well as signs. Scores near 0 on some rows mean the net is unsure
  there (its training data was likely ambiguous or contradictory). Match the sign to get an exact
  pass, but mention that those rows are fragile.
- Sigmoid nets on bool inputs are piecewise constant on the corners, so the hidden units can be
  dropped once the truth table is known. Pick the shortest boolean formula that fits the table
  (OR / AND / XOR / x0 > x1 ...).
- When a simpler formula would also pass in snap mode by flipping only unsure-zone rows, prefer the
  one that matches every decision, since exact beats snap.
