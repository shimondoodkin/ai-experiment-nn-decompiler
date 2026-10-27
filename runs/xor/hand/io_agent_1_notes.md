# Notes

The function has two boolean inputs and four samples:
(0,0) gives -0.5, (0,1) gives +0.5, (1,0) gives +0.5, (1,1) gives -0.5.
So the decision is XOR, and the raw score is exactly XOR - 0.5 (+-0.5 margin around the threshold 0).

`agent.py`: `differ = x0 != x1; return differ - 0.5`. `check.py` reports PASS (exact), checked exhaustively on all 4 inputs.

## Heuristics

- With only boolean inputs, write out the full truth table first. The decision is then a known
  boolean function (here XOR/parity), so name it and don't fit a linear model.
- Look at the raw output values as well as the class. If only two values occur and they are symmetric
  around the threshold (+-c), the score is `c * (2*decision - 1)`, which is `decision - 0.5` when c = 0.5.
  You get an exact match, not only a snap match.
- Comparisons return 1/0, so `x0 != x1` is XOR for bools and the result can go straight into arithmetic.
