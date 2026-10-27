# Notes

- `repair.csv` has all four boolean input combinations. The correct labels are 0,1,1,0 for
  (0,0),(0,1),(1,0),(1,1): exactly XOR of x0 and x1.
- The existing system answers 1,1,0,0, i.e. it outputs `not x0` (it ignores x1). That is why it fails
  at (0,0) and (1,0).
- The general rule is "output 1 when exactly one input is 1", so `agent.py` returns `x0 != x1`
  (a comparison, giving 1/0). There are no special cases; the rule covers the whole input domain,
  which is only these four points.
- Score: right on 4/4, fixed both failures, broke none.
