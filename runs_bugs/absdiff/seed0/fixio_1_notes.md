# Notes

## What I did
- Read `spec.json`: two float inputs x0, x1 in [-3, 3], one regression output; a case counts as right
  when within the tolerance in `score.json` (~0.119).
- Loaded `repair.csv` (2000 rows, 1010 marked failing) and checked a few simple candidate formulas
  against the `label` column: x0*x1, x0+x1, max, min, x0^2 and |x0 - x1|.
- |x0 - x1| matched every label exactly (max abs error 0.0). The others were off by several units.

## Program
```python
def net(x0, x1):
    return abs(x0 - x1)
```

## Result
`python score.py agent.py`: right on 2000/2000; fixes all 1010 failures of the existing system and
breaks none of its 990 correct answers.

## Why
The labels are exactly the absolute difference of the two inputs, so the general rule is
the absolute difference itself. The existing system approximates it, with errors above tolerance
mainly where the true value is small (near the diagonal x0 = x1, e.g. net 0.53 vs label 0.03).
No per-point special cases were needed.
