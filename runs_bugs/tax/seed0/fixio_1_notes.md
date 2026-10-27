# Notes

## What I did
- Read `spec.json`: one float input `x0` in [0, 200], regression output, scoring tolerance ~1.24.
- Sorted `repair.csv` by `x0` and looked at `label` against `x0`. The label is piecewise linear and
  continuous. The slope is 0.1 below 20, 0.2 from 20 to 80, and 0.4 above 80.
- Checked the formula `0.1*x + 0.1*max(x-20,0) + 0.2*max(x-80,0)` against every label. The largest
  error is 7e-15, so it matches exactly and not only within tolerance.
- Wrote `agent.py` as a three-bracket marginal-rate schedule (like income tax):
  10% of the part up to 20, 20% of the part from 20 to 80, and 40% of the part above 80.

## Why
- The existing system fails on x0 from about 70 to 170. In that range it overshoots, because its
  slope is about 0.38 where the true slope is 0.2, and its kink is in the wrong place. Below and
  above that range it happens to stay within tolerance.
- The labels follow one simple rule across the whole input range, so the program uses that rule
  instead of patching the failing interval. It has no special cases for single points.

## Result
`python score.py agent.py`: right on 2000/2000. It fixes all 989 failures and breaks 0 of the
1011 points the system had right.
