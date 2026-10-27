# Notes

## What the failing cases show
- Labels in repair.csv follow one rule: class 1 iff |x0| < 1/2. The label changes between
  -0.507 and -0.495, and between 0.498 and 0.500.
- The network's right edge sits too far in, at about 0.489 to 0.491, instead of 0.5. So it
  answers 0 on the 3 failing points (x0 = 0.4914, 0.4954, 0.4979), which are inside the band.
  Its left edge (about -0.5) is correct.

## What I changed
- Nothing in the logic. `program.py` already computes `0.5 - abs(x0)`, the symmetric band
  |x0| < 1/2. The decompiled constants were already snapped to the clean value 1/2, so the
  network's slightly shifted right edge (about 0.49) did not carry over.
- `agent.py` is the same one-line rule. It is symmetric and has no special cases.

## Score (python score.py agent.py)
- right on 2000/2000
- network failures fixed: 3/3
- network-correct answers broken: 0/1997
