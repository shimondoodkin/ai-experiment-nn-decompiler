# Notes

- `spec.json`: three float inputs in [-3, 3], regression output; the scorer accepts |pred - label| <= ~0.119.
- I compared the `label` column with a few simple functions of the inputs (max, min, sum, median).
  The median of (x0, x1, x2) matches every label exactly (max abs error 0.0 over all 2000 rows,
  including the 402 rows where the existing system is wrong). Max, min and sum are far off (errors > 5).
- The existing system looks like an approximation of the median that misses by more than the
  tolerance on about 20% of the points. So I did not patch it; I replaced it with the exact rule.
- Program (`agent.py`): median of three in the allowed language,
  `max(min(x0, x1), min(max(x0, x1), x2))`. There are no special cases.
- Score: 2000/2000 right, 402/402 failures fixed, 0 previously-right points broken.
