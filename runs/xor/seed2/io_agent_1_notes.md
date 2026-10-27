# Notes

## What I did
- Inputs: two booleans x0, x1; binary output with threshold 0.
- All four raw outputs are tiny (|out| < 5e-7), i.e. the score is essentially zero everywhere;
  only its sign carries information. Sign: positive when x0 = 0, negative when x0 = 1,
  independent of x1.
- So the decision is `not x0` (x1 is unused / a dead input). Returning the 0/1 decision rather
  than a score, since the score magnitude is noise-level and not meaningful to reproduce.
- `python check.py agent.py` -> PASS (exact), exhaustive over 4 inputs.

## Heuristics
- For small boolean input spaces, read the truth table directly from `class`; the samples are
  exhaustive, so match the table first and formulate afterwards.
- Check each input for relevance: if flipping an input never changes the class, drop it (dead input).
- If the raw output magnitude is near zero everywhere (≈ atol or below), treat the score as
  degenerate and return the decision, not a fitted score.
- Name the logic by its truth table (here NOT x0) rather than as a weighted sum with a threshold.
