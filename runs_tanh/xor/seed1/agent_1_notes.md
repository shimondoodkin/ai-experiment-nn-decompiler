# Notes

## What I did
- 2 boolean inputs, binary output, so the input domain is only 4 points. I evaluated the dump on all
  four points instead of simplifying the four tanh units algebraically.
- Raw scores: (0,0) -11.51, (1,0) +11.42, (0,1) -3.8e-5, (1,1) +7.3e-5.
- With x1=0 the net is confident and follows x0. With x1=1 the score is about 1e-4, essentially
  zero and inside the checker's unsure zone, but its sign still follows x0.
- So the decision is exactly `x0`, and x1 does not affect it. `agent.py` is `return x0`.
  The check gives PASS (exact), with 0 disagreements, confident or unsure.
- Alternative: `x0 and not x1` (inhibition) also passes, with one unsure-zone disagreement at (1,1).
  The x1=1 scores are about 0, so the net seems to sit on the fence there; the target task may be
  either "x0" or "x0 and not x1". I kept `x0` because it matches every sign the net produces.

## Heuristics
- For all-boolean (or small finite) input domains, enumerate the truth table from the dump first.
  The program is then a boolean formula of the inputs, and the hidden units do not need decoding.
- For binary output, check the raw score's magnitude on each row, not only its sign. Rows with |score|
  far below the confident rows are "fence" rows: the net is undecided there. Pick the formula that
  matches every sign, the exact one, and note the other fence-row readings as alternatives.
- With a 0/1 input and threshold 0, `return x0` is a valid binary program: 0 is not > 0. No offset
  like `x0 - 0.5` is needed.
- An input whose flip changes only the score's magnitude, never its sign, can be dropped from the
  decision program, even when its weights are large (here |w| ~ 6-7 on x1 in every unit).
