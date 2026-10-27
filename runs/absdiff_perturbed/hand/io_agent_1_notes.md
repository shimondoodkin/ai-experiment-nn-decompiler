# Notes

## What I did
- Read `spec.json`: two float inputs in [-3, 3], one regression output.
- Tested a few cheap symmetric candidates against `samples.csv` by max residual:
  `|x0 - x1|` (max err 0.077, mean err -0.002), `|x0| + |x1|` (5.8), `max(x0, x1)` (3.0).
  `|x0 - x1|` is clearly the function; the leftover error is the network's smooth
  approximation error (largest at the corners of the box, e.g. x0=-3, x1=3 gives 5.92 instead
  of 6, and near-zero gaps come out slightly positive, ~0.07-0.16, i.e. a rounded "V" bottom).
- The residual is not a clean scale or offset (negative at large gaps, positive at small
  gaps), so there is nothing meaningful to snap beyond coefficients 1 and -1.
- `agent.py`: `gap = x0 - x1; return abs(gap)`. `check.py` result: PASS (snap), max diff 0.080
  vs snap tolerance 0.121, 0 inputs beyond tolerance. PASS (exact) would require copying the
  network's approximation artifacts, which would make the program less readable and less true
  to the intended function.

## Heuristics
- Start by reading the spec (input ranges, output kind) and eyeballing a few rows; small
  integer-coefficient combinations of the inputs often jump out (here out ~= |x0 - x1|).
- Score a short list of simple candidate formulas by max and mean residual over all samples
  before building anything complex; a candidate with max residual a few percent of the output
  range is almost surely right.
- Look at where the largest residuals occur: errors concentrated at the box corners or around
  kinks (where abs/relu/max bend) are typical trained-net approximation error, not missing terms.
- Check whether residuals are a systematic scale/offset before snapping; if they change sign
  with the input magnitude, it is approximation noise and the snapped formula is the answer.
- Prefer the readable, snapped formula once it passes within snap tolerance; do not chase
  exact agreement by modelling network artifacts.
- Name intermediate quantities (e.g. `gap`) to make the intended functionality obvious.
