# Instructions

`samples.csv` holds samples of a black-box function: inputs and the function's output. It roughly
resembles some functionality. Write a short, readable program that reproduces the function. Snap
coefficients and thresholds to nearby meaningful values (0, ±1, small integers, simple fractions,
half-steps between integer inputs, one shared threshold for nearby thresholds). Detect constants and
fixtures. Try to form formulas and to name variables. Check your version with `check.py`. Afterwards,
write down what you learned and any heuristics you used that would apply to other functions.

## Practical details

- Read `samples.csv` and `spec.json` (inputs, their types and bounds, and the output kind). In
  `samples.csv`, `out` is the function's raw output (`out0, out1, …` for a function with several
  outputs); `class` is the decision when the output kind is binary (`out > threshold`) or argmax
  (index of the largest output).
- Write your version to `agent.py` as `def net(x0, ...)`, with the input names from `spec.json`. For
  a binary output, return either the score or the 0/1 decision; for argmax, return the class index.
- Check it with `python check.py agent.py`. `PASS (snap)` is enough; `PASS (exact)` is better.
  `FAIL` lists inputs where your version disagrees with the function.
- The language your code must use: {ALLOWLIST}
- Write `agent_notes.md`: what you changed and why, then a section titled exactly
  `## Heuristics` listing heuristics that would apply to other functions.
- Use only `samples.csv`, `spec.json` and `check.py`. Do not open `oracle.bin` or `_checker/`.
- Work only with the files in this folder.
