# Instructions

This code is a dump from a neural network. It roughly resembles some functionality.
Refactor the code and optimize it. Snap coefficients and thresholds to nearby meaningful values
(0, ±1, small integers, simple fractions, half-steps between integer inputs, one shared threshold
for nearby thresholds). Detect constants and fixtures. Try to form formulas and to name variables.
Remove redundant code and code that doesn't help. Check your version with `check.py`. Afterwards,
write down what you learned and any heuristics you used that would apply to other networks.

## Practical details

- Read `dump.py` (the network as code) and `spec.json` (inputs, their types and bounds, and the
  output kind).
- Write your version to `agent.py`. Keep `def net(x0, ...)` with exactly the same arguments as in
  `dump.py`.
- Check it with `python check.py agent.py`. `PASS (snap)` is enough; `PASS (exact)` is better.
  `FAIL` lists inputs where your version disagrees with the network.
- The language your code must use: {ALLOWLIST}
- Write `agent_notes.md`: what you changed and why, then a section titled exactly
  `## Heuristics` listing heuristics that would apply to other networks.
- Work only with the files in this folder.
- Put any scratch scripts or files in the scratch/ folder here.
