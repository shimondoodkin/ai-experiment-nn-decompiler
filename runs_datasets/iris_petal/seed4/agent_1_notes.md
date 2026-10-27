# Notes

## What I changed and why
- The dump is a 2 -> 8 -> 3 ReLU MLP with argmax output. Inputs (x0 in [1, 6.9], x1 in [0.1, 2.5])
  look like iris petal length/width, so three ordered classes were expected.
- Instead of simplifying weights neuron by neuron, I evaluated the net on a grid and printed the class
  map. It showed three bands separated by two almost straight, almost parallel lines.
- I bisected along x0 for many fixed x1 to locate each boundary precisely:
  - 0|1 boundary: x0 = 3.262 - 0.984*x1  -> snapped to `x0 + x1 < 3.3`.
  - 1|2 boundary: x0 = 6.755 - 1.083*x1  -> plain `x0 + x1` was too far off at the edges (failed
    with confident disagreements), so the slope was snapped to 11/10: `x0 + 1.1*x1 < 6.75`.
- All 8 hidden neurons and the 3 output scores were replaced by two linear threshold tests in an
  if-chain (the classes are ordered, so a decision list is exact in structure).
- Result: `python check.py agent.py` -> PASS (snap), 0 disagreements where the net is confident.
  3.2 for the first threshold failed; 3.25 and 3.3 both pass, 3.3 gives fewer unsure-zone disagreements.

## Heuristics
- For low-dimensional argmax nets, map the decision regions on a grid first; the region picture
  tells you the shape of the program (decision list, linear cuts, kinks) far faster than the weights.
- Locate each boundary by bisection at many points along the other input; fit a line, check
  residuals for kinks (none here), then snap slope and intercept.
- Ordered classes in bands -> a chain of `if score < t: return k`, with the last class as the default.
- Try the simplest slope (1) first; if check.py reports confident disagreements clustered at one end
  of the boundary, the slope is wrong, so move to the next simple fraction (here 1.1) rather than
  shifting the threshold.
- Among thresholds that pass, prefer the one closest to the fitted boundary (fewer unsure-zone
  disagreements) while keeping it a round/half-step value.
- Hidden neurons with tiny output weights or that are inactive over the input box (here several
  all-negative-weight neurons with negative bias) are dead code; a region-level rewrite removes them
  all at once.
