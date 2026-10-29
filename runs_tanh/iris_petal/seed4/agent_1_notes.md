# Notes

## What the network does
A 2 -> 8 tanh -> 3 argmax classifier. Mapping argmax over a grid shows three ordered classes
in bands across the (x0, x1) plane, with no islands: class 0 at small x0/x1, class 2 at large
x0/x1, class 1 between them. Class 0 and class 2 never touch.

## What I changed
- Dropped the 8 tanh units and the 3 score sums. Two linear decision boundaries replace them.
- Found the boundaries by scanning x0 at 49 fixed x1 values for argmax changes, then fitting
  `x0 = a*x1 + c` to the change points of each class pair:
  - 0|1: x0 = -2.405*x1 + 4.202 (max residual 0.006, so practically straight) -> snapped to
    `x0 + 2.4*x1 < 4.2`.
  - 1|2: x0 = -0.997*x1 + 6.588 (max residual 0.03, slightly curved) -> snapped to
    `x0 + x1 < 6.6`.
- Wrote the result as an ordered if-chain: test class 0 first, then 1, else 2.
- Result: `PASS (snap)`, with 0 disagreements where the net is confident and 3496 inside the unsure
  zone (mostly near the corner x0~6.5, x1~0.1, where the curved 1|2 boundary bends away from the line).
  An exact pass is not realistic with straight lines, because the tanh boundary is slightly curved.
- Tried other snaps: 6.5 fails (921 confident disagreements), 6.55 passes with more unsure
  disagreements (5148), and slope 2.5 / 4.25 fails (1 confident disagreement). So 2.4 / 4.2 / 1 / 6.6 is kept.

## Heuristics
- For small argmax nets with 2 inputs, print the argmax over a coarse grid first. If the classes
  form ordered bands, the program is just a chain of linear threshold tests, and the hidden units
  do not need to be interpreted one by one.
- Locate boundaries by 1-D scans (fix one input, sweep the other on a fine grid, record class
  changes). Then fit `x_a = slope*x_b + c` for each class pair. The max residual tells you whether a
  straight line is enough.
- Snap slope and intercept to one decimal (or a simple fraction) and let the checker decide.
  When several snaps pass, keep the one with the fewest unsure-zone disagreements. When a boundary is
  curved, the fitted intercept is pulled by the curved end, so try the snapped value on both sides.
- When the classes are ordered bands, write the decision as an ordered if-chain (lowest class first).
  Each test then only needs the one boundary between adjacent classes, and pairs that never touch
  (here 0 and 2) need no test.
- Inputs bounded like x0 in [1, 6.9] and x1 in [0.1, 2.5], with three ordered classes, suggest a
  measurement-based classification. The boundaries have mixed slopes (one steep, one at 45 degrees),
  so keep the per-boundary weights rather than forcing one shared combination.
