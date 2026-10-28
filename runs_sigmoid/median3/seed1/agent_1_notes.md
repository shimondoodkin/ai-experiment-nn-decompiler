# Notes

## What the network computes
- Sampling the dump (200k uniform points) and correlating the output with simple candidates: the
  output is close to the **median of the three inputs**, with correlation 0.998 (sum 0.90, max/min 0.58).
- The hard median does not pass, though: `max(min(x0,x1), min(max(x0,x1),x2))` is off by up to 0.64
  (snap tolerance 0.118). The net is a *soft* median. Near ties (two inputs close) it is pulled toward
  the third input by up to ~0.3. At box corners where x0 and x1 are far apart and x2 ties one of them
  (e.g. (-3,3,-3) -> -2.36), the gap reaches 0.64. The error is also not symmetric in the inputs.
- Smooth soft-median formulas didn't pass either. I fitted soft max/min built from sigmoid gates
  (`a*g + b*(1-g)`, g = sigmoid(k(a-b))), both chained and symmetric, minimising the max error.
  The best was 0.35. The net's deviation from the median is idiosyncratic, not a clean temperature.

## What I did instead (structural compression, then snapping)
1. PCA of the layer-2 pre-activations: 93% of the variance is in one direction. The 16 layer-2 units
   split into two mirror groups: the positive-output units are sign-flipped copies of the negative ones,
   since sigmoid(z) = 1 - sigmoid(-z).
   Replacing them by **one** layer-2 unit (initialised as the group average, then refit) gives a max
   error of 0.061.
2. Layer 1 has 9 "near-linear" units: weights ~0.15 on every input, so they stay in sigmoid's linear
   range. I replaced them with a direct linear term in x inside the score. 6 steep units remain:
   - two x0-vs-x1 gates;
   - x2 vs x0;
   - x2 vs mean(x0,x1);
   - x1 vs x2;
   - a soft step in x2.

   Max error 0.063.
   Ablation:
   - dropping one of the two x0-vs-x1 gates gives 0.24. The two gates have opposite output weights
     and nearly cancel. Their difference is a bump at x0 ~ x1, tilted by x2;
   - dropping x1-vs-x2 or x2-vs-mean gives 0.24 / 0.74;
   - dropping the x2 step gives 0.096.
3. Snapping the whole first layer to a 0.25 grid at once broke it (0.17). Snapping greedily worked.
   The order was:
   - closest-to-grid weight first;
   - fix it, refit all still-free parameters with an L8 loss (a max-error proxy);
   - keep the snap only if the validation max error stays under a threshold;
   - otherwise try a finer grid (0.25 -> 0.125 -> 0.05).

   This brought all first-layer weights to quarter/eighth values and the biases to 0 (one 0.05 left).
   Most mixing weights ended on 0.05-0.125 steps. The x0 linear coefficient (-0.112) and the score
   bias (0.353) would not snap. Forcing the linear term to -sum/8 doubles the error.
4. The output, `6.55 - 13.15*sigmoid(score)`, is about `-6.575*tanh(score/2)`.

Result: `PASS (snap)`, max |program - net| = 0.099 against a tolerance of 0.118. That is 6 sigmoid
comparisons and one output sigmoid, instead of 16+16 hidden units. A less-snapped variant
(more 2-3 digit weights) reached 0.082.
I did not find an exact closed form. The honest reading is "a soft median whose smoothing the net
learned unevenly".

## Heuristics
- For regression nets, first correlate the output with a library of symmetric candidates
  (sum, max, min, median, products, abs differences). A correlation above 0.99 names the intent even
  when the formula itself won't pass the tolerance.
- Probe the residual (net - candidate) along tie slices: two inputs equal, the third offset by b.
  If it peaks at moderate b and decays, the net is a smooth/soft version of a piecewise function.
  Large residuals at box corners are extrapolation artefacts that no clean formula will match.
- Sigmoid nets: units with small weights on all inputs (|w|*range < ~1.5) sit in the linear part of
  the sigmoid. Fold them into a linear skip term.
- Sigmoid nets: layer-2 units with mirrored weights and opposite-sign output weights are the same
  unit (sigmoid(z) = 1 - sigmoid(-z)). PCA of the pre-activations tells how many distinct units the
  layer really has. Collapse to that many and refit.
- Two near-identical steep units with opposite output weights are not redundant. Their difference is
  a derivative-like bump along the shared boundary, so ablate before pruning.
- Snap greedily and refit the free parameters after each snap, with an L^p loss (p 8-12) to control
  the max error. Snapping a whole layer at once to a coarse grid fails for smooth (sigmoid) nets.
- Keep a margin (~15-30% of the snap tolerance), because the checker adds boundary and corner
  points that a uniform validation set under-samples. Validate on a grid that includes the corners.
