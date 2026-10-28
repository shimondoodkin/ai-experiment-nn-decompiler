# Notes

## What the network computes

A 3 -> 16 -> 16 -> 1 sigmoid net, regression, inputs in [-3, 3].

- Fitting simple candidates against the net: the **median of the three inputs** explains it by far
  best (lstsq slope 0.986, rms 0.088; the mean gets rms 0.59; max/min/any single input are much worse).
- It is a *soft and imperfect* median. The exact median `max(min(x0,x1), min(max(x0,x1),x2))` FAILS:
  max error 0.62 against a snap tolerance of 0.117. The worst points are corners such as (3, 3, -3),
  where the net gives 2.38 instead of 3. The net also compresses toward the domain edge: (3,3,3) -> 2.86.
- It is **not permutation symmetric**. Swapping x1 and x2 changes the output by at most 0.035, but
  swapping x0 with another input changes it by up to 0.35. So x0 is handled differently from the pair
  (x1, x2), and no symmetric formula can pass.
- Symmetric smooth medians (tanh-smoothed min/max, sigmoid "probability of being the middle"
  weights, a soft clip of x0 into [min(x1,x2), max(x1,x2)], plus a cubic squash) all reach only
  0.30 to 0.55 max error.

## What I did

1. **Layer 1.** It has 5 "sharp" neurons (weights about 1 to 2.5, bias about 0). They compare one
   input with a blend of the others: x1 vs x2, and xi vs the mean of the other two. The other 11
   neurons have small weights along the sum direction.
2. **Distillation.** I fitted small sigmoid nets to the net's outputs, using least squares followed by
   Lawson reweighting toward minimax error. A 5 -> 2 net with a linear skip got max error 0.065.
   Its layer-2 units ran in their near-linear regime with huge output weights (±15 to ±165), so in
   effect they computed products of the comparison sigmoids. A 7-neuron 5 -> 2 net without the skip
   also passed (0.092), but its rounding was fragile.
3. **Reformulation.** I made the products explicit: output = linear trend + Σ v_k h_k + Σ q_ij h_i h_j,
   where h_k are the 5 soft comparisons. This model is linear in its coefficients, so I solved it by
   minimax-reweighted least squares (24 terms, max 0.055).
4. **Snapping.**
   - Comparison biases → 0.
   - Comparison weights → the 0.1 grid. Greedy snapping, then ±0.1 coordinate descent: max 0.070.
   - None of the 5 comparisons can be dropped; dropping any one gives 0.32 or more.
   - Greedily pruned the product terms down to 6. Tied the x1 and x2 linear coefficients.
   - Rounded the output coefficients one at a time with refits (linear terms first): x0 → 0.3,
     x1 + x2 → 0.36, h3 → -5, two products → ±3.0, and the rest to 2 decimals.
5. **Result:** `agent.py`. PASS (snap), max |program - net| = 0.083 (tolerance 0.117). It uses 5
   sigmoids and 6 products, versus the 32 sigmoids of the dump.

Reading of the program: it is a mean-like trend (0.3·x0 + 0.36·(x1+x2)) plus corrections from soft
comparisons and their ANDs (products). In effect it is a smooth "pick the middle one" that treats
x0 differently from x1 and x2.

## Heuristics

- **Probe first.** Before refactoring, regress the net against a library of candidates (median, mean,
  min, max, range, products, norms). For 3 inputs, check the median explicitly: it is easy to miss,
  because it is nearly linear on average.
- **Test symmetries numerically.** Check every input permutation and f(-x) = -f(x). An asymmetry larger
  than the snap tolerance rules out every symmetric closed form, and tells you which inputs to treat
  as a group.
- **Sigmoid nets learn soft versions of exact rules.** If the exact rule fails only near corners or
  domain edges, the net's smoothing must be reproduced, not snapped away.
- **Spot fake-linear units.** A layer-2 sigmoid unit with a huge output weight (|w| > 10) and a small
  input range is acting as a linear or quadratic term. Replace it with explicit linear and product
  terms, which are more readable and much less sensitive to rounding.
- **Treat layer 2 as products.** In a 2-hidden-layer sigmoid net, layer 2 mostly builds ANDs of
  layer-1 comparisons. Use the model `c + a·x + Σ v h + Σ q h_i h_j`. Once the h directions are fixed it
  is linear, so fitting, pruning and rounding are cheap and stable.
- **Fit for the worst case.** Use minimax, via Lawson reweighting of least squares, rather than plain
  least squares, because the checker judges max error. An rms fit can have twice the max error.
- **Snap in the right order.**
  - Snap first-layer directions to a grid with linear refits, then coordinate-descend on the grid.
  - Round output coefficients one at a time with refits in between, starting with the coefficients
    of raw inputs (they are multiplied by values up to 3), then the large ones.
  - Rounding everything at once to 2 decimals can add 0.02 to 0.03 of error when terms cancel.
- **Distill from the net's own neurons.** Initialise the small net from the large net's "sharp" layer-1
  neurons (large weights, bias ≈ 0) and drop the small, nearly linear ones. Run several random
  restarts: the result varies a lot (max error 0.065 to 0.7).
