# Notes: 3 -> 16 -> 16 -> 1 regression network

## Result
The network computes the **median of three inputs**:

    median(x0, x1, x2) = x0 + x1 + x2 - max(x0, x1, x2) - min(x0, x1, x2)

`python check.py agent.py` gives `PASS (snap)`. The largest error is 0.092 against a snap tolerance of 0.120.
The worst errors sit near the edges of the input box (inputs near +-3) and where two inputs are almost equal.
At those points the trained ReLU net is a rough piecewise-linear fit. It does not follow a different rule there.

## What I changed and why
- I replaced all 32 hidden ReLU units and the output layer with one closed-form expression. The network
  weights don't snap cleanly one by one: 32 noisy units together approximate a single sorting operation.
  So I found the function from what the network outputs, not by reading its weights.
- How I found it: I sampled 20k uniform inputs and computed a least-squares fit, plus correlations, against
  a set of candidate features (linear terms, products, abs values, pairwise max and |xi - xj|, and the
  max, min and median of all three). The median correlated at r = 0.99999. A linear fit
  `y ~ 1.0007 * median - 0.001` confirmed it: slope 1, intercept 0, so both snap away.
- I wrote the median as sum - max - min so it fits the allowed language (no sort, no median function).
- Exact pass is not attainable in a meaningful way. Matching to 1e-5 would mean copying the network's
  approximation error, which is the noise this task asks me to remove.

## Heuristics
- Find out what the network computes before you try to simplify its weights. For a small number of
  inputs, sample the input box and regress the output on a set of candidate features: linear terms,
  products, abs, pairwise max/min, |xi - xj|, and order statistics (max, min, median). A correlation
  near 1 with one feature tells you what the network computes.
- Symmetric behaviour (equal coefficients on every input, equal pairwise max terms) suggests a
  permutation-invariant function: sum, mean, max, min, median, count of positives, and so on.
- Hidden-layer weights that come in near-opposite pairs (for example +0.73*x1 - 0.73*x2, or
  -0.78*x0 + 0.77*x1) are pairwise comparisons `xi - xj`. Those are the building blocks of max, min and sorting.
- After you name a candidate, fit `y ~ a*f + b`. Then snap a and b to simple values (here 1 and 0) and
  check the largest error against the tolerance.
- Rewrite order statistics with the allowed primitives. For 3 inputs: median = sum - max - min. In general,
  the k-th order statistic can be built from nested min/max.
- In regression tasks the remaining errors gather at the edges of the input domain and on ties
  (xi close to xj), where a small ReLU net can only approximate a kink. If the errors are confined to
  those places, it's approximation noise and not a sign that the formula is wrong.
