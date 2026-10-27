# Notes

## What the network computes
The 3 -> 16 -> 16 -> 1 ReLU net is an approximation of the **median of the three inputs**.
`agent.py` replaces all 32 hidden neurons with the exact closed form

    median(x0, x1, x2) = max(min(x0, x1), min(max(x0, x1), x2))

`check.py` result: PASS (snap), max |program - net| = 0.109 against a snap tolerance of 0.120.
The remaining error is the net's own approximation error. It is largest near the corners of the
input box, e.g. (3, 3, -3), where the net gives 2.89 instead of 3. A program that matches exactly
would have to copy that error, so I kept the clean formula.

## How it was found
- Layer 1 weights come in near +/- pairs on input differences: x2-x0 (neurons 3, 8, 14 and 4 with
  the opposite sign), x0-x1 (0) and x2-x1 (1). Other neurons follow the sum x0+x1+x2 with both signs
  (7/12, 10/11). Pairwise differences plus a sum point to an order statistic or another symmetric
  function of the inputs.
- I sampled the real network (via `weights.npz`) on 20k uniform points. Then I fit `a*f(X)+b` for a
  list of candidate functions: max, min, median, sum, mean, range, products, norms and a linear model.
  Median gave slope 0.998 and intercept -0.016, with RMS residual 0.008. Every other candidate had
  residuals above 0.5, so median was the clear winner. With the slope snapped to 1 and the intercept
  to 0, the formula passes.

## Heuristics
- Look at layer 1 before anything else. Neurons with weights like (+a, -a) on two inputs compute
  differences or comparisons, and paired neurons with opposite signs (relu(z) and relu(-z)) build
  |z| or a z = relu(z) - relu(-z) identity. These patterns suggest sorting, min/max or abs.
- Don't simplify the network weight by weight. Treat it as a black box: sample it and fit a library
  of simple candidates (linear, min/max/median, abs, products, norms, range) with a free scale and
  offset. A slope near 1 and an offset near 0 then confirm the candidate.
- Compare the residual against the checker's tolerance (here 2% of the output range). A small net
  approximating a piecewise-linear target makes its largest errors at the corners and edges of the
  input box. Expect the worst-case error there, and don't add correction terms only to imitate
  those errors.
- Write order statistics with nested min/max. For example, median3 = max(min(a,b), min(max(a,b),c)).
- Snap any fitted scale or offset to 1 or 0 when the result stays inside tolerance. The clean formula
  is the intended function, and the remaining gap is the network's training error.
