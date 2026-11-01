# Notes

## What I did

- The net is 1 input -> 8 sigmoid units -> 1 linear output, on x0 in [-2, 2].
- Sampling it on a 0.1 grid gave a symmetric bowl: about 0.004 at 0, about 1.00 at +-1, about 3.97 at +-2.
  Every sampled value is within about 0.04 of x0*x0. It is not exp(x), 2^x or |x|: the values at +-0.5
  (0.245) and +-1.5 (2.246) match x^2.
- The 8 sigmoids are a smooth way to build the parabola. The units with negative slopes cover the left
  branch, the ones with positive slopes cover the right branch, and h1_7 (with a negative output weight)
  flattens the bottom. None of the units maps to a term in the formula on its own, so I replaced them all.
- `agent.py` is `return x0 * x0`. Check: PASS (snap), max |program - net| = 0.036 against a snap
  tolerance of 0.079. The largest error is at the box edges (the net gives 3.964 at x0=2, where x^2 is
  4), where the sigmoids start to saturate.
- I did not aim for PASS (exact). That would need the 8-sigmoid sum itself, which is the dump, not a
  formula.

## Heuristics

- For a 1-input regression net, first tabulate the output on a fine grid over the input box and compare
  it with candidate formulas (x^2, |x|, exp, log, sqrt, 1/x, 2^x) before reading the weights at all.
- Check symmetry first: if f(-x) is close to f(x), look at even functions (x^2, |x|, cosh-like), and use
  ratios such as f(2x)/f(x) (4 means x^2, 2 means |x|) to tell them apart.
- Sigmoid nets that fit a smooth convex curve split the work: units with slopes of one sign build each
  branch, and a unit with a negative output weight shapes the bottom. Don't try to simplify the units one
  by one. Fit the whole curve.
- Expect the largest error at the box edges, where the sigmoid units saturate and the net falls short of
  the true curve (here 3.964 vs 4 at x=2). An edge error like this does not mean the formula is wrong.
- Compare the max error with the snap tolerance (2% of the output range). If the clean formula fits with
  margin, prefer it to a fitted polynomial with odd coefficients.
