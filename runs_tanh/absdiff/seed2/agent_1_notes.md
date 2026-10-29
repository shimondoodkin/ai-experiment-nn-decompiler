# Notes

## What the network does
2 inputs in [-3, 3], 8 tanh units, regression output (range about 0.29 .. 5.12).

- Units 3 and 7 are a mirrored pair: `tanh(0.384*(x0-x1) + 0.80)` and `tanh(-0.384*(x0-x1) + 0.80)`,
  both with output weight about -3.73. Their sum is an even function of `d = x0 - x1`, so together
  they form a smooth "valley" in d: the output is a smoothed `|x0 - x1|` (about 0.5 at d = 0,
  about 5.1 at |d| = 6, slope about 1.2 in the middle).
- The other six units (0, 1, 2, 4, 5, 6) have tiny input weights (|w| < 0.1) along roughly the
  `x0 + x1` direction, so on the box they work in the near-linear part of tanh. Together they add
  only a constant plus a weak trend along `s = x0 + x1` (the diagonal drifts from 0.50 to 0.29).

## Final program (`agent.py`)
```
d = x0 - x1;  s = x0 + x1
valley = tanh(0.38*d + 0.8) + tanh(0.8 - 0.38*d)
return 5.475 - 3.75*valley - s/50 - s*s/400
```
PASS (snap), max |program - net| = 0.028 against a tolerance of 0.097.

## How I got there
1. Spotted the mirrored pair (3, 7) by comparing weight vectors: equal magnitudes, opposite signs,
   same bias. Rewrote both in terms of `d`, with averaged slope, bias and output weight.
2. Least-squares fitted the residual (net minus the pair) with constant + linear terms: the
   residual depends almost only on `s`. A linear term in `s` alone left 0.086 max error, which is too
   close to the tolerance. Adding `s^2` brought it down to 0.011.
3. Nonlinear fit of the whole model `c - w*valley(a, b) + k*s + q*s^2`, then snapped to
   a = 0.38, b = 0.8, w = 3.75, k = -1/50, q = -1/400. Then re-centred the constant (the midpoint
   of the min/max error) to c = 5.475, which halved the max error (0.051 -> 0.028).
   Snapping `a` to 3/8 instead of 0.38 nearly doubled the error, so `a` stays at 0.38 (the steep
   pair is the sensitive part, with output weight ~3.75).

An exact pass would need all eight units (no two are exact duplicates), so I aimed for snap.

## Heuristics
- Look for mirrored unit pairs (weights w and -w, same bias, same output weight). Their sum is an
  even function of `w.x`: a bump or valley, i.e. a smoothed `|w.x|`. Rewrite them over a shared
  named variable such as `d = x0 - x1`.
- Units whose input weights are tiny relative to the input range (|w| * range << 1) sit in the
  near-linear part of tanh/sigmoid. Fit their combined contribution with a low-order polynomial
  (constant, linear, then quadratic if needed) in the direction they share, instead of keeping them.
- Fit the residual after removing the dominant structure. It tells you which direction the leftover
  units care about (here `x0 + x1`), and how many polynomial terms you need.
- After snapping the coefficients, re-fit only the additive constant to the midpoint of the error
  range (minimax shift). It is free and often halves the max error.
- Snap the parameters of high-gain units (large output weight times steep slope) less aggressively.
  Their error is amplified, so keep two significant digits there and snap the low-gain ones harder.
- Keep a margin: target a max error well below the snap tolerance (here under 1/3 of it), because
  the checker also uses boundary and corner points that a random sample may miss.
