# Notes

## What the network computes

Approximately the distance between the two inputs, `|x1 - x0|`, rounded off near 0 and flattened at
the far ends of the range.

Final program (`agent.py`, PASS (snap), max error 0.096 against tolerance 0.104):

    dist = abs(x1 - x0)
    if dist < 1:  return 0.25 + 0.75 * dist * dist     # smooth bottom, continuous at dist = 1
    return min(dist, 3 + 0.4 * dist)                    # identity, then slope 0.4 above dist = 5

## How I got there

1. **Grouped hidden units by direction.** All 8 sigmoid units use one of two directions:
   - `x0 + x1` (units 0, 1, 5, 7). Their output weights are small (|w| < 0.3) and they almost cancel.
     Along `x0 + x1` the output changes by only about 0.01, so I replaced them with a constant.
   - `x1 - x0` (units 2, 3, 4, 6). These carry big weights (2 to 3.9) and come in mirrored pairs:
     (2, 6) with slope about 1.2 and offset about 4.6, and (3, 4) with slope about 2.2 and offset about 2.3.
     A mirrored pair `sig(a*d - b) + sig(-a*d - b)` is an even function of `d`, so the output is a
     function of `|d|`.
2. **Profiled the 1-D function** `g(|x1 - x0|)` (min/max over the orthogonal direction). It follows
   `|d|` with wiggles of about ±0.09 (the sigmoid "staircase"), plus two deviations: g(0) ≈ 0.25
   instead of 0, and it saturates above |d| ≈ 5 (g(6) ≈ 5.4).
3. **Minimax piecewise fits** (LP over sampled points) of the form `c + a*t + sum b_k*relu(t - k)`.
   The middle segment came out as exactly `t` (slope 1, intercept 0). A linear segment near 0 could
   not follow the curvature (error 0.10 to 0.19 at t ≈ 0.4 to 0.75). A parabola `0.25 + 0.75*t^2` fits
   the bottom and meets `t` at t = 1 with no jump. For the tail, the line `3 + 0.4*t` meets `t` at t = 5.
4. The remaining error, 0.096, is the sigmoid wiggle at |d| ≈ 4.35 (net is above `|d|`). This is in
   the network itself and cannot be removed by any simple formula of the kind used here.

Removed: all 4 units along `x0 + x1`, the output bias, and all sigmoids.

## Heuristics

- **Group hidden units by weight direction** (normalise each input-weight vector). Units that share a
  direction collapse to a 1-D function of that projection. Here it was two directions, `x0+x1` and
  `x1-x0`.
- **Check the size of each direction's contribution before modelling it.** A direction whose units
  have small output weights that partly cancel can often become a constant. Measure the output
  range along that direction with the other direction held fixed.
- **Mirrored sigmoid pairs** `w1*sig(a*d - b) + w2*sig(-a'*d - b')` with similar parameters make an
  even function of d, so work with `abs(d)`. Several such pairs at increasing offsets approximate
  `|d|` as a staircase.
- **Profile, then fit minimax** (LP with hinge features at round knots) rather than least squares,
  because acceptance is a max-error test. Snap the fitted coefficients, and then look at what the
  pieces reduce to: here the middle piece became exactly `t`.
- **Sigmoid nets that approximate a kinked target** (like abs) round off the kink. Model the rounded
  part with a parabola that meets the linear part continuously (`0.25 + 0.75*t^2` meets `t` at
  t = 1), not with extra line segments.
- **Sigmoid nets saturate at the edges of the input domain.** Expect the far ends of the range to
  flatten, and add a `min(…, a + b*t)` tail segment.
- **Some error cannot be removed.** The leftover error is the staircase wiggle of the summed
  sigmoids. Its size (about 2*w/8 for steps of weight w) sets a floor for any clean formula, so
  check early that it is below the snap tolerance.
