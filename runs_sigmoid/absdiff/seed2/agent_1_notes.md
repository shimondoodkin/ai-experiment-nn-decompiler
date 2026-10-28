# Notes

## What the network computes

The net is a 2 -> 8 -> 1 sigmoid MLP over x0, x1 in [-3, 3], with regression output in about
[0.24, 5.40]. On a grid it looks like a smoothed `|x1 - x0|`:
- It is about 0.3 on the diagonal.
- It is close to |d| for 1 <= |d| <= 4, where d = x1 - x0.
- It saturates near 5.2 to 5.4 at the far corners.

The output depends almost only on d. At fixed d it varies by at most about 0.17 (peak to peak),
so the best possible pure f(d) has a max error of about 0.086. The snap tolerance is 0.103.

## Changes

1. **Grouped the hidden units by input direction.**
   - Four units have weights of the form (-a, +a) on (x0, x1), so they are functions of d:
     - neurons 1, 4, 5 and 6;
     - these carry almost all of the output (output weights 1.7, 5.5, 2.8 and 1.7).
   - The other four have small output weights (|w| <= 0.33, each contributing at most 0.33).
     Two of them (neurons 0 and 2) are functions of x0 + x1.
2. **Rewrote the four d-units on the shared variable `d`, with snapped slopes and biases.**
   - The left side (d < 0) is one wide ramp, `sigmoid(-0.8 d - 1.9)`.
   - The right side (d > 0) is three staggered steps:
     - `sigmoid(3.5 d - 3)`
     - `sigmoid(2 d - 4.4)`
     - `sigmoid(1.4 d - 5.7)`
3. **Replaced the four small fixture units** with one symmetric pair on `total = x0 + x1`:
   `sigmoid(2 - 2 total)` and `sigmoid(2 + 2 total)`.
   - Removing them entirely leaves a max error of about 0.094 to 0.097, which passes but with almost
     no margin.
   - Neurons 3 and 7 (mixed directions) gave less improvement than the x0 + x1 pair, so I dropped them.
4. **Refit the output weights and bias by minimax** (an LP over a 121x121 grid) with the snapped
   hidden units fixed, then rounded them to 2 or 3 significant digits.
   - A full nonlinear minimax refit of every parameter only reached 0.074, so the snapped form loses
     very little.
5. **Result:** 6 sigmoids instead of 8, and two named intermediate variables.
   - `check.py`: PASS (snap), max |program - net| = 0.084, against a tolerance of 0.103.
   - PASS (exact) is not reachable without keeping all eight original units at full precision.

## Heuristics

- **Look for unit directions first.** A first-layer unit with weights (a, -a) or (a, a) is a
  function of the difference or the sum. Rewrite it on a shared `d = x1 - x0` or `s = x0 + x1`,
  which collapses a 2-D net to a 1-D function plus small corrections.
- **Measure the error floor before snapping.** Bin the net's output by the candidate variable (d)
  and take half the spread in each bin. That is the best any pure f(d) can do. Compare it with the
  snap tolerance: if the floor is close to the tolerance, keep a small correction term in the other
  direction.
- **Rank units by output-weight times activation range.** Units contributing under about 0.3 of a
  range of 5 are candidates to drop or merge, but drop them together and re-measure, since together
  they can matter.
- **Sigmoid nets that imitate |d| use a pattern:** one wide sigmoid on one side and several
  staggered steps on the other. The steps' centres (-b/k) are spaced about one unit apart, and
  their output weights add up to the saturation height. A single `abs()` will not pass when the
  ends saturate. Keep the sigmoid sum and snap k and b to one decimal place.
- **Snap the hidden layer, then refit the output layer.** Snap the hidden slopes and biases, then
  solve for the output weights by minimax (LP), not least squares. The checker uses the max error,
  and least squares left 0.12 where minimax gave 0.097 for the same features.
- **Get a lower bound to know when to stop.** A quick joint nonlinear minimax optimisation shows
  how much any further snapping or tuning could still gain.
