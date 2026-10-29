# Notes

## What the network computes

The net is a 3 → 16 → 16 → 1 tanh regression net on x0, x1, x2 ∈ [-3, 3]. It approximates the
**median of the three inputs**.

- A least-squares fit to linear and quadratic terms gives about 0.4·(x0+x1+x2), with error up to 1.6.
- Candidate symmetric functions (max, min, sum, median, max|x|) were compared against the net. The
  median has correlation 0.9997, slope 1.003 and bias 0.007.
- Plain `mid = x0 + x1 + x2 - min - max` FAILs the check: max error 0.25, against a snap tolerance
  of 0.119 (output range about 6). About 0.3% of points are outside the tolerance.

## Where the net departs from the exact median

- **Ties.** Where two inputs nearly tie, the median has a kink. The net rounds the kink off, which
  pulls the output about 0.1 toward the third input. For example, at x0 = x1 = t with x2 = -3 the net
  gives about t - 0.1.
- **Corners.** The pull grows to about 0.25 at the corners, such as (3, 3, -3) → 2.75.
- **Mean.** There is a slight overall pull toward the mean.
- **Asymmetry.** The net is not exactly symmetric under permuting the inputs: outputs over the 6
  permutations of one point spread by up to 0.19. It is not exactly odd either: |f(x) + f(-x)| is
  up to 0.07.

## What I did

1. Wrote the program as `lo = min`, `hi = max`, `mid = sum - lo - hi`, with
   `gap_lo = mid - lo` and `gap_hi = hi - mid`.
2. Fitted the residual net - median with a few tanh units. To fit for the worst case, I used Adam
   with an Lp loss and stepped p up through 2, 6 and 12. Training points were uniform samples plus
   extra samples on the faces and corners of the box and near ties.
3. Compared feature sets for the correction:
   - raw x only: max error 0.18–0.22 with 2–4 units;
   - raw x plus sorted values: 0.10 with 6 units;
   - (gap_lo, gap_hi, mid): 0.099 with 5 units;
   - (gap_lo, gap_hi) only: 0.10 with 5 units. The fitted weights on `mid` went to about 0, so the
     correction depends only on the two gaps, which makes it a function of the gaps alone (shift
     equivariant: adding c to every input adds c to the output).
   - Separable models f(gap_lo) - g(gap_hi) plus a skew term (gap_hi - gap_lo) plateau at about
     0.113. The gaps interact.
   - An odd-symmetric correction plateaus at about 0.15: the net's asymmetry is too large.
4. Final program: median plus 5 tanh units in (gap_lo, gap_hi). Coefficients are rounded to 3
   decimals. At 2 decimals the max error reached 0.1204, just over the 0.119 tolerance, because units
   t1 and t2 partly cancel.

Result: `PASS (snap)`, max |program - net| = 0.0949 against tolerance 0.119. On 5M extra
uniform, face and grid points (my own stress set) the worst error is 0.103.

`PASS (exact)` is not realistic short of re-emitting the net: its departures from the median are
asymmetric, unit-scale "noise" from training.

## Heuristics

- **Test order statistics early.** For a regression net on a few inputs with equal ranges, compare
  against max, min, median and sum (and their abs versions) before fitting polynomials. A
  correlation near 1 with slope about 1 identifies the core function at once. The median can be
  written in the program language as `x0 + x1 + x2 - min(...) - max(...)`.
- **Look at where the residual is large, not just how large.** Bin the residual by the gaps between
  sorted inputs and by position in the box. Smooth nets (tanh/sigmoid) round off every kink of a
  piecewise-linear target. Near a tie of two order statistics the output is pulled about w/4 toward
  the odd input out, where w is the width of the smoothing. Corners of the input box are where these
  errors are largest.
- **Use the right coordinates for the correction.** For order-statistic targets, use the sorted
  values and the gaps between them. Use (gap_lo, gap_hi) rather than raw inputs: a tanh correction
  in raw x needed more units for a worse fit. If fitted weights on a feature collapse to 0, drop it.
  This gives an invariance (here, shift equivariance).
- **Fit for the worst case.** The check is max-error based. Fit residual units with a soft-minimax
  loss (an Lp loss, raising p over training). Sample extra points on the faces and corners of the
  box and near ties, because that is where the checker's worst points are.
- **Check symmetry before imposing it.** Measure the permutation spread and |f(x) + f(-x)| of the net
  first. If either is a sizeable share of the tolerance, a symmetric or odd correction will plateau
  above the tolerance.
- **Round coefficients last, and check every rounding.** Units that partly cancel, such as two tanh
  units with similar arguments and opposite output weights, are sensitive to rounding. Keep enough
  digits for those.
- **Report the plain formula alongside the correction.** When the clean formula fails only by the
  net's own smoothing error, say so. The correction terms describe the net's imperfections, not the
  task.
