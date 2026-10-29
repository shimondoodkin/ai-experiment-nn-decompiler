# Notes

## What the network computes

The 3 -> 16 -> 16 -> 1 tanh network computes the **median of its three inputs**, with
fit noise of about ±0.16 on an output range of about 5.9.

How I found it:
- Symmetry probes. f(perm(x)) is close to f(x) for every permutation, with max deviation 0.13–0.18,
  which is fit noise. f(-x) is close to -f(x) (max deviation 0.09). So the function is symmetric and
  odd, with no dependence on input order.
- Correlation of the output with candidate features: median 0.9997, sum 0.89, min/max 0.58,
  |x_i| and pairwise products about 0. A least-squares fit y = a·median + b gives a = 1.0002 and
  b = 0.0025.

The median alone **fails** the check: 205 inputs beyond tolerance, max error 0.163, tolerance
0.1186. The leftover error has two systematic parts, and I modelled both.

1. **Rounded corners at ties.** The residual y − median, binned by the gap to the nearest
   neighbour, is the same at every median level:
   - upper tie (median ≈ max): −0.073 at gap 0, 0 at gap 0.2, +0.03 at gap 0.5, 0 at gap about 1.2;
   - lower tie: the mirror image.

   So the net rounds the corners of min/max with a smooth (tanh) kink, and the rounded min of the
   top two values undershoots. I kept only the main lobe: `0.375 * relu(0.2 - gap)`, which is 0.075
   at an exact tie and 0 beyond a gap of 0.2. This term alone cuts the check failures from 205 to 54.
2. **Wide-spread wave.** All remaining failures had lo ≈ −3 and hi ≈ +3, with spread > 5, which is
   the sparse corners of the cube. There the residual, as a function of the median, is about +0.07
   at median −1 and about −0.04 at median +1. I modelled it as `relu(spread − 5)` times two triangle
   bumps:
   - +0.12 at median −1;
   - −0.06 at median +1.

   I chose the coefficients with a minimax LP over about 3.3M points (uniform points, edge-biased
   points and the checker's set), then snapped them: 0.1225 → 0.12, −0.059 → −0.06, slope 0.375 = 3/8.
   This is a training artefact, not part of the task.

Result: `PASS (snap)`, max |program − net| = 0.103 against tolerance 0.1186, with 0 failures. It
passes the same way for checker seeds 1–3 (max 0.103–0.104) and for my own 3M-point uniform and
edge-biased sets (max 0.106). The bare median `max(min(x0,x1), min(max(x0,x1), x2))` is the intended
function. The other two terms only make the program follow the net's approximation error closely
enough to pass.

## What I rejected

- Clipping the output to ±2.85 to model saturation at the cube corners. It had no effect once the
  tie term was in.
- A pull toward the mean, `c·(sum − 3·median)`. It made things worse for every c ≥ 0.02.
- Polynomial fits of the residual in (lo, med, hi). Degree 3 still fails, and higher degrees blow
  up. The residual that depends on permutation is as large as the symmetric part, so a symmetric
  smooth fit cannot capture it.
- Gated linear or quadratic terms in the raw inputs. They pass, but they are ugly and fitted to
  noise; the median-bump form is equally good with 2 coefficients.

## Heuristics

- **Check symmetries first.** Compare f with f under each input permutation and each sign flip.
  Then correlate the output with order statistics (min, median, max), the sum, |x_i| and products.
  A correlation of 0.9997 with the median was decisive at once.
- **Order statistics as a basis.** When a function is symmetric under permutation, rewrite it in
  sorted coordinates (lo, med, hi) and the gaps (med − lo, hi − med). Residuals then become 1-D
  curves.
- **Tanh/sigmoid nets that approximate min/max/median/abs have rounded kinks.** Near a tie the
  output is offset by a fixed amount (here 0.075, about 1.3% of the range), and the offset dies out
  within a small gap (here 0.2). It does not depend on location. Bin the residual by the gap to see
  it, and model it with `k*relu(w - gap)`, which is linear in the language.
- **Wide-spread corners are poorly trained.** Points where the inputs sit at opposite extremes of
  the cube are rare under uniform training data, so the biggest errors cluster there. Gate any
  correction with `relu(spread - w)` so that it cannot disturb the rest of the domain.
- **Fit corrections by minimax, then snap.** The snap criterion is a maximum error, not RMS. A
  cutting-plane LP (`scipy.optimize.linprog` on a working set that grows with the worst violators)
  finds a sparse correction with the fewest terms. Least squares left failures where minimax
  passed.
- **Test on more than the checker's set.** Use an edge-biased sample (most coordinates within 10%
  of a bound) and other checker seeds, so that the margin is real and not luck of one sample.
- **Compute in NumPy.** The checker's language evaluator is slow on millions of points. Cache the
  checker's input set (`build_input_set`) and evaluate candidate formulas in NumPy, and use
  `check.py` only to confirm.
