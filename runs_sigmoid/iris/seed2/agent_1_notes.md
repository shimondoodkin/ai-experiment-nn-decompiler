# Notes

## What the network does
4 float inputs, 8 sigmoid hidden units, 3-class argmax. All 8 hidden units look along nearly
the same direction: about `-x1 + x2 + x3` (x2 and x3 positive, x1 negative, x0 small). Units 5-7 are
the same direction with the opposite sign. So the net is close to a one-dimensional ordinal classifier:
small values of (x2 + x3 - x1) give class 0, middle values give class 1, large values give class 2. x0 and x1
change the slope a bit between the two cuts.

No confident point has class 0 and class 2 as its top two outputs, so the classes form a chain
0 | 1 | 2, and two cuts are enough: "0 vs the rest" and then "1 vs 2".

## What I did
1. Sampled the checker's own input set (uniform + corners + boundary points) and computed the net's
   three outputs.
2. For each pair (0-1, 1-2), fitted a linear model of `o_a - o_b` with least squares, using only points
   where |o_a - o_b| < 1..4 (near the boundary). The fits were stable across window sizes. The plain
   fitted tree `0 if L01 > 0 elif L12 > 0: 1 else 2` already had 0 disagreements outside the unsure
   zone, so the sigmoids' curvature does not matter at this tolerance.
3. Snapped the coefficients with a vectorised grid search per boundary. For each candidate I counted
   confident points (outside the unsure zone) that land on the wrong side, using only points near the
   fitted boundary. I tried several normalisations (x2 coefficient = 1, x0 = 1, x3 = 2.5) and kept
   the one with the simplest numbers and 0 errors:
   - 0 vs rest: `x1 - x2 - 1.1*x3 - x0/40 + 0.55 > 0`. Dropping the x0 term costs about 13 errors at best,
     so the small x0 weight is real. The feasible bias range for this form is about 0.51-0.56.
   - 1 vs 2: `0.6*x0 + x1 - 1.5*x2 - 2.5*x3 + 4.85 > 0`. This is tight: bias 4.8 or 4.9 already gives
     20-30 confident errors.
4. Result: `PASS (snap)`, 0 confident disagreements. Checker seeds 1-5 (fresh random sets) also gave 0.
   `PASS (exact)` would need the full sigmoid net, so I did not try for it.

The inputs' ranges look like Iris measurements (x0, x1 sepal length and width; x2, x3 petal length
and width). Class 0 is "small x2/x3", class 2 is "large x2/x3", but I kept the neutral names.

## Heuristics
- When all first-layer rows of a small net point in nearly the same direction (up to sign), the net is
  an ordinal 1-D classifier. Look for a chain of classes and write it as an if/elif chain of linear cuts,
  not as an argmax.
- Check which class pairs are actually adjacent. Count points where (top1, top2) is each pair near
  the decision boundary. Pairs that never meet need no rule.
- Get each boundary by least-squares fitting the output difference `o_a - o_b` on points near zero
  (|diff| < a few units), not over the whole domain. Saturated sigmoids far away spoil a global fit.
- Snap per boundary, not jointly. Restrict the evaluation to confident points near the fitted boundary,
  and evaluate all candidates at once with a matrix product. A naive full-set loop over a 5-D grid is
  far too slow.
- Try several normalisations (divide by different coefficients) before snapping. One of them often
  gives much rounder numbers (here x3 = 2.5, x2 = 1.5 gave 0.6 / 1 / 1.5 / 2.5).
- The unsure-zone width divided by the gradient of the score difference gives how precisely each
  coefficient must be snapped. For sigmoid argmax nets with steep outputs, this allows only about
  0.01-0.05 of slack, so "nice" values may need two decimals.
- Re-run the checker logic with other random seeds to make sure the snapped rule did not overfit one
  input set.
