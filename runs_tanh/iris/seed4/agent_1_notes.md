# Notes

## What the network does

4 inputs -> 8 tanh units -> 3 scores, argmax. Over the input box:

- Unit 6 is saturated at +1 everywhere (smallest value about 0.99995), so it is a constant bias.
- Classes 0 and 2 never touch. No sampled point lies within score 1.0 of a 0/2 tie, so the
  decision is a two-step list: class 0 against the rest, then class 2 against class 1.
- The class 0 / class 1 boundary depends on units 0 and 2 only (the others are saturated there).
  It is a hyperplane: a linear SVM on confident points separates it with no errors.
- The class 1 / class 2 boundary depends on units 1, 3, 4, 5 and 7 (units 0, 2 and 6 are saturated
  at -1, -1 and +1 there). It is curved. The best plane, fit by minimax, misses by up to 4.4x the
  unsure-zone half-width (about 0.03 in x3). A full quadratic fits. Its quadratic part is close to
  rank 1: one eigenvalue is 0.063, along the direction (-0.54, 0.83, 0.13) in (x0, x1, x2), and the
  rest are 10x smaller. So the boundary is a plane bent by one parabola.

## Result (`agent.py`)

```
if x3 + 5/6*x2 < 0.125*x0 + 0.625*x1 + 0.37:            -> 0
u = x1 - 0.65*x0 + 0.15*x2
if x3 > 2.1 + 0.18*x0 + 0.43*x1 - 0.56*x2 + u*u/24:      -> 2
else                                                     -> 1
```

`check.py` gives PASS (snap): 0 disagreements where the net is confident. 4599 disagreements fall in
the unsure zone, which covers 3.09% of the domain; the dump itself has 1042 there because of float
ties. On 2 x 3M fresh uniform points, the largest net margin at a disagreement was 0.44, under the
0.486 zone limit. I did not reach PASS (exact). The net's boundaries are smooth tanh curves, so any
short formula differs from it in a thin band.

## How I got there

1. Sampled 1M uniform points. For each, recorded the class and margin (top1 - top2), and checked
   which units are saturated, both overall and near each pairwise boundary.
2. Bisected along x3 to get exact boundary points: about 127k for 0/1 and about 105k for 1/2. At
   each point I computed the zone half-width in x3, as zone margin / (d score / d x3).
3. Fit the boundary as `x3 = f(x0, x1, x2)` by minimax linear programming on the residual divided by
   that half-width. A form passes when the max is below 1.
4. 0/1: the plane fits with a max of 0.49. Snapped coefficients (1/8, 5/8, -5/6) keep the max at
   0.86. Integer-coefficient searches were worse (best 0.77, with x3 coefficient 17), so I kept the
   simple fractions.
5. 1/2: plane 4.4, rank-1 parabola in (x0, x1) only 1.3, rank-1 parabola in (x0, x1, x2) 0.53
   (a = 0.65, d = 0.15). Adding x2^2 did not help (its coefficient fit to about 0). Coefficients
   snapped to 2 digits and 1/24 give about 0.85-0.94.

## Heuristics

- **Check saturation first.** A unit that is constant over the whole box (here |tanh| > 0.9999) is
  a bias; fold it in.
- **Analyze each pairwise boundary separately.** Look for class pairs that never become top-2
  together (no point near their tie). That turns an argmax into a short decision list.
- **Find the active units at each boundary.** Restrict to points near a boundary and list which
  hidden units are unsaturated there. Each boundary usually involves only a subset of units, which
  tells you how complex it can be.
- **Measure the error budget in input units.** Bisect onto the boundary along one input. Divide the
  zone margin by the score gradient along that input to get the allowed error per boundary point.
  Then fit `x_k = f(others)` by minimax LP on residual / half-width. A max below 1 means it will pass.
  This is far more predictive than counting misclassified samples.
- **Read curvature off the Hessian.** If a plane is not enough, fit a full quadratic and
  eigen-decompose its quadratic part. A dominant eigenvalue means "plane + c*u^2" with u along that
  eigenvector: a few readable terms instead of 10. Grid-search u's direction and solve the rest by
  LP.
- **Check snaps with the same minimax metric.** Try simple fractions first. Integer-coefficient
  searches over small multiples rarely beat 2-digit fractions when the margin budget is tight.
- **Check the dump first.** Even the exact dump can disagree in the unsure zone (float32 ties), so
  only confident disagreements matter.
