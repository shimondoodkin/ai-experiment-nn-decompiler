# Notes

## What the network does

4 inputs -> 8 sigmoid units -> 3 scores -> argmax. Over the whole input box the net assigns
class 1 to about 60% of points and classes 0 and 2 to about 20% each.

- Unit h1_6 has small weights, and over the box its pre-activation stays between 4.5 and 10.5, so
  sigmoid(h1_6) is between 0.989 and 1. It is a constant and folds into the output biases.
- The other 7 units all point roughly the same way: +x1 (and a little +x0) against -x2 and -x3.
  They come in sign-flipped pairs (h1_0 and h1_1, for example), so the net is close to sorting
  points along a single direction, with two cut points.
- I fitted a linear separator for each class against the rest on the points where the net is
  confident (the checker's own input set, outside the unsure zone). Both were linearly separable,
  but only just: the largest margin was about 0.03 to 0.04 in units of x2.
- To snap the coefficients I used an LP that maximises the margin with some coefficients fixed to
  round values. I kept a snapped choice only while the margin stayed positive.

## Result (`agent.py`)

```
class 0  if  x0 + 6*x1 + 3.1 > 8*(x2 + x3)
class 2  elif 20*x2 + 33*x3 > 8*x0 + 14*x1 + 63
class 1  otherwise
```

- The class-0 cut in normalised form is x2 + x3 < x0/8 + 3/4*x1 + 0.3875.
- The class-2 cut in normalised form is x2 + 1.65*x3 > 0.4*x0 + 0.7*x1 + 3.15.
- The whole network is replaced by two linear inequalities with integer coefficients. There are
  no sigmoids and no hidden units.
- `check.py`: PASS (snap). There are 0 disagreements where the net is confident; the 4573
  disagreements are all in the unsure zone, which covers 2.8% of inputs. With the checker's input
  set re-drawn from seeds 1 to 5 it also passes, with 0 disagreements outside the zone each time.
- It is not exact. The net's boundaries are slightly curved sigmoid surfaces, so no flat cut
  reproduces them exactly.
- Coefficients I could not snap further:
  - Setting the x3 weight of the class-0 cut to exactly 1 still works.
  - x0/10 instead of x0/8 fails.
  - Dropping x0 from either cut fails, so x0 really matters, a little.
  - In the class-2 cut, the x3 weight has to stay between about 1.6 and 1.67.

## Heuristics

- **Sigmoid units with a pre-activation range far from 0 are constants.** Compute the range of
  each pre-activation over the input box. If it stays above about 4.5 (or below -4.5), replace
  the unit by 1 (or 0).
- **Check whether the hidden units share one direction.** If most units have the same sign pattern
  in their input weights, possibly negated, a softmax/argmax net is effectively ordinal along a
  projection. Try one linear cut per extreme class before anything structural.
- **Find each class against the rest with a max-margin LP on the confident points only.** Points
  in the unsure zone may disagree, so leave them out of the fit. A positive LP margin proves that
  a linear rule can pass.
- **Snap by fixing coefficients in the LP one at a time.** Fix a coefficient to a round value,
  re-solve for the rest, and keep it only while the margin stays positive. Then scale the whole
  inequality to clear the denominators and get integer coefficients.
- **Avoid brute-force grid searches over coefficients on 200k+ points.** They are slow, and a
  coarse grid misses thin feasible regions. The LP finds the feasible region directly.
- **Write the cascade in order: extreme class 0, then extreme class 2, then the middle class.**
  Each one-vs-rest cut is valid on its own, so the order of the `if`s does not create errors.
- **Re-check with other input-set seeds when margins are thin.** This confirms that the pass is
  not luck from one sample.
