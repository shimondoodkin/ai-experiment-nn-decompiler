# Notes

## What the net is

A 4 -> 8 tanh -> 3 argmax classifier. The input bounds (4.3-7.9, 2.0-4.4, 1.0-6.9, 0.1-2.5) match the
Iris measurements: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width. This is a guess from
the ranges only. Over the box the net predicts class 0 for 22% of points, class 1 for 59% and class 2 for
19%. Class 0 means small petals relative to the sepals and class 2 means large petals.

## What I changed

- **Dropped the tanh layer.** Refitting the output layer with one hidden unit left out still gave
  100 to 3700 confident errors for 7 of the 8 units. So a smaller tanh net would not have read much
  better.
- **Rewrote the net as a decision list.** It checks class 2 first, then class 0, else returns 1.
  - Class 2 vs {0,1} and class 0 vs 2 are clean.
  - Class 0 vs 1 and class 2 vs 1 are *not* linearly separable on the net's confident points (an LP
    gave a negative margin). One half-plane is not enough for either boundary; each one is curved.
- **Fitted each boundary as an AND of two half-planes.** I alternated LPs: give each negative point to
  the half-plane that rejects it most, then solve a max-margin LP per half-plane. Confident points are
  those with top-2 gap ≥ 0.636; points in the unsure zone are free.
  - Two half-planes are enough for both boundaries.
  - A third half-plane barely raised the margin (0.02 → 0.03 in x3 units), so the curvature is gentle
    and spread out.
- **Snapped the coefficients by search.** For each half-plane I made the coefficient of one input 1 and
  put the others on a grid (steps 1, 1/2, 1/3, 1/4, 1/5), within 5° of the LP direction. I kept the pairs
  that still separate every confident point, plus the band of feasible thresholds. I then scaled to
  integer coefficients and put each threshold mid-band.
  - The margins are thin (about 0.01-0.03 in units of x3), so fully "nice" coefficients like
    `x2 + x3 > c` fail. The coefficients that survived are small integers (up to 24).
- **Checked robustness.** I fitted on the checker's input sets for seeds 0-3 together. Seeds 4-7 were
  only used to choose between the remaining candidates, and the final program has 0 confident
  disagreements on all of them. One class-2 pair that passed seeds 0-3 failed on seeds 4-6 (1-3 points),
  and I dropped it.

Result: `PASS (snap)`, with 0 disagreements where the net is confident. About 2% of inputs differ, all in
the unsure zone, all of them points next to a curved boundary.

```
class 2 if 12*x2 + 24*x3 - 3*x0 - 11*x1 > 50.4 and 11*x2 + 19*x3 - 5*x0 - 6*x1 > 38.2
class 0 if 6*(x2 + x3) - 2*x0 - 3*x1 < 0.7  and 17*x2 + 19*x3 - 4*x0 - 11*x1 < 3.36
else class 1
```

How to read it:
- **Class 0:** the petal size (x2 + x3, weighted by 6) is small compared with the sepal size
  (2*x0 + 3*x1).
- **Class 2:** petal size, with petal width weighted about twice as much as length, is large compared
  with the sepal size.
- **Sepals push against both:** for the same petals, larger sepals push toward class 1 (from class 2)
  and toward class 0 (from class 1).

## Heuristics

- For argmax nets on continuous boxes, first find out which class pairs share a boundary. Then fit a
  decision list: the class that is cleanly separable from the others goes last or first, and each test
  only has to separate the classes that are still left.
- Fit each boundary only on points outside the checker's unsure zone, using the checker's own
  definition: top-2 gap ≥ margin_frac × p95. Points inside the zone are free, and that slack is what
  makes a simple rule possible.
- Test linear separability exactly, with an LP that maximises the margin and fixes one coefficient to
  ±1. Logistic or SVM fits can hide a separator that exists. If the LP margin is negative, the boundary
  is curved.
- For a curved boundary, try an AND/OR of 2 half-planes, fitted by alternating LPs: assign each
  negative point to the half-plane that rejects it most, then re-solve. Initialise by splitting on the
  coordinate where a split makes both halves separable (testing splits at quantiles of each input
  finds it quickly). Adding a 3rd piece rarely pays off.
- Snap with a feasibility search instead of rounding. Normalise by each input in turn, put the other
  coefficients on 1, 1/2, 1/3, 1/4, 1/5 grids around the LP direction, and keep the ones that still
  separate. The threshold then has a feasible interval, so choose a round value inside it. Then
  multiply through to get integer coefficients.
- Bounded inputs with tanh/sigmoid units: the margins across the whole box can be thin (about 1% of an
  input's range), much thinner than on real data. Check the final rule on several fresh input sets
  (other seeds of the checker's sampler), and choose between snapped candidates by their worst case
  on sets they were not fitted on.
- Recognising the dataset from the input bounds helps with naming, but the rule must still match the
  net everywhere in the box, including unrealistic corners (for example a tiny sepal with a huge petal
  width). Most of the curvature that blocks a single-line rule was in those corners.
