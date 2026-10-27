# Notes

## What I changed and why

- **Dead neurons removed.** Over the input box, hidden neurons 0, 3, 6 and 7 never become positive
  (their preactivation max is -3.7, -2.4, -2.3, -2.0), so they always output 0 and were dropped.
- **Always-on neuron folded in.** Neuron 2 is always positive (min 5.99), so its ReLU does nothing and
  it is just a linear function of the inputs; it was folded into the output scores.
- **Activation patterns per class.** With the 3 remaining ReLUs (1, 4, 5): every confident class-2
  point has all of them active, class 1 has 1 and 5 active (4 on or off), class 0 mostly has 1 and 5
  active with 4 off. Within each pattern the network is linear, so the pairwise score differences
  became plain linear forms:
  - class 0 vs 1 (neuron 4 off): o0 - o1 is proportional to `1.6 - (0.1 x0 - 0.8 x1 + x2 + 1.06 x3)`
  - class 1 vs 2 (all on): o2 - o1 is proportional to `(-0.39 x0 - 0.70 x1 + x2 + 1.80 x3) - 3.45`
- **Decision list.** "class 0 if the first form is below its threshold, else class 2 if the second
  is above its threshold, else class 1" reproduces the net everywhere it is confident.
- **Snapping.** Each test was normalised so the x2 (petal length) coefficient is 1, then the others
  were rounded: 0.1 / -0.8 / 1 / 1.05 with threshold 1.6, and -0.4 / -0.7 / 1 / 1.8 with threshold 3.45.
  Coarser snaps fail: dropping the 0.1 x0 term or using 1.0 for x3 in the first test, or 3.5 / 0.75
  in the second, misclassify confident points. The inputs look like iris measurements
  (sepal length/width, petal length/width) and the classes like setosa / versicolor / virginica.
- Result: `python check.py agent.py` -> PASS (snap), 0 disagreements where the net is confident.
  PASS (exact) is not reachable with two lines, since the net's real boundaries bend at ReLU kinks
  inside the unsure zone.

## Heuristics

- Compute each hidden unit's preactivation range over the input box (random samples or interval
  arithmetic). Units that are never positive are dead and can be deleted; units that are never negative
  are linear and can be folded into the next layer.
- For argmax outputs only differences between scores matter; work with pairwise differences
  o_a - o_b rather than with the raw scores.
- Tabulate ReLU on/off patterns among confidently classified points of each class. If each pairwise
  boundary lives mostly inside one pattern, the net is linear there and the boundary is one linear
  inequality; a decision list of those inequalities often reproduces the whole classifier.
- Normalise each linear test so its dominant coefficient is 1 before snapping; then snap the rest one
  at a time and re-check, because a small coefficient times a wide input range (0.1 * 3.6) can still
  shift the boundary more than the tolerance.
- Measure correctness only on confident points (margin above the checker's unsure-zone threshold),
  and keep the threshold close to the fitted value; adjust it only together with the coefficients.
- Recognise known datasets from the input bounds (here iris) to name variables and classes.
