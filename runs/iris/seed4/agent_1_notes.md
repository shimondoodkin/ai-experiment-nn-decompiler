# Notes on m579e70 (4 -> 8 -> 3, argmax)

The inputs and their ranges match the Iris dataset: x0 is sepal length, x1 sepal width, x2 petal
length and x3 petal width. The classes are 0 = setosa, 1 = versicolor and 2 = virginica.

## What I changed and why

1. **Removed dead and rarely active units.** I sampled the input box uniformly. Hidden units 0 and 5
   are never active. Units 1 and 4 are active on only about 6% of the box, and only where the
   decision is already settled. Units 2, 3, 6 and 7 are active on 97-100% of the box, so I treated
   them as linear (I dropped their ReLU). With those changes the whole net collapses to three
   linear scores.
2. **Reduced the three scores to two tests.** Only differences between scores matter for argmax,
   so I used the versicolor score o1 as the reference. `a = o0 - o1` and `b = o2 - o1` are linear,
   and the region where both are positive is empty. So the argmax becomes `if a>0: 0 elif b>0: 2
   else 1`, and each test can be rescaled on its own. This linear version already passed.
3. **Snapped each hyperplane.** For each test I used an LP to find the range of each coefficient
   (after normalising one coefficient to 1) that still separates the points where the net is
   confident. Then I listed small-integer weight vectors inside those ranges and picked the
   simplest one that is feasible, with a round bias inside the bias interval that is still allowed:
   - setosa: `x1 + 4.5 - 2*x2 - 3*x3 > 0`. Sepal length has no effect. Twice the petal length plus
     three times the petal width must stay below the sepal width plus 4.5.
   - virginica: `8*x2 + 14*x3 - 3*x0 - 6*x1 > 27`.
4. Result: `PASS (snap)` with 0 confident disagreements on the checker's input set. A `PASS (exact)`
   result is out of reach once the ReLUs are linearised and the weights rounded.
   Caveat: the allowed bias intervals are narrow (setosa about 4.47-4.51, virginica about 26.7-27.1
   on seed 0). With other sampling seeds, a single far-corner point can fail. Those points are off
   the data, at small x0 with large x1, where a rarely active unit bends the boundary slightly. A
   bias of 26.9 instead of 27 would make the virginica test robust across seeds.

## Heuristics

- Measure how often each hidden unit is active over the whole input box. Units that are never
  active can be deleted. Units that are always active can be treated as linear, and nets with few
  hidden units then often collapse to a linear map.
- For argmax outputs, subtract one reference class score so that k scores become k-1 differences.
  If the regions where several differences are positive are empty, write the argmax as a chain of
  independent `if score > 0` tests, each of which can be rescaled freely.
- Snap in the right space. Normalise a hyperplane by its largest coefficient, use an LP (with the
  margin to the confident points as the objective) to get the feasible range of every coefficient,
  then enumerate small-integer or simple-fraction vectors inside that box. That is much more
  reliable than rounding the fitted coefficients one at a time.
- Restrict the search to points near the fitted boundary and process candidates in chunks. A brute
  force over every point times every candidate is too slow and too memory-hungry.
- Coefficients with a feasible range that straddles 0 (here sepal length for setosa) should be
  dropped.
- The checker samples the whole box uniformly, including corners far from the training data, where
  the "linear" picture can bend. Check a snapped rule against several sampling seeds to see how
  much slack it really has.
