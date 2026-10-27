# Notes: 4 -> 8 -> 3 argmax network (iris-like)

## What the network is
The input ranges (4.3-7.9, 2.0-4.4, 1.0-6.9, 0.1-2.5) are exactly the iris feature ranges, so I read
x0 = sepal length, x1 = sepal width, x2 = petal length, x3 = petal width, and the classes as
0 = setosa-like (small petals), 1 = versicolor-like, 2 = virginica-like (large petals).
This is only naming; the program was derived from the weights.

## What I changed and why
1. **Dead / negligible neurons removed.** On the checker's input set, neurons 2, 3, 5 and 7 are
   almost never active (0.3%, 7%, 1.3%, 0%) and tiny; zeroing each changes no confident decision.
2. **Always-on neurons linearized.** Neuron 6 is active on 100% of the box and neuron 0 on 99.6%;
   dropping their ReLU changes no confident decision, so they fold into linear terms.
3. **Only two real ReLUs remain** (neurons 1 and 4). With everything else linear, I wrote the three
   pairwise logit differences (o0-o1, o2-o1, o0-o2) as linear forms in x for each on/off pattern
   of those two ReLUs.
4. **Decision list from the main region.** In the region where both ReLUs are on (84% of the box)
   the rule `if d01 > 0: 0 elif d21 > 0: 2 else 1` already matched everything except 8 points.
   The `o0 > o2` test was never needed: class 0 never borders class 2 directly.
5. **Kink in the class-2 boundary.** o2-o1 has a *negative* weight on relu(neuron 4). So, where
   neuron 1 is on, `d21 = min(form with neuron 4 on, form with neuron 4 off)`. That turns
   "class 2" into an AND of two linear tests. Dropping the second test leaves 21-49 confident
   errors, so it stays.
6. **Snapping.** I divided each test by its x2 coefficient so the petal-length weight is exactly 1.
   Then I searched simple fractions (denominators 1-20) near each coefficient and kept the
   simplest set with zero confident errors. I counted errors on 10 pooled checker input sets
   (seeds 0-9, about 2.2M points), not just the one the checker uses. Then I chose biases near
   the middle of the bias range that still passed:
   - class 0: `0.8*x1 + 1.4 > x2 + 1.2*x3 + 0.05*x0`. Setting the x0 coefficient to 0 was simpler,
     but it left a bias window only 0.035 wide and failed on checker seed 3. With -0.05 the window
     is 1.375-1.48.
   - class 2: `x2 + 1.75*x3 - 0.4*x0 - 0.7*x1 > 10/3`. The bias window is 3.29-3.35.
   - class 2 kink: `x2 + 2*x3 - 0.4*x0 - 0.65*x1 > 3.6`. The bias window is 3.54-3.67.

Result: `check.py agent.py` -> PASS (snap), with 0 confident disagreements. Checker seeds 1-5 and
the pooled 10-seed set also give 0. The remaining differences are all inside the net's
unsure zone.

## Heuristics
- First measure how often each hidden unit is active over the real input box. A unit that is
  never on is dead and can be deleted. A unit that is (almost) always on is linear, so drop its
  ReLU and fold it into the next layer.
- Do ablations against the checker's own metric: set a unit to zero or to its mean and count
  *confident* disagreements. Raw weight size is misleading, so use |w_out| x std(activation).
- For argmax outputs, work with pairwise logit differences, not the raw logits. Biases and shared
  terms cancel there, and each difference is a single linear form inside every ReLU pattern.
- Find the dominant activation pattern and fit a decision list from its linear forms. Then add
  extra tests only where the residual errors sit.
- The sign of a difference's weight on a ReLU tells you the shape. A negative weight on relu(z)
  gives `min(form_on, form_off)`, which is an AND of two linear tests. A positive weight gives
  `max`, which is an OR. This turns a kink into a clean boolean condition.
- Check which pairwise tests are actually needed (here o0 vs o2 never mattered), because some
  classes never touch each other.
- Normalize each linear test so one dominant input has coefficient 1 before snapping. Then
  coefficients and bias are comparable, and simple fractions show up.
- Snap with a search ordered by simplicity and require zero confident errors. Then scan the bias
  window and pick a value near its middle, not at its edge.
- Validate on several resamples of the checker's input set. A snap that passes one seed with a
  very narrow bias window is fragile. A slightly less pretty coefficient that widens the window
  is worth keeping.
- Use input ranges to recognise well-known datasets (here iris) and name variables, but derive
  the rules from the weights, not from domain folklore. The checker samples the whole box,
  including off-distribution combinations.
