# Agent notes

## What I did

- The dump is a 1-input regression net: 8 sigmoid units, one linear output, domain x0 in [-1.5, 4].
- Tabulated the net on a grid (scratch/look.py). The curve is monotone, concave, crosses 0 at
  x0 = -1 (net 0.0015), is 0.690 at 0 (ln 2 = 0.693), 1.100 at 1 (ln 3 = 1.099) and 1.788 at 4
  (ln 6 = 1.792). That fits `log(x0 + 2)`.
- Checked the candidate on a dense grid: max |log(x0+2) - net| = 0.031, at the left edge
  x0 = -1.5 (net -0.662 vs ln 0.5 = -0.693). The snap tolerance is 2% of the output range = 0.049.
  The rest of the domain is much closer; the net flattens the steep log near its pole.
- Final program: `return log(x0 + 2)`. `check.py` gives PASS (snap), max error 0.0310, 0 inputs
  beyond tolerance.
- All 8 units collapse: five of them (slopes ~0.49-0.62, staggered biases) together with the
  steeper units 0 and 4 (slopes 1.9 and -3.0, sitting at the left end) are just a
  sum-of-sigmoids approximation of the log curve; the steep ones bend the curve down near x0 = -1.5.
- I did not try for PASS (exact): that needs 1e-5 agreement, i.e. keeping the sigmoid sum itself,
  which is what the formula replaces.

## Heuristics

- For a 1-input net, tabulate the output on a grid first and look at values at integer inputs;
  compare them to familiar constants (ln 2 = 0.693, ln 3 = 1.099, sqrt 2, e, pi) before analysing
  units one by one.
- A zero crossing at an integer x = a with a concave, monotone curve suggests log(x - a + 1);
  check the value one step to the right (ln 2) and further out (ln 3, ln 6) to confirm the shift.
- Many sigmoid units with nearly equal small slopes and staggered biases are a smooth-curve
  approximation (a "sum of soft steps"), not separate features; steep units at the domain edge
  correct curvature near a pole or boundary. Replace the whole layer by the closed-form curve.
- Expect the largest error of a closed-form snap at the domain edge closest to a singularity
  (pole of log or 1/x, kink of sqrt): the net cannot follow the steep part. Compare that edge
  error to the snap tolerance (2% of the output range) before deciding the formula is wrong.
- Measure the candidate's max error on a dense grid yourself before running the checker, so a
  near-miss can be told apart from a wrong formula.
