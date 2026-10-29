# Notes

Net: 4 inputs -> 8 tanh units -> 3 scores, argmax. Result: `agent.py`, PASS (snap). It has 0 confident
disagreements on the check set, and also 0 on 14 other input-set seeds (1-18). Every disagreement lies in the
net's unsure zone (score gap < 0.584).

## What I changed and why

1. **Constant unit removed.** Unit 6's pre-activation is 3.5..7.4 over the whole box, so tanh is 0.988..1.0. I
   folded it into the output biases.
2. **Near-parallel units merged.**
   - Units 4 and 5 have nearly the same direction and nearly equal output weights (D0: 3.78 / 3.98), so I
     replaced them with one tanh of the averaged pre-activation.
   - Unit 1 and the *negated* unit 3 are near-parallel too (tanh is odd, so `-tanh(-z)` = `tanh(z)`), and their
     output weights have opposite signs. I merged them the same way.

   This leaves 5 units, still with 0 confident errors.
3. **Scores to a decision list.** I rewrote the scores relative to class 1 (s0 = o0-o1, s2 = o2-o1). Where the net
   is confident, s0 > 0 and s2 > 0 never hold together. So `if s0>0: 0 elif s2>0: 2 else 1` matches, each test is
   an independent binary decision, and each test's scale is free.
4. **Unused units dropped per decision.** I refitted the output weights with an LP (maximise the margin over the
   confident points near the boundary, then verify on all confident points). The class-0 test needs only 3 units
   (small, large, short), and the class-2 test needs only 3 (mid, thin, short). Removing any one of them is
   infeasible.
5. **Hidden weights snapped**, one unit at a time on a 0.1 grid (and 0.2 where it still fit), with the output LP
   re-solved after each snap. For robustness I fitted on a pool of 10 input-set seeds (2.2M confident points)
   rather than only the checker's seed. A version tuned on seed 0 alone failed by 1-4 points on other seeds.
6. **Output coefficients snapped** by a grid search around the LP solution. I picked the fewest-decimals
   solution with 0 errors on the whole pool, then scaled it to integers: `20a - 7b - 5c > 6` and
   `20d + 12e + 4c < -9`.

What was not possible:
- A purely linear rule. The LP is infeasible for class 0 vs rest, for 2 vs 1, and for 3 linear scores.
- A 2-unit tanh model per decision. I trained it from several starts: 4 and 29 errors at best, with sharp,
  ugly weights.
- PASS exact, which would need the full net.

Reading the program: the bounds look like the Iris measurements. Class 0 = small x2/x3 (petal). Classes 1/2 are
split mainly by x3, then x2, with a small x0/x1 correction (unit "short").

## Heuristics

- **Constant units:** check each unit's pre-activation range over the whole input box. A tanh/sigmoid unit with
  |z| > ~3.5 everywhere is a constant; fold it into the bias.
- **Merging candidates:** look for pairs of hidden units with near-parallel weight vectors *and* similar output
  weights. Also compare against the negated vector: for tanh, `-tanh(-z)` = `tanh(z)`, so a unit and a
  sign-flipped unit with opposite output weights merge. Replace a pair by one unit on the averaged pre-activation,
  with the summed output weight. Test with a confident-point mismatch count.
- **Argmax nets:** subtract a reference class's score and check whether the positive regions of the differences
  overlap where the net is confident. If they don't, the argmax becomes a decision list of independent binary
  tests, each with a free scale. That makes snapping much easier.
- **Feasibility by LP:** with the hidden features fixed, refitting the output weights is a linear feasibility
  problem: maximise the margin t subject to sign constraints on the confident points. Use only the points near
  the boundary (gap < a few margins) for speed, then verify on all confident points. The LP also tells which
  units each decision needs (drop a feature, see if t stays > 0), and whether a purely linear rule exists.
- **Snap with re-solve:** snap one unit at a time on a coarse-to-fine grid (0.5, 0.25, 0.2, 0.1), re-solve the
  output LP after each snap, and keep the coarsest grid that is still feasible.
- **Don't overfit the check set:** the checker's input set is fixed, and a rule tuned on it can fail by a few
  points on other random sets. Fit and snap on a pool of several input-set seeds, and verify on held-out seeds.
- **Output snapping:** grid-search the remaining output coefficients (step 0.01) around the LP solution, rank by
  number of decimals then by distance to the LP centre, and scale the winner to integers.
- **Expect thin unsure zones on smooth nets:** with tanh nets the boundaries are curved. If a linear or 2-unit
  rule is infeasible, keep a few soft-step units rather than force a piecewise rule.
