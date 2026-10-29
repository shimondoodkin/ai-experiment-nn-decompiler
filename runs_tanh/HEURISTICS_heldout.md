# Heuristics learned by Claude Code

Appended by `nndc agent collect`, one section per refactoring attempt.

## xor/seed2 — agent_1 (m6b1262, 2026-10-29)

- When all inputs are boolean or small integers and the domain is tiny (here 2^2 points),
  enumerate the dump's output on every point before reading the weights. The truth table is the
  program; the hidden layer is only one way of producing it.
- For binary outputs, look at the sign pattern of the score on the corners and match it to a
  named boolean function (AND, OR, XOR, majority, threshold count). XOR on bools is `a != b`.
- Check the margins: if every corner's score is far from the threshold, a pure boolean
  expression will pass exactly, with no need to keep scale factors.
- A binary program can return a comparison directly (1/0), as long as the threshold is 0 and
  "true" maps to the positive class.
- For tanh/sigmoid nets with large input weights (|w| about 3 to 5 on 0/1 inputs), each unit acts
  close to a step function; but reading unit-by-unit is unnecessary when the input space can be
  enumerated.

## window/seed2 — agent_1 (m398b2a, 2026-10-29)

- Before refactoring, bound the output range over the input box (dense scan for 1-D, or
  interval arithmetic: |w*x| <= |w|*max|x| per unit). If the whole range sits on one side
  of the threshold, the binary net is a constant and the program is `return <constant>`.
- Units whose input weights are tiny relative to the input range (|w| * range << 1) are
  constants: replace tanh(w*x + b) by tanh(b) and fold it into the bias.
- Units with tiny output weights (~1e-7) can be dropped outright.
- A degenerate (collapsed) net, with all weights near zero except biases, usually means
  training failed or the data was one class; check this first, since it ends the work.

## neareq/seed2 — agent_1 (ma36b61, 2026-10-29)

- For tanh/sigmoid nets, sort units by input-weight size. Units with small input weights and large
  |bias| saturate across the domain, so treat them as constants plus a small tilt, not as features.
- When a unit's input weights are nearly equal and opposite (w, -w), it is a function of the
  difference x1 - x0. Several such units with switch points on both sides of 0 and opposite-signed output
  weights form a band or window detector: `|x1 - x0| < c`.
- For binary outputs, scan the sign change of the dump along a few 1-D slices. The crossings give the
  thresholds directly. Then snap them to a round shared value (here ±0.2) and write the score as
  `c - abs(...)`, so the sign carries the decision.
- Expect a few disagreements in the unsure zone after snapping. They are fine for snap, but exact needs the full net.

## absdiff/seed2 — agent_1 (m1af8c6, 2026-10-29)

- Look for mirrored unit pairs (weights w and -w, same bias, same output weight). Their sum is an
  even function of `w.x`: a bump or valley, i.e. a smoothed `|w.x|`. Rewrite them over a shared
  named variable such as `d = x0 - x1`.
- Units whose input weights are tiny relative to the input range (|w| * range << 1) sit in the
  near-linear part of tanh/sigmoid. Fit their combined contribution with a low-order polynomial
  (constant, linear, then quadratic if needed) in the direction they share, instead of keeping them.
- Fit the residual after removing the dominant structure. It tells you which direction the leftover
  units care about (here `x0 + x1`), and how many polynomial terms you need.
- After snapping the coefficients, re-fit only the additive constant to the midpoint of the error
  range (minimax shift). It is free and often halves the max error.
- Snap the parameters of high-gain units (large output weight times steep slope) less aggressively.
  Their error is amplified, so keep two significant digits there and snap the low-gain ones harder.
- Keep a margin: target a max error well below the snap tolerance (here under 1/3 of it), because
  the checker also uses boundary and corner points that a random sample may miss.

## tax/seed2 — agent_1 (m76535b, 2026-10-29)

- For a tanh/sigmoid unit, compute its centre `-b/w` and its width `1/|w|`. Units whose centre
  lies more than ~3 widths outside the input domain are constants on the domain; fold them into
  the bias.
- A unit with negative input weight and negative output weight is a rising step; normalise signs
  before reading the shape (tanh is odd).
- Several same-height steps spread across the domain add up to a smooth ramp; their spacing
  gives the local slope (step height / gap). Denser steps mean a steeper slope, so a curve made
  of unevenly spaced equal steps is a piecewise-linear function in disguise.
- For 1-input regression, tabulate the net and do a least-squares piecewise-linear fit with 1 and
  2 knots first: it shows quickly how many pieces the snap tolerance needs.
- A convex 2-piece fit is `max(line1, line2)`; this is shorter and clearer than `relu` sums.
- When rounding slopes, check the domain ends: the last step is often only half-finished at the
  boundary, so the net flattens there and a "nicer" slope (0.4 vs 0.39) can fail at the edge.
- Grid-search snapped coefficients by max (not mean) error, since the checker uses max error.

## iris/seed4 — agent_1 (m63c5ce, 2026-10-29)

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

## iris/seed5 — agent_1 (m0d4dd2, 2026-10-29)

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

## iris_petal/seed3 — agent_1 (m214154, 2026-10-29)

- For 2-input argmax nets, first print a coarse ASCII class map over the input box; it shows how
  many regions there are, which pairs of classes share a boundary, and whether boundaries look straight.
- Find each pairwise boundary by bisection along one input for a sweep of the other, then fit a
  line; the max residual tells you whether a single linear test can work.
- Express each boundary as `x0 + a*x1 <op> t` (normalize one coefficient to 1), then snap `a` and `t`
  one at a time to round values, re-checking after each; keep the slope more precise than the
  threshold, since slope errors grow across the box while a threshold shift is uniform.
- With nested regions (class 0 | class 1 | class 2 in bands) an if-chain of half-plane tests
  (return the outer classes first, the middle class last) replaces the argmax completely.
- tanh hidden layers with many near-duplicate units (here pairs 1/2, 4/5, 6/7 with near-equal weights)
  usually implement just a few soft cuts; don't simplify unit by unit, read the decision geometry.

## iris_petal/seed4 — agent_1 (mda93be, 2026-10-29)

- For small argmax nets with 2 inputs, print the argmax over a coarse grid first. If the classes
  form ordered bands, the program is just a chain of linear threshold tests, and the hidden units
  do not need to be interpreted one by one.
- Locate boundaries by 1-D scans (fix one input, sweep the other on a fine grid, record class
  changes). Then fit `x_a = slope*x_b + c` for each class pair. The max residual tells you whether a
  straight line is enough.
- Snap slope and intercept to one decimal (or a simple fraction) and let the checker decide.
  When several snaps pass, keep the one with the fewest unsure-zone disagreements. When a boundary is
  curved, the fitted intercept is pulled by the curved end, so try the snapped value on both sides.
- When the classes are ordered bands, write the decision as an ordered if-chain (lowest class first).
  Each test then only needs the one boundary between adjacent classes, and pairs that never touch
  (here 0 and 2) need no test.
- Inputs bounded like x0 in [1, 6.9] and x1 in [0.1, 2.5], with three ordered classes, suggest a
  measurement-based classification. The boundaries have mixed slopes (one steep, one at 45 degrees),
  so keep the per-boundary weights rather than forcing one shared combination.

## iris_petal/seed5 — agent_1 (m29f9c3, 2026-10-29)

- For low-dimensional argmax nets, first print the class map on a grid; the number and shape of
  boundaries tell you the program structure (here: two lines -> an ordered if-chain).
- When classes appear as ordered bands, write the decision as a cascade of threshold tests
  (`if s1 < t1: 0 elif s2 < t2: 1 else 2`) instead of computing all scores and an argmax.
- Find boundaries numerically (scan + bisection along one input at fixed values of the other),
  then fit a line; the residual tells you whether a linear rule is enough.
- Normalize each boundary to coefficient 1 on one input (`x0 + a*x1 < b`) before snapping, so
  only one slope and one threshold need rounding.
- Snap greedily and verify each step: round slope to the nearest simple value, then re-tune the
  threshold; if a snap fails, step back to one decimal place. A slightly curved boundary can still
  snap to a straight line, with the threshold placed to balance the error at both ends.
- Smooth (tanh/sigmoid) hidden layers in small classifiers often just implement near-linear
  boundaries; do not try to simplify unit by unit when the end-to-end decision map is simple.

## median3/seed2 — agent_1 (m544c41, 2026-10-29)

- **Test order statistics early.** For a regression net on a few inputs with equal ranges, compare
  against max, min, median and sum (and their abs versions) before fitting polynomials. A
  correlation near 1 with slope about 1 identifies the core function at once. The median can be
  written in the program language as `x0 + x1 + x2 - min(...) - max(...)`.
- **Look at where the residual is large, not just how large.** Bin the residual by the gaps between
  sorted inputs and by position in the box. Smooth nets (tanh/sigmoid) round off every kink of a
  piecewise-linear target. Near a tie of two order statistics the output is pulled about w/4 toward
  the odd input out, where w is the width of the smoothing. Corners of the input box are where these
  errors are largest.
- **Use the right coordinates for the correction.** For order-statistic targets, use the sorted
  values and the gaps between them. Use (gap_lo, gap_hi) rather than raw inputs: a tanh correction
  in raw x needed more units for a worse fit. If fitted weights on a feature collapse to 0, drop it.
  This gives an invariance (here, shift equivariance).
- **Fit for the worst case.** The check is max-error based. Fit residual units with a soft-minimax
  loss (an Lp loss, raising p over training). Sample extra points on the faces and corners of the
  box and near ties, because that is where the checker's worst points are.
- **Check symmetry before imposing it.** Measure the permutation spread and |f(x) + f(-x)| of the net
  first. If either is a sizeable share of the tolerance, a symmetric or odd correction will plateau
  above the tolerance.
- **Round coefficients last, and check every rounding.** Units that partly cancel, such as two tanh
  units with similar arguments and opposite output weights, are sensitive to rounding. Keep enough
  digits for those.
- **Report the plain formula alongside the correction.** When the clean formula fails only by the
  net's own smoothing error, say so. The correction terms describe the net's imperfections, not the
  task.

## iris/seed3 — agent_1 (m04dbaf, 2026-10-29)

- **Saturated units.** Check every hidden unit's pre-activation range over the whole input box.
  If |z| > 4 everywhere, tanh/sigmoid is a constant. Fold it into the next layer's bias.
- **Argmax nets.** Subtract one output from all the others (fix a reference class at 0). This
  removes one score row and shows which features separate which pair of classes.
- **Merge candidates.** Look for hidden units whose weight vectors have |cosine| > 0.95. Units
  that are nearly parallel, or anti-parallel with a different bias, are redundant.
- **Remove units by re-solving the output layer.** Drop a unit, solve the output layer by least
  squares on the remaining activations, then fine-tune.
  - Rank the units to try by the disagreement count right after the least-squares step.
  - The unit with the fewest disagreements is usually the removable one. It is not always the
    unit with the smallest output weight.
- **Decision-aware refit loss.** Use a small MSE on the score differences plus a squared hinge.
  The hinge asks the target class to beat the others by about half the net's own top-2 gap, and
  only at points where the net is confident. Plain MSE distillation leaves boundary errors that
  the checker counts.
- **Snap one coefficient at a time.**
  - Snap the coefficient closest to the grid first, refit the free ones, and keep the snap only
    if the check still has zero confident disagreements.
  - Joint quantization-aware training (straight-through rounding) on a 0.1 grid oscillated
    badly; one-at-a-time snapping was far more reliable.
  - The last few coefficients often need a finer grid (0.05 / 0.025) because nothing is left free
    to compensate.
- **Re-check on fresh points.** Re-check the final program on input sets with other random seeds,
  to make sure the refit did not overfit the checker's own sample.
- **Expected outcome for small tanh classifiers.** A full symbolic rewrite (linear rules, a
  decision list) does not hold over the whole input box. The pairwise boundaries agree with a
  linear fit on only about 98% of points. Keeping the tanh features and simplifying their
  coefficients is the realistic target.

