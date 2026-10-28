# Heuristics learned by Claude Code

Appended by `nndc agent collect`, one section per refactoring attempt.

## iris_petal/seed3 — agent_1 (m2f9338, 2026-10-28)

- For low-dimensional argmax nets (especially sigmoid ones), print the class map as ASCII on a grid
  first; it shows the region structure (bands, corners, which classes touch) at once.
- If the class regions are ordered bands, write the program as a chain of cuts (`if s1 < t1: return 0;
  if s2 > t2: return 2; return 1`). You only need one cut per adjacent pair, not a full argmax.
- Locate each boundary with a fine 1D scan per row, fit a line, and normalise the coefficient of one
  input to 1 (x0 + a*x1 vs t). Small coefficients and thresholds are then easy to snap.
- Snap one boundary at a time and measure its mismatch rate separately on random points. Prefer the
  simplest value whose mismatch is close to the best fit. Quarter steps (4.25) can beat a nearby
  integer by several times.
- Sigmoid nets give smoothly curved boundaries, so straight cuts usually reach only a snap pass. The
  curvature is largest where several hidden units switch at once, which is where the remaining
  mismatches are.

## iris_petal/seed4 — agent_1 (m11555d, 2026-10-28)

- For low-dimensional argmax nets, print the class map on a coarse grid first. The topology
  (bands, wedges, islands, which classes touch) suggests the program's shape before any algebra.
- When classes form ordered bands, write a chain of threshold tests in that order
  (`if s0 < t0: 0 elif s1 > t1: 2 else 1`). Only adjacent classes need a boundary.
- Recover each boundary directly: bisect along one input at many values of the other, then fit a
  line. This is more reliable than reading slopes off the hidden-unit weights. Here several hidden
  units were almost parallel and combined into each boundary.
- Snap the slope first and the intercept second. Then clear denominators to get integer
  coefficients (x0 + 1.6*x1 < 3.6 becomes 5*x0 + 8*x1 < 18).
- Sigmoid nets give slightly curved boundaries. A straight line usually stays inside the unsure
  zone, but check the ends of the input range, where the curvature errors add up (failures showed
  up at the corners x1 ~ 0.1 and x1 ~ 2.5).
- Try a few candidate snaps against the checker in one loop. Keep the simplest one that passes,
  and use the number of unsure-zone disagreements to break ties.

## neareq/seed2 — agent_1 (m696176, 2026-10-28)

- Look for hidden units whose input weights are equal and opposite (w0 ~ -w1): they depend only on
  the difference x0 - x1 (equal weights: the sum). Rotate coordinates to (difference, sum) and scan.
- For a sigmoid unit `sigmoid(a*z + b)`, the step location is `z = -b/a` and its sharpness is |a|;
  units with |a| * input-range << 4 are nearly linear/constant and can usually be dropped for a
  binary decision.
- A pair of opposite-signed steep steps on the same direction with big output weights forms a band
  (window) detector; read off the two step locations and the output sign to get `lo < z < hi`.
- Locate the actual decision boundary numerically (scan along one direction at several offsets of
  the orthogonal direction) instead of trusting single-unit thresholds; the combined boundary can
  sit away from any one unit's step.
- If the band is symmetric around 0, write it as `abs(z) < c`; return `c - abs(z)` as a signed margin.
- Snap the boundary to a round number (0.2 here) if the measured drift across the domain is
  smaller than the net's unsure zone.

## absdiff/seed2 — agent_1 (m2a2bb9, 2026-10-28)

- **Look for unit directions first.** A first-layer unit with weights (a, -a) or (a, a) is a
  function of the difference or the sum. Rewrite it on a shared `d = x1 - x0` or `s = x0 + x1`,
  which collapses a 2-D net to a 1-D function plus small corrections.
- **Measure the error floor before snapping.** Bin the net's output by the candidate variable (d)
  and take half the spread in each bin. That is the best any pure f(d) can do. Compare it with the
  snap tolerance: if the floor is close to the tolerance, keep a small correction term in the other
  direction.
- **Rank units by output-weight times activation range.** Units contributing under about 0.3 of a
  range of 5 are candidates to drop or merge, but drop them together and re-measure, since together
  they can matter.
- **Sigmoid nets that imitate |d| use a pattern:** one wide sigmoid on one side and several
  staggered steps on the other. The steps' centres (-b/k) are spaced about one unit apart, and
  their output weights add up to the saturation height. A single `abs()` will not pass when the
  ends saturate. Keep the sigmoid sum and snap k and b to one decimal place.
- **Snap the hidden layer, then refit the output layer.** Snap the hidden slopes and biases, then
  solve for the output weights by minimax (LP), not least squares. The checker uses the max error,
  and least squares left 0.12 where minimax gave 0.097 for the same features.
- **Get a lower bound to know when to stop.** A quick joint nonlinear minimax optimisation shows
  how much any further snapping or tuning could still gain.

## tax/seed2 — agent_1 (me0f03a, 2026-10-28)

- For 1-input sigmoid nets, compute each unit's centre `-b/w` and width `1/|w|`; flip falling
  units (`c*sigmoid(-z) = c - c*sigmoid(z)`) so all are rising steps and fold the constants.
- Many sigmoid steps of similar height spread over the domain mean "ramp", not "staircase":
  look at the numerical derivative of the net rather than at individual units.
- Plateaus in the derivative give the snapped slopes; the places where the derivative moves
  between plateaus give the kinks -> write as `a*x + c + d*relu(x - k)` (or max/min of lines).
- Fit with slopes fixed at snapped values, then choose the offset as the midrange of the
  residual (minimax), not the least-squares mean: the checker bounds the max error.
- Sigmoid nets roll off at domain edges (saturating units); expect the worst errors at the
  ends and at kinks, and do not add hinges just to model that rounding unless needed to pass.
- Check margin vs snap tolerance (2% of output range) before snapping the offset; scan a few
  snapped offsets/kinks (quarter/half steps) and keep the one with the lowest max error.

## iris_petal/seed5 — agent_1 (m62d6e6, 2026-10-28)

- For argmax nets with 2 inputs, first print the argmax on a coarse grid: it shows the region
  topology (which classes border each other, ordered bands vs. wedges) right away.
- Bisect each pairwise boundary (only where the pair is the top two) along one input for many
  values of the other, then fit a line by least squares. The residual tells you if one linear test
  per boundary is enough.
- When the classes are ordered bands, write the program as a decision chain (if A: 0, elif B: 2,
  else 1); the middle class needs no test of its own.
- Write each boundary as `x0 + k*x1 < t` (normalise one coefficient to 1), then snap k and t
  separately: scan a few nice k values and for each the set of passing t, then pick the centre of
  the widest passing range. Check the snap with the checker, not by eye: the allowed slope range
  can be narrow (here +-0.1).
- Sigmoid nets give smooth, slightly curved boundaries. A straight-line rule usually passes snap but
  not exact; the remaining disagreements cluster at the boundary end with the most curvature.
- The many hidden units (8 here) are no hint of complexity: several sigmoids are near-duplicates
  (similar weight directions) that together give one sharp boundary.

## window/seed2 — agent_1 (ma9f2b1, 2026-10-28)

- For a 1-input binary net, do not decode the units first: scan the score densely over the input
  range, list the sign changes, and bisect each one to full precision. The decision is then an
  interval rule whose number of intervals equals the number of crossings + 1.
- Read each sigmoid unit as a step at x = -b/w with sharpness |w|. Steep units with large output
  weights carry the decision; a shallow unit with a small output weight is a bias correction and
  can be dropped.
- Two steps of opposite direction with opposite-sign output weights form a bump (window). Rising
  step with negative weight and falling step with negative weight around a positive bias = "inside
  the band" detector.
- When the two edges of a band are near +-c, snap them to a symmetric pair and write `abs(x) < c`.
  Prefer simple fractions (1/2) for edges on a [-2, 2] float range.
- If a snapped threshold fails, keep the bisected value with ~8 significant digits: that usually
  gives PASS (exact) for binary outputs, since disagreements then only sit within float noise.
- Return +-1 instead of the raw score for a binary rule; only the sign matters.

## xor/seed2 — agent_1 (m13abfd, 2026-10-28)

- With all-boolean inputs and a small number of them, enumerate the full truth table first. The
  table is the specification; then name the boolean function (XOR, AND, majority, ...) and write it
  directly.
- For binary outputs, look at the margin |score| at each point. If every point is far from the
  threshold, the exact weights don't matter, and a clean boolean expression will match exactly.
- Return a symmetric score (+1/-1) rather than 0/1 when the threshold is 0 and the decision is
  `score > 0`. It is clearer and avoids a tie at the threshold.
- Hidden units with near-identical incoming weights (within about 10%) are duplicates. Merge them
  by summing their outgoing weights. Here two "NOR" units merged into one.
- Sigmoid units with large weights (|w| > 6) on 0/1 inputs are effectively step functions. Read
  each one as a threshold on a linear combination of the inputs: AND, OR, NOR, or "x0 and not x1".

## iris/seed5 — agent_1 (m33747e, 2026-10-28)

- For argmax nets, first find which class pairs are actually adjacent. Look at the top-2 pair
  where the gap is small. Non-adjacent pairs need no boundary in the program. If the adjacent pairs
  form a chain (0|1, 1|2), the program is an ordered if-chain with one plane per link.
- Fit boundaries directly instead of the outputs. Bisect random segments whose ends have different
  classes onto the boundary, then SVD the points. A tiny last singular value means the boundary is a
  plane, whatever the activation function, and the singular vector gives the coefficients.
  Fitting the output logits linearly gave poor R^2 (0.6) even though the decision boundaries
  were near-perfect planes.
- Strongly correlated hidden pre-activations (|corr| > 0.9) suggest the net is mostly one-dimensional,
  and its smooth sigmoid boundaries are close to planes.
- Normalise each plane so the dominant input has coefficient 1. Then snap the other coefficients
  and the bias by a grid search that minimises "decimal digits used" subject to zero confident
  disagreements. The unsure-zone width decides how far you can snap. Check the result on a few
  input sets with other seeds before trusting it.
- Integer-coefficient planes only appear at large scales (15-20) when the margin is tight. Prefer
  2-decimal coefficients in that case.
- Separate class-ordering checks (`if low < t0: return 0`, `if high > t1: return 2`) are safe
  when the two boundaries don't cross inside the box. Check that no confident class-0 point lies
  above the second plane.

## iris/seed3 — agent_1 (mb320b7, 2026-10-28)

- For argmax/binary nets, first label the checker's input set with the net, keep only the
  confident points, and test whether each class pair is linearly separable in the raw inputs.
  Sigmoid/tanh nets with few hidden units often are, even when the hidden layer looks
  complicated. In that case the whole net becomes 1-2 linear comparisons in a decision list
  (one class vs rest first, then the remaining pair).
- Correlated hidden units (|corr| > 0.9 over the input box) are copies of one feature. Summing
  their output weights per group shows the decision-list order (which class each group turns on).
- To snap a separating plane, keep the direction separate from the threshold. For each direction,
  the allowed thresholds form an interval [max over the negatives, min over the positives].
  Its width is concave in the direction, so local search finds the max-margin direction.
  Then round the direction as far as the interval stays non-empty, and pick a round threshold
  inside it.
- Logistic regression is a good starting point, but it is not max-margin. A plane with a tiny
  margin passes one input set and fails on another seed. Re-check with a few other
  `check_program` seeds before trusting a snapped plane. Failures show up in box corners, far
  from the training data, where the net's boundary curves.
- When the margin is thin (gap < ~0.05 in input units), prefer 3-decimal coefficients over a
  "nicer" but fragile plane.

## median3/seed2 — agent_1 (mdb2d22, 2026-10-28)

- **Probe first:** check the symmetry (odd? which input swaps leave it unchanged?), then a few integer
  points, then compare with candidate formulas (median, max, sum...). Measure the max error against
  the tolerance before trusting a "clean" formula. A trained net can be a rough approximation of
  the clean function, so the program must follow the net, not the idea.
- **Sigmoid sign-flip:** `w*sigmoid(z) = w - w*sigmoid(-z)`. Flip units with negative outgoing weight.
  If the rows then line up, the layer is one unit copied many times. Collapse it to one unit and
  refit its row by least squares; plain averaging is not enough.
- **Near-linear sigmoid units:** units with small weights and small biases (|arg| < ~1 over the
  domain) act linearly. Replace them as a group with one linear term in the inputs, then refit.
- **Centered sigmoids:** use `sigmoid(.) - 0.5` (= tanh(./2)/2). Biases then often vanish and the
  function's oddness shows.
- **Pairs of near-identical units with opposite weights** compute a derivative-like bump
  (sig(a+d) - sig(a-d) ~ 2d*sig'(a)). They are not redundant. Keep them as a difference and test
  before deleting.
- **Snap first-layer directions** in a basis of pairwise differences (2xi - xj - xk, xj - xk, plus the
  sum) for order-statistic tasks (min/max/median), and look for mirror pairs under input swaps to tie
  their coefficients.
- **Order of work:** fix the directions (snapped), refit only the few scalar coefficients by least squares, then
  snap those to 2 significant digits, checking the max error at every step.

## iris/seed4 — agent_1 (mbd0d9e, 2026-10-28)

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

