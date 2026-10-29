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

## xor/seed3 — agent_1 (mb7fb76, 2026-10-29)

- For all-bool inputs with a small domain (2^n ≤ a few thousand), enumerate the truth table first
  and write the program from it. Don't try to simplify the weights unit by unit.
- Look at score magnitudes as well as signs. Scores near 0 on some rows mean the net is unsure
  there (its training data was likely ambiguous or contradictory). Match the sign to get an exact
  pass, but mention that those rows are fragile.
- Sigmoid nets on bool inputs are piecewise constant on the corners, so the hidden units can be
  dropped once the truth table is known. Pick the shortest boolean formula that fits the table
  (OR / AND / XOR / x0 > x1 ...).
- When a simpler formula would also pass in snap mode by flipping only unsure-zone rows, prefer the
  one that matches every decision, since exact beats snap.

## xor/seed4 — agent_1 (mc7a2ae, 2026-10-29)

- With all-bool inputs and a small input count (2^n up to a few thousand), evaluate the dump on
  every input and read the truth table first. Name the Boolean function (AND, OR, XOR, majority,
  threshold count, ...) and write it directly; this beats simplifying the weights unit by unit.
- For binary outputs, check the margin: if every |score| is well above the threshold, the
  Boolean formula is the whole function and an exact pass is expected.
- Sigmoid units with large weights on 0/1 inputs act as Boolean gates. Read each one as a gate by
  checking the sign of its pre-activation at each corner. Near-duplicate units (similar weights and
  bias) are one gate whose output weights add up.
- XOR of two bools is `x0 != x1`; XNOR is `x0 == x1`. Comparisons return 1/0, which works directly
  as a binary output with threshold 0.

## xor/seed5 — agent_1 (m5f1759, 2026-10-29)

- With all-boolean inputs (or any small discrete domain), enumerate the full truth table first.
  The truth table is the function. Reading the weights is only needed to explain it.
- For binary outputs, look at the score margins. If all of them are far from the threshold, a pure
  boolean expression gives an exact match. Keep only the sign, not the magnitudes.
- Recognise standard boolean functions from the truth table (XOR -> `a != b`, XNOR -> `a == b`,
  AND -> `a and b`, OR -> `a or b`, a single input or its negation). The language's comparisons
  return 1/0, which already sit on the right side of a threshold of 0.
- A sigmoid unit with large weights on boolean inputs acts as a logic gate. Read its gate from the
  sign of its pre-activation at each corner. Two units that are the same gate (here, two NOR units)
  are duplicates; add their output weights together.
- A unit that is active only where the bias alone already decides the class (here, x0 AND NOT x1) is
  redundant; drop it.

## window/seed3 — agent_1 (m0f7a10, 2026-10-29)

- For a 1-input binary net, first scan the score on a dense grid and bisect its sign changes. The
  number and positions of the crossings tell you the whole decision rule before you read the units.
- A sigmoid with |w| times the input range >> 1 is a step at -b/w. Write each unit as (threshold,
  direction, output weight). Then the score is a sum of steps, and the plateau values show which
  units matter.
- Steps in the same direction whose thresholds are within about 1% of the range are duplicates: add
  their output weights and treat them as one step.
- A band of class 1 between two crossings at -t and +t (to within a few thousandths) is |x| < t.
  Write it with abs and one shared threshold. Returning a margin-like score (t - abs(x)) keeps the
  decision `raw > 0` readable.
- Snap thresholds to the nearest half-integer or simple fraction when the offset is well below the
  net's transition width (here about 0.005 against a sigmoid width of about 0.1). The checker's unsure
  zone takes up such offsets.
- An unsnapped program with the bisected crossings gives an exact pass for reference. It shows the
  snap only moves the boundary, not the topology.

## window/seed4 — agent_1 (mb5790b, 2026-10-29)

- For 1-input binary nets, skip reading the units one at a time: scan the score densely, find
  the sign changes, and snap each one. The units only matter for explaining the shape.
- Two zero crossings at roughly ±c usually mean an `abs(x) < c` (or `abs(x) > c`) band. Snap
  to a symmetric threshold and write it as the margin `c - abs(x)`, which is shorter than an
  if/else.
- Sigmoid units with almost the same step location (-b/w) and the same weight sign are one
  step split in two. Add their weights.
- A sigmoid step with a large weight that sits far from every zero crossing (only making
  confident regions more confident) does not affect the decision and can be dropped.
- The step location of sigmoid(w*x + b) is -b/w. Compare it with the score's actual zero
  crossings: the boundary falls where the weighted steps balance, not at any single unit's
  step.
- Before snapping a threshold, check that the net's score at the snapped value is inside the
  unsure zone (here about -0.13 and -0.08 against a 0.44 zone).

## window/seed5 — agent_1 (m11da94, 2026-10-29)

- For a sigmoid unit with a large |weight|, treat it as a step at x = -bias/weight; the direction is
  the sign of the weight.
- Evaluate the output score region by region between the step points: it is piecewise constant, so a
  small table of region values gives the decision directly.
- Steps at almost the same point with the same direction merge (sum their output weights); a pair of
  opposite-direction steps at the same point with opposite-sign output weights is one "switch".
- For binary 1-input nets, find the actual zero crossings of the score with a dense scan, then snap
  them to simple values (here +-1/2). Symmetric boundaries -a and +a become `a - abs(x)`.
- Return a signed margin (like `0.5 - abs(x0)`) rather than a 0/1 comparison, so the score has the
  same sign convention as the net (> threshold means class 1).

## neareq/seed3 — agent_1 (mb22a97, 2026-10-29)

- Look for hidden units whose input weights are large, equal in size and opposite in sign. They are
  steps in a difference (x1 - x0), placed at -bias/slope. Group units with the same direction and close
  thresholds into one step, and add their output weights.
- Units with small input weights and a large |bias| are saturated on the whole input box. Treat them as
  constants (sigmoid(b) ~ 1 or 0) and fold them into the output bias. Bound how much they vary over the
  box, to confirm they cannot move a decision.
- For binary outputs, add up the plateau values between the steps (below, between, above). A
  "+big then -big" pair with a negative base gives a band indicator: `t - abs(d - c)` when the band is
  symmetric about c.
- Measure the actual decision boundary numerically along the main direction, at several points across
  the orthogonal direction, and snap to the shared nice value (here ±0.2). Do not rely only on
  -bias/slope of each unit, because the summed steps and offsets move the crossing point.
- When the only gap to an exact pass is slow drift from near-saturated units, accept PASS (snap). A
  readable formula is worth more than matching the drift of the boundary.

## neareq/seed4 — agent_1 (m260bb7, 2026-10-29)

- Look at the ratio of the input weights in each first-layer unit. When many units share the
  same direction (here w*(x1 - x0)), rewrite the net as a 1-D function of that feature and
  analyse it along that line.
- Units whose input weights and output weight are both small next to the others mostly tilt
  the boundary a little; drop them first and see whether the check still passes.
- Near-duplicate units (same slope and bias) add up to one unit; with large slopes,
  sigmoid(k*(d - t)) acts as a step at d = t = -bias/slope.
- For a binary output, find the sign changes of the score along the main feature at several
  values of the orthogonal coordinate. If they barely move, the boundary is a set of
  axis-parallel cuts in the feature: snap them to round values, and use abs() when the cuts
  are symmetric (`c - abs(d)` for a band, `abs(d) - c` for outside a band).
- A large negative pair of steps on each side plus a positive bias is the motif for
  "class 1 inside a band" (the net's version of `abs(a - b) < c`).

## neareq/seed5 — agent_1 (m4994d5, 2026-10-29)

- Group hidden units by the direction of their weight vector. Units whose weights are (w, -w) or
  (-w, w) depend only on the difference x0 - x1; compute each unit's threshold -b/w on that axis.
- Steep sigmoid units (|w| large relative to the input range) are steps; pairs of rising/falling steps
  with opposite output signs at nearly the same threshold usually form one edge of a band.
- Shallow sigmoid units with large biases are near-constant over the input box; treat them as
  part of the output bias first and only keep them if the check fails.
- For binary nets, directly locate the decision boundary by scanning sign changes along lines
  orthogonal to the suspected direction; snap the found thresholds to a simple value (here 0.2)
  and express the region with abs() for symmetric bands.
- A band |a - b| < t can be written as a single return value `t - abs(a - b)`, which keeps the
  output's sign convention (positive = class 1) and gives a margin-like score.

## absdiff/seed3 — agent_1 (m55dec4, 2026-10-29)

- **Look for shared input directions first.** If a unit's weights on two inputs are nearly equal
  or nearly opposite, rewrite it in terms of `x0+x1` and `x0-x1`. Then probe the net on that
  rotated grid (fixed d, varying s, and the reverse). This usually shows the function is mostly
  1-D, here a soft `|x0-x1|`.
- **Estimate the 1-D floor before snapping.** At each fixed d, take the min and max of the net
  over s. Half the largest spread is the best error any d-only program can reach. If it is close
  to the snap tolerance, keep a correction term for the other direction.
- **Refit the output layer after changing hidden units.** It is linear in the output weights,
  so a minimax LP (`linprog` minimising t subject to |A w - y| <= t) gives the best weights. Do this
  instead of snapping each weight separately; snapping one at a time compounds errors quickly.
- **Separate snapping hidden and output parameters.** Snap hidden slopes and offsets by searching
  a small grid of nice values, solving the output LP for each combination. Then search the output
  weights on a fine grid (0.05) and score each set by how coarse its values are, under an error
  limit of about 2/3 of the tolerance.
- **Greedy "coarsest first" snapping can throw away a useful term.** It snapped a small weight to
  0 and lost the term that made the snapping possible. Score coarseness together with error, or
  search jointly.
- **Watch large output weights on shallow sigmoids.** A unit like `6.5*sigmoid(-0.7d-1.7)` over
  |d| <= 6 makes the error very sensitive to the slope (0.01 in slope moves the error by about
  0.04). Snap such units early and let the other parameters absorb the change.
- **Symmetric bumps** across a direction can be written as
  `k*(sigmoid(a*s+b) + sigmoid(-a*s+b))`, one shared pair instead of several one-sided units.
- **Drop units with tiny output weights** (here |w| < 0.01) at once. They are below any
  tolerance.
- **Check the probe yourself before trusting a "nice formula" guess.** `abs(x0-x1)` looked right
  but was 3x over tolerance at d = 0, where the net smooths out the corner.

## absdiff/seed4 — agent_1 (m7282e4, 2026-10-29)

- **Look for units with weights `(w, -w)` (or a common ratio).** They depend on one combined input such
  as `x0 - x1`. When all the strong units share a direction, reduce the problem to one variable.
- **Weak units are linear plus a constant.** A sigmoid unit with |pre-activation slope| * input range
  well below about 4 stays in its near-linear zone. Fit `a.x + c` to their summed contribution on a grid.
  Check the max residual before you drop them; here linear alone left 0.09, which used most of the
  budget until it was merged with the rest of the fit.
- **Rewrite `sigmoid(w*x + b)` as `sigmoid(s*(x - k))` with `k = -b/w`.** Centres and slopes are easier
  to snap than raw biases. A negative output weight can be turned positive with
  `-a*sigmoid(z) = a*sigmoid(-z) - a`.
- **Fit against the per-coordinate envelope, not the mean.** When the reduced variable leaves some
  spread, bin the outputs by `d` and record min and max per bin. Minimax fitting against `[lo, hi]` gives the true
  max error: for each bin it is max(fit - lo, hi - fit). Get those inequalities the right way round: a sign slip
  made the error look like it was 0.001.
- **Snap in stages.** First snap the nonlinear parameters (centres, slopes) to a grid. Solve the linear
  output weights by LP (minimax). Then round those weights to nice values and re-centre the constant as the
  midpoint of the residual range.
- **Leave margin.** The checker samples edges and corners, where the error was about 0.007 larger than on
  my grid. Aim to use at most about 60% of the snap tolerance.
- **Piecewise-linear is not always simpler.** A smooth V needed 5 or more ReLU knots to reach the
  tolerance, while 3 sigmoid steps in `d` fit it with lots of room.

## absdiff/seed5 — agent_1 (maec2a3, 2026-10-29)

- **Group hidden units by weight direction.** Normalise each unit's input-weight vector. Units with
  (w, -w) only see x0 - x1, and units with (w, w) only see x0 + x1. That turns a 2-D net into a sum of 1-D
  functions of a few projections.
- **Measure a group's total swing before keeping it.** Multiply each output weight by the
  sigmoid's actual variation over the domain. A group whose swing is well under the snap tolerance
  becomes its mean, folded into the bias.
- **Look for mirrored units, then use abs.** Units that face +d and -d with similar centres and
  similar summed amplitudes form a symmetric function of d. Rewriting it as f(|d|) halves the unit count.
  Check the asymmetry first, by comparing net(d) with net(-d), so you know how much of the budget it uses.
- **Several sigmoids in a staircase usually approximate a ramp** (here, |d|). Compare the net with the
  simple formula. If the formula alone fails, because of the soft corner at 0 and saturation at the
  domain edge, keep a small number of steps and refit them instead of forcing the ideal formula.
- **Fit to the envelope, not to samples.** For a model in one projection, compute for each value of
  that projection the min and max of the net over the remaining directions. Then do a minimax fit to
  (lo+hi)/2 with error |f - mid| + half-spread. It is fast (1-D) and matches the checker's max-error criterion.
- **Snap in a grid search.** Fix slopes to nice values (2.5, 1.25), then grid-search centres and
  amplitudes in steps of 0.05–0.1. The constant has a closed form: (max(hi-f) + min(lo-f))/2.
  Sigmoid-sum fits have nearly flat valleys, so sequential snap-and-refit wanders. A small grid over
  rounded values works better.
- **Write sigmoid arguments as `w * (x - centre)`**, so the threshold can be read directly.

## tax/seed3 — agent_1 (m035d38, 2026-10-29)

- For a 1-input regression net, plot/sample the output and its numerical derivative first:
  plateaus in the slope reveal a piecewise-linear target; slope transitions mark knots.
- Sigmoid nets fit kinks by smooth ramps; write the line for each plateau (y = a*x + b),
  intersect consecutive lines to get the knot, then snap knots and slopes to round values.
- A unit whose centre (-b/w) lies far outside the input domain is a constant; fold it into the bias.
- Expect edge effects near domain bounds (saturation makes the slope sag); an extra hinge near
  the edge may be needed to stay within tolerance even if the "true" rule has no such knot.
- Do a small grid search over snapped candidates (knots on a 5/10 grid, slopes in multiples
  of 0.1/0.2) scoring max error against the net, and keep the simplest under the snap tolerance
  with margin.
- The worst error of a hinge approximation to a sigmoid net sits at the knot where the net's
  transition is widest (lowest |w| of the units involved); check those first.

## tax/seed4 — agent_1 (m8c658f, 2026-10-29)

- For 1-input sigmoid regression nets, tabulate the output and its numerical derivative across the
  domain. Plateaus in the derivative are the linear pieces. Their slopes are often simple
  numbers (0.1, 0.2, 0.4).
- A sigmoid's centre is -b/w and its width is about 4/|w|. A decreasing sigmoid with a negative
  output weight is an increasing step plus a constant. Units centred well outside the domain are
  constants, or near-linear pieces at the domain edge. Units with tiny output weights can be dropped.
- When the derivative rises step by step, the curve is convex: write it as `max` of lines and no
  branches are needed. When it falls step by step, use `min`.
- Fit each line on its plateau and snap the intercept. Then check that adjacent lines meet at round
  knots (here 20 and 80). Continuity at round knots is good evidence that the snap is right.
- A slope that fades at the edge of the domain often hides a cap or saturation. Try `min(..., round
  value)` before trying extra segments.
- Measure the error against the snap tolerance (2% of the output range) before and after each
  simplification. The worst error usually sits at a soft corner, which a sigmoid net can only
  approximate.

## tax/seed5 — agent_1 (m42a0ee, 2026-10-29)

- For 1-input sigmoid nets, compute each unit's center (-b/w) and width (1/|w|). A falling sigmoid
  with a negative output weight is a rising step plus a constant: `-a*sigmoid(-z) = a*sigmoid(z) - a`.
  Pair units with nearby centers into one step.
- A unit whose center lies far outside the input domain is nearly constant or nearly linear there;
  fold it into the constant / slope.
- Many wide steps spaced closer than about 3-4 widths add up to a ramp, not a staircase. Look at the
  numerical derivative: plateaus in the derivative are linear pieces, ripple between them is an
  artifact of discrete units.
- Fit `c + a*x + sum b_k relu(x - k)` by minimax (linprog over the max abs error), since the snap
  check is a max-error check. Then snap slopes to simple values and knots to multiples of 5 or 10,
  grid-searching over the snapped values and taking the midrange constant.
- Try dropping the constant: if it is small compared with the slack, 0 is nicer.
- Write convex piecewise-linear parts as `max(line, line, ...)` and concave bends as `min(...)`;
  with snapped slopes the intercepts usually become integers and the knots can be read off.
- Sigmoid nets soften slopes at the domain edges; expect extra short pieces at the ends that the
  "true" rule probably does not have, but keep them if the check needs them.

