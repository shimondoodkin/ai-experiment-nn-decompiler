# Heuristics learned by Claude Code

Appended by `nndc agent collect`, one section per refactoring attempt.

## xor/seed0 — agent_1 (mc4d1f6, 2026-10-29)

- For all-boolean inputs with few inputs (2^n small), skip the weights: evaluate the net on the
  whole truth table and read the boolean function off it. Then write the simplest boolean
  expression (here `x0 != x1`).
- Check the margin at each truth-table point; if every |score| is far above the threshold, the
  boolean rewrite is safe and no "unsure zone" handling is needed.
- For a binary output, returning a 0/1 comparison works as the raw score when the threshold is 0.
- Saturated tanh units with weights ~±4 and biases ~±2 on 0/1 inputs act as logic gates: the sign
  of the pre-activation at each corner gives the gate (bias ~ +2 with one +4 and one -4 is an
  "implies"-type gate; bias ~ +1.6 with two -4 is NOR; bias ~ -5 with two +3 is AND).
- Mirror-image units (same magnitudes, swapped input signs) and AND/NOR pairs usually signal a
  symmetric function such as XOR/XNOR; expect redundancy beyond the minimal 2 hidden units.

## xor/seed1 — agent_1 (m6b5d0f, 2026-10-29)

- For all-boolean (or small finite) input domains, enumerate the truth table from the dump first.
  The program is then a boolean formula of the inputs, and the hidden units do not need decoding.
- For binary output, check the raw score's magnitude on each row, not only its sign. Rows with |score|
  far below the confident rows are "fence" rows: the net is undecided there. Pick the formula that
  matches every sign, the exact one, and note the other fence-row readings as alternatives.
- With a 0/1 input and threshold 0, `return x0` is a valid binary program: 0 is not > 0. No offset
  like `x0 - 0.5` is needed.
- An input whose flip changes only the score's magnitude, never its sign, can be dropped from the
  decision program, even when its weights are large (here |w| ~ 6-7 on x1 in every unit).

## window/seed0 — agent_1 (m3f8005, 2026-10-29)

- tanh/sigmoid units whose input weight is tiny relative to the input range (|w| * range << 1)
  are constants: replace with tanh(bias) and fold into the output bias.
- A tanh unit `tanh(w*x + b)` with large |w| is a step at x = -b/w; read the threshold off as
  -b/w before snapping.
- Two steps at a and b combined with same-sign output weights (one rising, one falling) form a
  window/bump; for a binary output this is just `a < x < b`, and a symmetric pair becomes `abs(x) < c`.
- For 1-input binary nets, scan the score on a dense grid to find the exact zero crossings; then
  snap them to nearby simple values (here +-0.5) and check that disagreements stay in the unsure zone.
- If snapped thresholds only pass "snap", keep them anyway when they are clearly the intended
  values; the raw crossings give the "exact" variant if needed.

## window/seed1 — agent_1 (m37516e, 2026-10-29)

- Before simplifying units one by one, evaluate the output over the input range (endpoints and a
  coarse grid). If its sign never changes, the binary net is a constant classifier.
- Bound each unit's pre-activation over the input box: |w| * (hi - lo) / 2 tells how much a unit can
  move. When all weights are tiny relative to the domain, each unit is ~a constant
  (tanh(b) or sigmoid(b)), and the output reduces to a constant plus a tiny linear term.
- For a constant binary output, return a constant clearly on one side of the threshold (e.g. -1),
  not the threshold itself, so the decision is robust.
- A net that learned a constant usually means the training signal was dominated by one class in this
  domain (or the true boundary lies outside it); don't invent a boundary the net doesn't have.

## neareq/seed0 — agent_1 (m014168, 2026-10-29)

- Find units whose weights on two inputs are nearly equal and opposite (w, -w). They depend on the
  difference of those inputs, so rewrite them in terms of d = xi - xj.
- Units with small weights relative to their bias (|w|*range << |b|) are constants on the input box.
  Fold them, with the output bias, into one offset.
- A pair of steep sigmoid/tanh units on the same direction, with output weights of opposite sign and
  thresholds at -t and +t, is a window/bump: a test of the form |d| < t. The offset decides whether
  the inside or the outside of the window is class 1.
- To find thresholds, scan the real net along the found direction at several offsets and snap the
  average crossing to a simple value (0.2 here). Residual drift of a few thousandths along the band
  usually comes from the near-constant units and is safe to drop for a snap pass.
- For binary outputs, returning a signed margin (t - |d|) keeps the same decision and is easy to read.

## neareq/seed1 — agent_1 (mefb6f7, 2026-10-29)

- **Split units by steepness.** In a tanh/sigmoid layer, units with |w| * input range much less
  than 1 and |bias| > ~1.5 are saturated or near-linear over the domain. Fold them (with the
  output bias) into a constant, or a linear term if needed. The few steep units carry the logic.
- **Equal-and-opposite input weights mean a difference variable.** A unit with weights (-k, +k)
  depends only on `x1 - x0`; its step sits at `d = -bias / k`.
- **Two steps of opposite output sign on the same variable form a window.** A positive weight on
  an increasing step at d = a and a negative weight on an increasing step at d = b > a give
  "high when a < d < b". When a = -b, write it as `abs(d) < b`.
- **Evaluate the sign table with the folded constant** to decide which regions are class 1,
  rather than reading weight signs alone.
- **Measure the real boundary numerically** (root-find the net along the relevant direction at
  several positions) before snapping. The step midpoint `-bias/k` is only approximate (here
  -0.192/0.204) because the other terms shift the zero crossing; the actual roots were much
  closer to +-0.2.
- **Grid ties.** Checkers that sample on grids hit points exactly on a snapped boundary
  (here d = 0.2). Disagreements there are expected and fall in the unsure zone.

## absdiff/seed0 — agent_1 (mdc31cc, 2026-10-29)

- Look at the first-layer weight vectors. Units whose weights point the same way, here (-1, +1),
  depend only on that projection. Rotate the inputs into `diff = x1 - x0` and `total = x0 + x1`
  before you analyse anything.
- Two tanh units with the same slope, opposite biases (+b, -b) and opposite output weights of about
  the same size form an even bump or dip: `A*(tanh(k*u + b) - tanh(k*u - b))`. Merge them with a shared
  k, b and A.
- Units whose input weights are tiny compared with the input range (|w|*range < ~0.5) work as a
  constant plus a linear term. Fold them into one bias and a small linear term. Check whether the
  linear term is needed against the tolerance.
- Print 1-D profiles along the main projections (here diff at total = 0, and total at diff = 0). They
  show the function's shape and how big the leftover variation is, before any fitting.
- Fit a few small parametric forms with least squares, then grid-search rounded values near the fit
  and keep the simplest one that stays well inside the snap tolerance.

## absdiff/seed1 — agent_1 (mf34259, 2026-10-29)

- Check whether the hidden units' input weights come in (+w, -w) patterns or other shared
  directions. If every unit that matters has the same direction, the net is a 1-D function of one
  projection (here x0 - x1). Rewrite the program in that variable and probe the net along it.
- Units whose input weights are tiny compared with their bias (|w| * input range << |bias|) are
  constants over the domain. Fold `w_out * act(bias)` into the output bias. Their combined
  variation can still reach a few hundredths, so check the total against the tolerance.
- Two tanh (or sigmoid) units with the same slope, opposite biases (±b) and opposite output weights
  form an even bump or dip: `A * (tanh(s*d - b) - tanh(s*d + b))`. Merge them, averaging s, b and |A|.
- After choosing structure and snapping the nonlinear parameters (slopes, offsets), refit the linear
  parameters (output weights, constant) with least squares on random inputs. Then round them.
  This recovers most of the error the snapping introduced.
- The largest errors tend to be at the domain corners, where the net extrapolates. Compare
  candidates by max error over a sample that includes the corners, not by RMS.

## median3/seed0 — agent_1 (md993b3, 2026-10-29)

- **Check symmetries first.** Compare f with f under each input permutation and each sign flip.
  Then correlate the output with order statistics (min, median, max), the sum, |x_i| and products.
  A correlation of 0.9997 with the median was decisive at once.
- **Order statistics as a basis.** When a function is symmetric under permutation, rewrite it in
  sorted coordinates (lo, med, hi) and the gaps (med − lo, hi − med). Residuals then become 1-D
  curves.
- **Tanh/sigmoid nets that approximate min/max/median/abs have rounded kinks.** Near a tie the
  output is offset by a fixed amount (here 0.075, about 1.3% of the range), and the offset dies out
  within a small gap (here 0.2). It does not depend on location. Bin the residual by the gap to see
  it, and model it with `k*relu(w - gap)`, which is linear in the language.
- **Wide-spread corners are poorly trained.** Points where the inputs sit at opposite extremes of
  the cube are rare under uniform training data, so the biggest errors cluster there. Gate any
  correction with `relu(spread - w)` so that it cannot disturb the rest of the domain.
- **Fit corrections by minimax, then snap.** The snap criterion is a maximum error, not RMS. A
  cutting-plane LP (`scipy.optimize.linprog` on a working set that grows with the worst violators)
  finds a sparse correction with the fewest terms. Least squares left failures where minimax
  passed.
- **Test on more than the checker's set.** Use an edge-biased sample (most coordinates within 10%
  of a bound) and other checker seeds, so that the margin is real and not luck of one sample.
- **Compute in NumPy.** The checker's language evaluator is slow on millions of points. Cache the
  checker's input set (`build_input_set`) and evaluate candidate formulas in NumPy, and use
  `check.py` only to confirm.

## median3/seed1 — agent_1 (m108c46, 2026-10-29)

- Before reading weights, probe a small grid of slices (fix all inputs but one at a few values,
  sweep the last). Order-statistic functions (min, max, median, clamp) show up as piecewise-linear
  slices with slope 0 or 1 whose breakpoints equal the other inputs' values.
- Fit `y` against the candidate formula (slope/intercept/RMS) before refining; slope ~1 and
  intercept ~0 confirm the hypothesis.
- Median of three in a min/max language: `a + b + c - min(a,b,c) - max(a,b,c)`.
- Smooth-activation (tanh/sigmoid) nets that approximate a piecewise-linear function deviate most at
  its kinks. Bin the residual by the distance to each kink; a soft-min/soft-max correction
  `c * relu(w - gap)^2` (signed by which side the kink bends) usually captures it.
- Second, check residuals against "how close to the input box corners" features (spread, |x| near
  the bound): tanh nets often saturate and shrink the output at the extremes of the training domain.
  A hinge on spread (`relu(spread - s)`) localises the fix to the corners.
- Use least squares to pick candidate features, then tune the few snapped constants against the
  checker's max error, since acceptance is a max-error criterion, not RMS.
- Add the mean residual as a constant offset when the tolerance margin is tight.

## tax/seed0 — agent_1 (m45eb4d, 2026-10-29)

- For 1-input tanh/sigmoid nets, start with a unit census: centre = -b/a, width = 1/|a|, and the
  output weight. Units whose centre lies more than about 3 widths outside the domain are
  constants (or linear terms, if the width is large). Large, equal-magnitude output weights (all
  about ±7 here) mean the units are tiling a ramp, not encoding separate features.
- Look at the numerical derivative of the whole net on a fine grid before looking at units. A
  plateau-valued derivative profile (0.15, then 0.4) points to a piecewise-linear target. Wobble
  around a plateau is fitting noise and should not become extra pieces.
- Ignore derivative changes within about one unit width of the domain edge. They usually come
  from a tanh centred near the edge (here a drop near x=200 from the unit centred at 181).
- Convex piecewise-linear functions read best as `max(line1, line2, line3)` (and concave ones
  as `min`), with slopes snapped first and intercepts chosen so the pieces stay continuous at
  round knots.
- Do a free minimax fit first to see the error floor (0.67 here). Then snap slopes, knots and
  intercept to round values and accept the snap if the error stays well under the tolerance
  (here 0.997 against 1.228).
- A tanh net is almost never `PASS (exact)` after removing units. Saturated units still differ
  from ±1 by about 1e-4 near the domain edge, which exceeds the exact tolerance at small |y|.
  Go straight for a meaningful snapped formula.

## tax/seed1 — agent_1 (mc80d51, 2026-10-29)

- Normalise every tanh/sigmoid unit to `w * act(k*(x - c))` with `k > 0`, using
  `tanh(-z) = -tanh(z)`. Then the centre `c = -b/a` and the sign of `w` show which way each
  step goes. Read it as a table of steps.
- A unit whose centre is outside the input box by more than about 3/k is saturated. Fold it into
  the output constant (`w * tanh(+-inf) = +-w`). Check how far from +-1 it is at the nearest edge.
- When every output weight has about the same size (here +-7), the net is building a function
  from equal-height steps. If the step widths (about 2/k) are larger than the gaps between
  centres, the sum is a smooth ramp, not a staircase. Look at the numerical derivative and fit
  piecewise-linear pieces, not thresholds.
- For 1-input regression: tabulate the net, take the gradient, find the slope plateaus, fit
  hinges by least squares (`[1, x, relu(x-k)...]`), then grid-search rounded coefficients and
  keep the simplest that stays well under the tolerance (aim for about half of it, because the
  checker samples densely, including boundary points).
- Edge effects: a step centred near the domain edge bends the curve there (here the slope drops
  at x0 = 200). A hinge fit should tolerate this rather than add a piece for it.
- Keep a faithful intermediate (saturated units removed, exact rewrites) and check it first, to
  separate "my algebra is right" from "my approximation is good enough".

## iris/seed0 — agent_1 (m4e1234, 2026-10-29)

- For argmax nets, first find which class pairs actually share a boundary: bisect random segments
  onto class changes and tabulate the pairs. If some pair never touches (here 0|2), the classes
  are ordered, and the program becomes a decision list: test the extreme classes, and the middle
  class is the default.
- Fit a plane to each pair's boundary points (smallest singular vector) and write it solved for
  the input with the largest weight. Residuals binned by the other inputs show where the boundary
  bends. For tanh/sigmoid nets, the bends are at the box edges, where units saturate.
- If one plane is not enough, check which side is convex. If the boundary bends away from a
  class, that class's region is an intersection of half-spaces: `a < t1 and b < t2`. Two planes,
  fitted by alternating LPs, are usually enough for a small smooth net.
- Use the checker's unsure-zone threshold to define "confident" points, and require 0 errors on
  those only. That leaves a real gap, so an LP with a positive margin is feasible.
- Snap all the planes of one boundary together, and score each candidate by its joint slack (how
  far all its thresholds can move while it stays correct), not by its error count alone. Then
  pick round thresholds inside the window.
- Validate on a fresh, larger random sample (a different seed) as well as the checker's input
  set. Thresholds at the edge of the window fitted on one sample can fail on corner points of
  another sample. Move them toward the middle of the window.
- The tanh units themselves did not need to be understood one by one. Working on the decision
  geometry was far shorter than pruning or merging units.

## iris/seed1 — agent_1 (m9d250e, 2026-10-29)

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

## iris/seed2 — agent_1 (m147f70, 2026-10-29)

- Check each unit's pre-activation range over the input box first. A unit saturated over the whole box
  (|z| > ~4 for tanh/sigmoid) is a constant; fold it into the next layer's bias.
- Ablation test: set a unit to 0, or to its mean, and count confident disagreements on a large pool. Units
  with small outgoing weights often drop out with no change.
- For odd activations (tanh), compare weight rows up to sign: w and -w are duplicates. When normalized rows
  and outgoing-weight ratios agree, merge them into one unit (mean row, summed outgoing weights). Try it
  even when rows differ by ~20%; argmax decisions tolerate it.
- For argmax outputs, subtract one class's score from all of them: `argmax(s0, 0, s2)`. One fewer row, and
  units whose weight differences are ~0 disappear from a score.
- Before trying to fit linear rules, check how much slack the unsure zone allows: run an LP max-margin
  separator on confident points. If the margin is tiny, piecewise-linear rules will not snap. Keep the
  activation form and snap its weights instead.
- Build the test pool with box-face/corner-biased samples and points bisected onto the net's class
  boundaries, not just uniform samples. Simple rules failed first at corners.
- Snap greedily, coarse step first, accepting only zero confident errors. Snap output weights first
  (they become integers easily), then unit rows. Validate on a fresh independent pool afterwards.

## iris_petal/seed0 — agent_1 (m8889c8, 2026-10-29)

- For a 2-input argmax net, first print the class map on a coarse grid. It shows at once how many
  boundaries there are and which classes touch.
- Bisect the boundary along one input for a sweep of values of the other input. A table of crossing
  points shows whether the boundary is straight; check the first differences.
- Compute the checker's unsure-zone width in input units (score margin / gradient, or measure it
  directly along a line). It tells you how much you can round coefficients. Pick the simplest
  1-decimal line that stays inside the zone at both ends and in the middle of the boundary.
- When two classes never touch, the argmax is a chain of one-sided tests, e.g. `if A: 0 elif B: 2
  else 1`. The order of the tests does not matter where their regions do not overlap.
- tanh nets with smooth boundaries usually reduce to linear rules for a snap pass. Mild curvature
  (well under the zone half-width) can be ignored.

## iris_petal/seed1 — agent_1 (m09849e, 2026-10-29)

- For low-dimensional argmax nets (2 inputs), map the class regions on a grid before touching the
  weights. Eight tanh units that combine into two straight boundaries are far easier to see from the
  output than from the units.
- Find boundary points by scanning one input at fixed values of the other for label changes, then
  fit x_a = slope*x_b + c. If the residual is tiny, a linear threshold is enough. Check a quadratic
  term to confirm there is no curvature.
- Ordered bands (0 | 1 | 2 along one direction, with 0 and 2 never touching) become a decision list:
  `if s1 < t1: 0; elif s2 < t2: 1; else 2`.
- Choose snaps by the largest net margin among disagreements, not just by pass/fail. A snap that
  passes close to the unsure-zone limit is fragile: new counterexamples or another seed would break it.
- A slope error matters more than an intercept error when the other input has a wide range. Snap the
  slope only if slope error × input range is small compared with the boundary's tolerance band.
- Express each boundary as x0 + k*x1 < c (coefficient of one input normalised to 1), so the snapped
  values stay readable.

## iris_petal/seed2 — agent_1 (md95cab, 2026-10-29)

- For low-dimensional argmax nets, start from the decision map (class on a grid) rather than the
  hidden units. Many units are only there to bend one boundary slightly, and the readable program
  is a short if-chain over the boundaries.
- Find the order in which the regions are arranged (bands, corners) so that the if-chain can test
  one boundary at a time. Check that the non-adjacent classes really never meet inside the box.
- Measure each boundary by root-solving pairwise score differences (score_a - score_b = 0, with the
  third score below both) along one input. Also measure the checker's unsure-zone band around each
  crossing: that band is the snap budget for the boundary's position.
- Try a nice-coefficient line first (slope +/-1, simple fractions). If the boundary's slope changes
  by more than the band allows, express it as x0 = quadratic(x1) with simple coefficients rather
  than stacking line pieces. One product term is shorter than a min/max of 2-3 lines.
- Sweep the snapped constant a little (for example 6.5 / 6.55 / 6.6) and keep the simplest value
  that passes with 0 confident disagreements.
- Nets trained on clustered data extrapolate their boundaries smoothly into empty corners of the
  input box. The program has to follow the net there too, not the data's likely structure.

