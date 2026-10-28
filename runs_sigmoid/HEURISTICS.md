# Heuristics learned by Claude Code

Appended by `nndc agent collect`, one section per refactoring attempt.

## neareq/seed1 — agent_1 (m9d87f5, 2026-10-28)

- Look for hidden units whose input weights are equal-and-opposite (or proportional across units).
  This means the net depends on one linear combination, such as `x1 - x0`. Rewrite in that variable first.
- For a steep sigmoid `sigmoid(k*(z) + b)`, the switch point is `z = -b/k`. List these and group the units
  that share a direction and threshold into one step. Mirror pairs (`+z` vs `-z`) with similar
  thresholds and same-sign output weights mean an `abs(z)` band.
- Shallow, small-weight units (|k| <~ 2, |output weight| < 1) usually only tilt the boundary a little.
  Try dropping them first.
- For binary outputs, find the boundary directly: scan the sign of the net along lines and read off the
  crossings. Then snap them to a round number (0.2 here) and return a signed margin like
  `0.2 - abs(diff)`. That stays inside the language and is easy to read.

## neareq/seed0 — agent_1 (m4a51e3, 2026-10-28)

- **Look for steep units first.** In a sigmoid (or ReLU) net with a binary output, the units with
  large input weights (about 10x the others) carry the decision boundary. Small-weight units with
  large biases are usually a near-constant offset. To confirm, evaluate their sum over the domain
  and check that its range is narrow compared with the steep units' output weights.
- **Pairs of steep units with the same direction and opposite output signs form a band (bump).**
  A unit whose weights are (-a, +a) depends only on x1 - x0. Its boundary sits at -bias / a. Two
  such units, one with a positive and one with a negative output weight, give
  lo < x1 - x0 < hi. A symmetric band becomes `abs(x1 - x0) < c`.
- **Snap shared thresholds.** If the two edges of a band are about ±c (here -0.196 / +0.214 from the
  unit biases, and ±0.20 measured), use one symmetric value c, rounded to a simple decimal.
- **Measure the real boundary, not just the unit's switch point.** The constant offset moves where
  the output crosses 0. Root-find the net output along a few lines across the boundary to get the
  actual edges before choosing the snapped constant.
- For a binary output, returning a comparison (1/0) is enough. The checker uses `raw > threshold`,
  so a boolean output of 1 or 0 against threshold 0 works.

## iris_petal/seed0 — agent_1 (m094342, 2026-10-28)

- For low-dimensional argmax nets, first print the argmax on a coarse grid. That shows how many
  regions there are, which classes are adjacent, and whether the boundaries look straight.
- Find boundary points by bisection along one input for many values of the other, then
  least-squares fit `x0 = a*x1 + c` per adjacent class pair. If the max residual is small compared
  with the domain, the boundary is a line, whatever the hidden activation (sigmoid units included).
- When classes are ordered along one direction (0 | 1 | 2 bands), write the result as a decision
  list: test the lowest boundary first, then the next. Non-adjacent class pairs then need no test.
- Snap slopes to simple fractions (7/3, 9/8) rather than forcing integer slopes; argmax checks
  tolerate errors only inside the unsure zone, and slope errors grow across the whole domain, so
  check a coarse snap and step back to a finer fraction if it fails.
- Snap the intercept after the slope: refit or reread the intercept with the snapped slope before
  rounding it.
- Do not chase PASS (exact) on sigmoid nets: even the original dump differs from the reference in
  the unsure zone, so the snap pass is the realistic target.

## iris_petal/seed2 — agent_1 (m202bfd, 2026-10-28)

- For low-dimensional argmax nets (2 inputs), first print the decision map on a coarse grid
  instead of reading weights. Smooth sigmoid nets with many units often just draw a few
  straight cuts.
- To find a boundary, bisect along one input at many values of the other, then fit a line.
  The maximum residual tells you whether one linear threshold is enough.
- If the class regions are ordered bands (class k sits between k-1 and k+1) and the boundaries
  do not cross inside the input box, write the argmax as an if-chain of threshold tests.
- Snap the slope first (to 1, 1.1, 2.5, ...), then refit the intercept for that slope. If a
  rounder slope fails, the failures show up at the far end of the boundary, away from where
  the line pivots.
- For sigmoid nets, an exact pass is usually out of reach with piecewise-linear code. Aim for
  zero disagreements where the net is confident, and accept differences in the unsure zone.
- Coefficient sizes in the hidden layer hint at the boundary slopes. Here, units 2 and 3 have
  x1/x0 weight ratios of about 3-4.6, and units 5-7 about 1-1.4. Treat this only as a hint and
  confirm it with bisection.

## iris_petal/seed1 — agent_1 (m965da0, 2026-10-28)

- For small argmax nets with 2 inputs, first print the argmax on a coarse grid; the region shape
  (bands, wedges, islands) tells you the program structure before reading any weights.
- Bisect each class boundary along one input at many values of the other and fit a line; if the
  residual is small, a single linear test per boundary replaces the whole hidden layer.
- Compute the unsure-zone band (where top-two logit margin < threshold) along each boundary in
  input units; it is the real budget for snapping. Choose the simplest coefficient whose line stays
  inside the band at both ends of the input range (ends are where slope errors show up).
- Write the boundary as `x0 + k*x1 < c` (normalising the coefficient of one input to 1) so k and c
  can be snapped independently to short decimals.
- For ordered classes, a cascade `if s1 < a: 0 elif s2 < b: 1 else: 2` replaces argmax; check that
  the first test implies the second so the cascade cannot skip a class inconsistently.
- Sigmoid nets give smooth, nearly linear boundaries when the units are spread out; their curvature
  is usually far below the unsure-zone width, so piecewise rules are not needed.

## absdiff/seed1 — agent_1 (m75928e, 2026-10-28)

- Look for mirrored unit pairs: weight vectors that are near negatives of each other, with similar
  bias and output weight. Their sum `s(w·x - b) + s(-w·x - b)` is an even function of w·x, a
  "bump/ramp" in |w·x|. Several such pairs with increasing centers make a smoothed |w·x|. With
  ReLU the same pattern gives an exact abs.
- Rotate to the natural coordinates (difference x0 - x1 and sum x0 + x1). Then profile the output
  along one coordinate while sweeping the other. If the spread along the sum is small, drop the
  units that point along the sum and fold their mean into the constant.
- For sigmoid units, rewrite `sigmoid(a*u + b)` as `sigmoid(a*(u - c))` with c = -b/a. Snap the
  center c (a boundary in input units) and the slope separately; centers snap more meaningfully
  than raw biases.
- Do not drop the mirrored half of a sigmoid step pair just because it looks "off" in one half of
  the domain. At u = 0 each half contributes w*sigmoid(-b), which can be a large share of the
  constant.
- The snap tolerance is 2% of the output range. Find the ideal formula's residual profile first
  (here |d| had ±0.11 wiggles plus saturation at the corners). If the profile already uses up the
  tolerance, keep the smooth sigmoid form with snapped parameters instead.
- After fixing the structure, a small grid search over snapped candidate values (with the constant
  re-fit for each) is a quick way to pick the nicest numbers that still pass.

## window/seed0 — agent_1 (m838fdd, 2026-10-28)

- For sigmoid units, the centre is -b/w and the steepness is |w|; with |w| >~ 10 over a range of 4,
  treat each unit as a step function at its centre, with direction given by the sign of w.
- Units with nearly equal centres and same direction are duplicates; sum their output weights.
- For 1-input binary nets, the fastest route is to scan the net on a dense grid, list the sign
  changes of the score, and snap them; symmetric boundaries (-a, +a) become `abs(x) < a`.
- A decreasing step "x < -a" is 1 - "x > -a"; recognising this pairs units 2 and 3 into a single
  boundary, which explains why the output weights look like +-9 with a small bias.
- Snap boundaries to half-steps (0.5) when the net's boundaries are within ~0.5% of them; the
  checker's unsure zone absorbs the residual offset.

## absdiff/seed0 — agent_1 (m6a2994, 2026-10-28)

- **Group hidden units by weight direction** (normalise each input-weight vector). Units that share a
  direction collapse to a 1-D function of that projection. Here it was two directions, `x0+x1` and
  `x1-x0`.
- **Check the size of each direction's contribution before modelling it.** A direction whose units
  have small output weights that partly cancel can often become a constant. Measure the output
  range along that direction with the other direction held fixed.
- **Mirrored sigmoid pairs** `w1*sig(a*d - b) + w2*sig(-a'*d - b')` with similar parameters make an
  even function of d, so work with `abs(d)`. Several such pairs at increasing offsets approximate
  `|d|` as a staircase.
- **Profile, then fit minimax** (LP with hinge features at round knots) rather than least squares,
  because acceptance is a max-error test. Snap the fitted coefficients, and then look at what the
  pieces reduce to: here the middle piece became exactly `t`.
- **Sigmoid nets that approximate a kinked target** (like abs) round off the kink. Model the rounded
  part with a parabola that meets the linear part continuously (`0.25 + 0.75*t^2` meets `t` at
  t = 1), not with extra line segments.
- **Sigmoid nets saturate at the edges of the input domain.** Expect the far ends of the range to
  flatten, and add a `min(…, a + b*t)` tail segment.
- **Some error cannot be removed.** The leftover error is the staircase wiggle of the summed
  sigmoids. Its size (about 2*w/8 for steps of weight w) sets a floor for any clean formula, so
  check early that it is below the snap tolerance.

## xor/seed1 — agent_1 (mb6a022, 2026-10-28)

- When all inputs are boolean (or the domain is tiny), evaluate the net exhaustively and read the
  truth table first; recognise standard boolean functions (AND, OR, XOR, majority, ...) and write
  that directly.
- For binary output, a comparison that yields 1/0 (`x0 != x1`) is a valid score with threshold 0;
  no need to keep the real-valued score.
- Sigmoid units with weights (-a, +a) and bias b: the sign of b decides the gate. b < 0 with
  |b| < a means "x1 and not x0"; b > 0 with b < a means "not (x0 and not x1)", i.e. x1 or not x0.
- Several hidden units with proportional weight vectors and similar bias/weight ratios are
  duplicates of one gate; merge them by summing their output weights.
- Check the margin (smallest |score| over the domain) against the unsure zone to know how
  robust the snapped program is.

## window/seed1 — agent_1 (m42c751, 2026-10-28)

- For sigmoid or ReLU units on one input, compute each unit's centre `-b/w` first. Units whose centres
  coincide (within about 0.01) form one step, so add their output weights (sign-adjusted by direction)
  together.
- A steep sigmoid (|w| ≥ ~10 over an input range of a few units) acts as a hard step. For binary
  outputs the whole net then reduces to a set of thresholds on the input.
- Two opposite step groups whose centres are symmetric (±c) with output weights of the same sign
  mean a band or window: `abs(x) < c` or `abs(x) > c`. Check the sign of the score at the domain
  ends and in the middle.
- Find the real decision boundary by bisecting the net's score, not from the unit centres. The
  output bias and the soft overlap of the sigmoids move the crossing away from the centres (here by
  up to 0.01).
- Write the snapped version first (`abs(x0) < 0.5`). If you want PASS (exact), replace the snapped
  thresholds with the bisected crossings, and keep the snapped formula in the notes as the meaning.
- When nudging a threshold by 1e-6 makes the checker's disagreement count worse in both directions,
  you are at the float32 crossing. Stop there.

## tax/seed0 — agent_1 (m47484f, 2026-10-28)

- For 1-input sigmoid nets, compute each unit's centre `-b/w` and width `1/|w|`; if widths are large
  relative to the gaps between centres, the net is a smooth curve and no single unit is meaningful -
  fit the whole curve instead of simplifying unit by unit.
- Plot / tabulate the numerical derivative on a grid. Plateaus of the derivative are linear pieces;
  transitions mark knots. Here: plateaus near 0.2 and 0.4 gave a 2-piece ramp.
- Fit piecewise-linear forms with knots on a nice grid (multiples of 5), solve slopes by least
  squares, then snap slopes and search intercept by minimax (intercept = mid-range of residual).
  Compare the resulting max error with the snap tolerance (2% of the output range).
- Write a convex piecewise-linear function as `max(line1, line2, ...)` (concave as `min`); it is the
  shortest form and makes the knot implicit.
- The error is usually largest at the domain edges, where saturating sigmoids bend the curve; check
  the endpoints first and add an edge piece only if the tolerance requires it.
- Prefer the fewest pieces that pass with a comfortable margin over rounder coefficients that fail
  (0.39 passed, the rounder 0.4 did not because of the edge).

## xor/seed0 — agent_1 (me00780, 2026-10-28)

- For boolean (or small discrete) input domains, enumerate the whole truth table first; the dump's score
  at each point often identifies the logic function directly, and the margins tell whether any point is borderline.
- For binary outputs, the exact score values do not matter, only the sign relative to the threshold;
  replace the whole network with the boolean expression and return a comparison (1/0) or +-1.
- Nearly equal input coefficients in a unit mean it depends on the symmetric sum (count of true inputs);
  then the unit is a threshold on that count: bias/weight between -1 and 0 -> OR, between -2 and -1 -> AND,
  negative weights with positive bias -> NOR/NAND.
- Redundant hidden units (here two NOR detectors with the same role) are common in small sigmoid nets;
  they merge into one term once the function is identified.
- An OR unit minus an AND unit with a strongly negative bias/NOR correction is the usual XOR pattern.

## iris/seed2 — agent_1 (m4ba292, 2026-10-28)

- When all first-layer rows of a small net point in nearly the same direction (up to sign), the net is
  an ordinal 1-D classifier. Look for a chain of classes and write it as an if/elif chain of linear cuts,
  not as an argmax.
- Check which class pairs are actually adjacent. Count points where (top1, top2) is each pair near
  the decision boundary. Pairs that never meet need no rule.
- Get each boundary by least-squares fitting the output difference `o_a - o_b` on points near zero
  (|diff| < a few units), not over the whole domain. Saturated sigmoids far away spoil a global fit.
- Snap per boundary, not jointly. Restrict the evaluation to confident points near the fitted boundary,
  and evaluate all candidates at once with a matrix product. A naive full-set loop over a 5-D grid is
  far too slow.
- Try several normalisations (divide by different coefficients) before snapping. One of them often
  gives much rounder numbers (here x3 = 2.5, x2 = 1.5 gave 0.6 / 1 / 1.5 / 2.5).
- The unsure-zone width divided by the gradient of the score difference gives how precisely each
  coefficient must be snapped. For sigmoid argmax nets with steep outputs, this allows only about
  0.01-0.05 of slack, so "nice" values may need two decimals.
- Re-run the checker logic with other random seeds to make sure the snapped rule did not overfit one
  input set.

## tax/seed1 — agent_1 (ma96720, 2026-10-28)

- For a 1-input regression net, sample the net's numeric derivative on a grid first. Plateaus in the
  slope are linear segments; the jumps between them are ReLU knots. With sigmoid hidden units, each
  unit is a soft step in slope. Center = -b/w, and the slope change is about amp*w/4 per unit.
- Fit free-knot continuous piecewise-linear models (minimise the max error) with 2, 3, 4 knots.
  Then snap the slopes to a clean pattern (0.1/0.2/0.4) and grid-search round knots plus a constant
  offset against the max error.
- Sigmoid nets smooth corners and droop at the domain edges. The worst error is usually at a
  smoothed corner or at the boundary, so centre the program with a small constant offset (the
  midpoint of min and max residual) rather than forcing f(0)=0.
- If several knot pairs give the same max error, the knots are not identified by the net. Pick the
  roundest values.
- A final slope change right at the domain edge (here at 190 of 200) is likely edge saturation, not
  part of the function. Keep it only if the check fails without it.

## iris/seed0 — agent_1 (m8c7b3c, 2026-10-28)

- **Look for a decision list in argmax nets.** Measure each class pair's logit difference on the other
  classes' regions. If one pair never shares a boundary (its difference stays far from 0), the
  argmax is a chain of binary tests. Order the tests by class adjacency.
- **Fit each pairwise boundary separately.** Bisect points onto `o_i - o_j = 0`, regress one input on
  the others and compare the residual with the zone half-width (margin / |gradient| along that
  input). If the residual is well inside the half-width, a plane will do.
- **Snap by intercept interval, not least squares.** For each snapped coefficient vector, compute
  max(score on negative points) and min(score on positive points) over the confident points. A
  positive gap means a valid intercept exists; prefer the widest gap, then pick a round intercept
  inside it. This search is cheap and gives zero confident errors by construction.
- **For a curved boundary, find the curvature first.** Bin the plane residuals by each input. A U
  shape in one input points to that input's square term, here x3². Test single quadratic terms
  before adding several.
- **Do not drop a small coefficient without checking it.** Its effect is coefficient × input range,
  compared with the zone width: 0.025 × 3.6 = 0.09 is about twice the half-width of 0.046 here.
- **Check robustness on other seeds.** Re-run the checker's input-set builder with other seeds.
  Snapped intercepts chosen near the edge of a narrow feasible interval can overfit one sample.

## iris/seed1 — agent_1 (m72433c, 2026-10-28)

- For argmax nets, first find which class pairs actually border each other: sample points and
  count (top class, second class) where the top-two margin is small. If the classes form a chain
  (0|1, 1|2 only), a decision list with one cut per border reproduces the net.
- The SVD of the first-layer weights shows how many directions matter. A near-rank-1 hidden layer
  with sigmoid/tanh units means the boundaries are close to parallel hyperplanes, so try linear
  cuts before anything else.
- Fit each cut only on points outside the checker's unsure zone (margin_frac * p95 of the
  top-two score), and maximise the gap with an LP. Points inside the zone are free, which is what
  makes a linear cut possible for a mildly curved sigmoid boundary.
- Snapping with a bias interval: fix the snapped slope coefficients, then compute the feasible bias
  as `[max(-w.x_pos), min(-w.x_neg)]`. An empty interval rejects the snap at once; otherwise put the
  bias in the middle. Then try coarser snaps one coefficient at a time (exactly 1, then x.x0, then
  x.xx).
- Check robustness on extra seeds or a pooled set of inputs. A cut that passes one seed with a
  bias gap of about 0.005 can fail on another seed. Choose the bias from the pooled interval.

## median3/seed1 — agent_1 (m2e90f2, 2026-10-28)

- For regression nets, first correlate the output with a library of symmetric candidates
  (sum, max, min, median, products, abs differences). A correlation above 0.99 names the intent even
  when the formula itself won't pass the tolerance.
- Probe the residual (net - candidate) along tie slices: two inputs equal, the third offset by b.
  If it peaks at moderate b and decays, the net is a smooth/soft version of a piecewise function.
  Large residuals at box corners are extrapolation artefacts that no clean formula will match.
- Sigmoid nets: units with small weights on all inputs (|w|*range < ~1.5) sit in the linear part of
  the sigmoid. Fold them into a linear skip term.
- Sigmoid nets: layer-2 units with mirrored weights and opposite-sign output weights are the same
  unit (sigmoid(z) = 1 - sigmoid(-z)). PCA of the pre-activations tells how many distinct units the
  layer really has. Collapse to that many and refit.
- Two near-identical steep units with opposite output weights are not redundant. Their difference is
  a derivative-like bump along the shared boundary, so ablate before pruning.
- Snap greedily and refit the free parameters after each snap, with an L^p loss (p 8-12) to control
  the max error. Snapping a whole layer at once to a coarse grid fails for smooth (sigmoid) nets.
- Keep a margin (~15-30% of the snap tolerance), because the checker adds boundary and corner
  points that a uniform validation set under-samples. Validate on a grid that includes the corners.

## median3/seed0 — agent_1 (mf8d15a, 2026-10-28)

(no "## Heuristics" section found)



