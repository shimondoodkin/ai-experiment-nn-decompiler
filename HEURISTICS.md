# Heuristics learned by Claude Code

Appended by `nndc agent collect`, one section per refactoring attempt.

## absdiff/seed0 — agent_1 (m0969d1, 2026-10-27)

- Group hidden neurons by the direction of their input weights. Weights like (a, -a) mean the neuron only sees a difference x0 - x1, and (0, a) means it only sees one input.
- Two neurons on the same direction with opposite signs, relu(k*d) and relu(-k*d), make an abs-like V shape. Add up (output weight * input slope) on each side. If both sides come to about 1, the group is abs(d). If they differ, it is a*relu(d) + b*relu(-d).
- Several neurons can share one kink. Sum their effective slopes instead of snapping each neuron on its own.
- Neurons with small output weights, or with kinks at the edge of the input range, are often noise. Pairs with nearby kinks and opposite slopes largely cancel. Remove them and use the checker to confirm that the leftover error is within tolerance.
- Snap biases that are tiny compared with the weights (relative bias < 0.1) to 0, so the kinks sit exactly at 0.
- Try the simplest candidate formula early. The checker's report of the maximum deviation tells you whether the remaining terms matter.

## xor/seed0 — agent_1 (m212faf, 2026-10-27)

- With a small discrete input domain (here 2 bools = 4 points), enumerate every input and read
  off the truth table first; the decision rule is often a textbook boolean function (XOR, AND, majority).
- For binary/argmax outputs only the sign/ordering matters; the score magnitude can be dropped, so a
  comparison expression (returning 1/0) is a valid replacement for a whole weighted sum.
- Units whose pre-activation is negative on every reachable input are dead; delete them.
- Biases that equal minus one weight (e.g. 2.08*x0 - 2.08) put the ReLU kink exactly on an input
  corner; snap them to integer thresholds like relu(x0 + ... - 1).
- Recognize standard ReLU motifs: relu(1 - a - b) = NOR, relu(a + b - 1) = AND, relu(b - a) = b AND NOT a
  for booleans.
- Verify with the checker after each simplification; prefer the simplest formula that passes exact.

## neareq/seed0 — agent_1 (md5d6e0, 2026-10-27)

- Look for neurons whose input weights are (nearly) equal and opposite: they depend only on a difference
  (or with equal signs, a sum) of inputs. Re-express the whole net in that derived variable.
- Bound each pre-activation over the input box (min/max via weight signs); a ReLU that is never positive
  is dead and can be dropped, one never negative is linear.
- Neurons with proportional weight vectors (same direction, similar bias/scale ratio) are duplicates;
  merge them by summing output weights times scale.
- Large negative output weights on relu(+d - c) and relu(-d - c) pairs, plus positive "bias" units,
  form a band/ridge detector: output > 0 iff |d| < threshold. Find the threshold numerically by scanning
  the decision boundary, then snap to a round number.
- For binary outputs, measure the boundary directly (bisection/scan along the derived variable for
  several values of the orthogonal variable) rather than reasoning about every unit's magnitude;
  if the boundary barely moves, it is a constant threshold.
- Prefer returning a comparison (1/0) for binary outputs; the score's magnitude doesn't matter.

## absdiff/hand — agent_1 (m96ab95, 2026-10-27)

- A pair of ReLU neurons with negated weight vectors and equal output weights c is c*|w.x + b|;
  with opposite-sign output weights (c, -c) it is the linear term c*(w.x + b).
- relu(a) - relu(a - k) is a clamp of a to [0, k]; relu chains of this form often encode min/max/clip.
- Look for mirrored or scaled-copy neurons first; they collapse to abs, linear or clamp expressions.
- Exact-unit weights and zero biases suggest the dump is already the clean formula; verify with the
  checker before adding complexity.

## absdiff/seed1 — agent_1 (md98dda, 2026-10-27)

- Find dead neurons first: bound each pre-activation over the input box (|b| > sum |w_i|*range_i
  with the wrong sign means always 0; the opposite means always linear).
- Group neurons by the direction of their weight vector (normalize; look for w0 ~ -w1 etc.).
  Neurons along the same direction collapse: sum of (output weight * input scale) gives the
  effective gain, which is often exactly 1 or a small integer.
- A pair of groups along +d and -d with equal gain is abs(d); unequal gains give a leaky/asymmetric
  piecewise-linear function (a*relu(d) - b*relu(-d)).
- Small hidden biases combined with an output bias that cancels their weighted sum are smoothing
  artifacts around a kink; snap them to 0 together rather than individually.
- Compare the max error of the snapped version to the tolerance; if the only residual is at a kink,
  the snap is right.

## absdiff/seed2 — agent_1 (m7aeeca, 2026-10-27)

- Group hidden neurons by the direction of their input weights. Neurons with proportional weight vectors and the same bias (about 0 here) combine into one ReLU whose output weight is the sum of w_in*w_out.
- Find neurons whose pre-activation never changes sign over the input bounds. Check the min and max over the box corners. Replace them with a linear term (or 0 if always off), then fold their constants into the output bias.
- relu(d) and relu(-d) with slopes a and b, plus a linear term of slope s, become |d| when a+s = 1 and b-s = 1. Look for abs, or relu, hidden behind an always-on "linear correction" neuron.
- Find each neuron's maximum contribution, |w_out| * max relu, over the input box. Drop it if that is far below the snap tolerance.
- Try the simplest closed form first (abs, max, min of simple linear combinations) and verify it with the checker before tuning residual constants.

## absdiff_perturbed/hand — agent_1 (m464804, 2026-10-27)

- Snap weights within a few percent of +-1 / small integers, and biases that are tiny relative
  to the input range, to exact values first; then look for structure.
- A pair of ReLU units with opposite weight vectors (w, -w) and equal output weights is an
  absolute value: c*relu(z) + c*relu(-z) = c*|z|. With unequal output weights a, b it is
  a*relu(z) - b*relu(-z), i.e. a piecewise-linear "leaky" form; with a = -b it is linear.
- Weight vectors like (1, -1) indicate a difference x0 - x1; (1, 1) a sum.
- Residual error from snapping grows with input magnitude (slope error times range), so expect
  the biggest mismatches at the corners of the input box; judge snaps by relative slope error.
- Drop output biases that are far below the checker tolerance.

## deadredundant/hand — agent_1 (m3e92af, 2026-10-27)

- Use the input bounds to evaluate each pre-activation's range. If it is always <= 0, the neuron is dead, so delete it no matter how large its output weight is. If it is always >= 0, the relu is the identity, so drop the relu.
- Look for duplicate neurons (identical weights and bias). Merge them by adding their output weights. Two halves that sum to 1 are a common sign of this.
- Rewrite `a + relu(b - a)` as `max(a, b)`, and `a - relu(a - b)` as `min(a, b)`. Pairs of relus often reduce to min, max or abs: `relu(t) + relu(-t) = abs(t)`.
- Check the simplified closed form exactly with the checker, not just on a few samples.

## max/hand — agent_1 (m7f35cf, 2026-10-27)

- A pair of neurons `relu(w.x) - relu(-w.x)` with matching output weights is a linear pass-through: replace it by `w.x`.
- `relu(a - b) + b` = `max(a, b)`; `b - relu(b - a)` = `min(a, b)`; `relu(a) + relu(-a)` = `abs(a)`.
- First collapse pass-through pairs into linear terms, then look for max/min/abs patterns in what remains.
- Check that the simplified algebra is valid over the whole input domain (these identities hold everywhere, so no domain-dependent pruning was needed here).

## median3/seed0 — agent_1 (mcc7963, 2026-10-27)

- Find out what the network computes before you try to simplify its weights. For a small number of
  inputs, sample the input box and regress the output on a set of candidate features: linear terms,
  products, abs, pairwise max/min, |xi - xj|, and order statistics (max, min, median). A correlation
  near 1 with one feature tells you what the network computes.
- Symmetric behaviour (equal coefficients on every input, equal pairwise max terms) suggests a
  permutation-invariant function: sum, mean, max, min, median, count of positives, and so on.
- Hidden-layer weights that come in near-opposite pairs (for example +0.73*x1 - 0.73*x2, or
  -0.78*x0 + 0.77*x1) are pairwise comparisons `xi - xj`. Those are the building blocks of max, min and sorting.
- After you name a candidate, fit `y ~ a*f + b`. Then snap a and b to simple values (here 1 and 0) and
  check the largest error against the tolerance.
- Rewrite order statistics with the allowed primitives. For 3 inputs: median = sum - max - min. In general,
  the k-th order statistic can be built from nested min/max.
- In regression tasks the remaining errors gather at the edges of the input domain and on ties
  (xi close to xj), where a small ReLU net can only approximate a kink. If the errors are confined to
  those places, it's approximation noise and not a sign that the formula is wrong.

## median3/seed1 — agent_1 (ma789bc, 2026-10-27)

- Look at layer 1 before anything else. Neurons with weights like (+a, -a) on two inputs compute
  differences or comparisons, and paired neurons with opposite signs (relu(z) and relu(-z)) build
  |z| or a z = relu(z) - relu(-z) identity. These patterns suggest sorting, min/max or abs.
- Don't simplify the network weight by weight. Treat it as a black box: sample it and fit a library
  of simple candidates (linear, min/max/median, abs, products, norms, range) with a free scale and
  offset. A slope near 1 and an offset near 0 then confirm the candidate.
- Compare the residual against the checker's tolerance (here 2% of the output range). A small net
  approximating a piecewise-linear target makes its largest errors at the corners and edges of the
  input box. Expect the worst-case error there, and don't add correction terms only to imitate
  those errors.
- Write order statistics with nested min/max. For example, median3 = max(min(a,b), min(max(a,b),c)).
- Snap any fitted scale or offset to 1 or 0 when the result stays inside tolerance. The clean formula
  is the intended function, and the remaining gap is the network's training error.

## median3/seed2 — agent_1 (m8505ac, 2026-10-27)

- Start with a least-squares fit on raw inputs; equal coefficients on all inputs signal a symmetric
  function (sorting / order statistics, sum, mean, norm).
- Build a feature library of candidate piecewise-linear primitives (abs, pairwise max/min, |xi-xj|,
  global max/min) plus a few nonlinear ones (products, squares). Features that get ~0 weight can be
  eliminated; the surviving set names the function family.
- Order statistics of n inputs: linear coefficient ~ 1/n-ish on each, and the fit concentrates on
  max/min/|xi-xj| features. Median of 3 = sum - max - min = max(min(a,b), min(max(a,b),c)).
- Once a candidate is guessed, regress the net's output on it (slope, intercept): slope ~1 and
  intercept ~0 confirm it and give the snapped coefficients.
- Look at the worst residuals: if they sit at the domain corners / edges (where training data is sparse)
  and are small relative to the output range, they are approximation error of the net, not structure
  worth reproducing. Don't chase exact pass by encoding that error.
- Read the checker's tolerances first (here snap tol = 2% of output range) to know how much
  approximation error is acceptable.

## neareq/hand — agent_1 (m9015f0, 2026-10-27)

- A pair of ReLUs with opposite weight vectors (`relu(w.x + b)` and `relu(-w.x - b)`) and equal
  output weights is `abs(w.x + b)`. Unequal output weights give a piecewise-linear "leaky" form.
- A bias minus an absolute difference, compared to 0, is a band test: |a - b| < c. If the sign
  is reversed, it is an "inputs differ by more than c" test.
- Keep the real-valued score, not just a boolean, so the check can pass exactly rather than
  only by snapping.
- Before snapping anything, look for algebraic identities (abs, min, max) among the hidden units.
  They often remove the whole hidden layer.
- Boundary disagreements of about 1e-16 are float rounding, not logic errors.

## neareq/seed1 — agent_1 (m79551b, 2026-10-27)

- Look for neurons whose weights are opposite in sign and nearly equal in size. They read a difference (x0 - x1), so rewrite them with a new variable `d`.
- Divide each neuron by its weight size to get relu(d - t). Neurons with almost the same t and same direction can be merged into one hinge, with their output weights times their sizes added together.
- Use the input bounds to check each neuron's preactivation range. If it is always positive, the neuron is linear: fold it into the bias and the linear terms. If it is always negative, it is dead: remove it.
- For a binary output, solve for where the collapsed score is zero, then snap those crossing points (here +-0.198 and -0.202 become +-0.2). Then write the simplest score with the same sign, for example `t - abs(d)` for a band.
- Two hinge groups that mirror each other around 0, with a positive bias, usually mean a band `|d| < t` (an "almost equal" test).
- Disagreements exactly on the snapped boundary are expected. Accept them when the checker says they are in the unsure zone.

## neareq/seed2 — agent_1 (md0e4cf, 2026-10-27)

- Normalize each hidden neuron as `relu(scale * (direction . x - threshold))`: neurons with the
  same direction (up to sign/scale) and close thresholds are duplicates; merge them by summing
  `out_weight * scale`.
- Check each neuron's pre-activation range over the input box (interval arithmetic: sum of
  |w|*range); neurons that are never positive are dead, ones never negative are linear.
- Pairs of mirrored hinges `relu(d - t)` and `relu(-d - t)` with large negative weight plus a
  positive bias mean "|d| < t'": a band / closeness test, i.e. `t' - abs(d)`.
- Small-weight, always-positive neurons combined with the bias are often just a "positive
  offset" that shifts the threshold; estimate threshold = hinge_offset + offset / hinge_slope,
  or measure the decision boundary numerically along the relevant direction.
- For binary outputs, find the boundary empirically (sweep along the discovered direction at
  several positions) and snap the crossings to one shared round threshold; small drift across
  the domain is usually training noise.
- Directions with coefficients like (3.50, -3.50) or (-3.97, 4.00) are just +-(x0 - x1);
  look at coefficient ratios, not magnitudes.
- Try the simplest formula first with the checker; only add detail if confident points fail.

## ramp/hand — agent_1 (m51bbc9, 2026-10-27)

- `relu(x - a) - relu(x - b)` with a < b and equal unit slopes is `clamp(x, a, b) - a`,
  i.e. a ramp that saturates; replace with `min(max(x, a), b) - a`.
- A sum of ReLUs of the same 1-D input is a piecewise-linear function; locate the kinks
  (where each pre-activation is 0) and the slope change at each to read off the shape.
- If weights are already clean integers, look for a closed form (clamp, abs, max/min)
  rather than just simplifying arithmetic: `relu(x) + relu(-x) = abs(x)`,
  `relu(x) - relu(-x) = x`, `a + relu(b - a) = max(a, b)`.
- Check the input bounds: kinks outside the input range can be dropped.

## sparse8/hand — agent_1 (m8ae0e3, 2026-10-27)

- Bound each small weight's total contribution by |w| * (input range) summed over inputs; if that sum stays below
  the snap tolerance, drop those terms entirely.
- Weights within a few percent of 0, ±1 or small integers are almost always the intended values; snap them.
- Fold identity output layers (weight 1, bias 0) and single-neuron hidden layers straight into the return expression.
- A single ReLU of a signed difference `relu(a - b)` means "excess of a over b"; recognize common motifs
  (relu(a-b) + b = max(a, b), relu(a-b) - relu(b-a) = a - b, etc.) when there are more neurons.
- Inputs whose weights are all snapped to zero are unused arguments; keep them in the signature only.
- Verify with the checker after each snap; if failures appear, look at the listed inputs to see which snapped term mattered.

## sum8/hand — agent_1 (m6e942b, 2026-10-27)

- Read the raw weights file, not only the dumped code, to confirm whether coefficients are exactly
  "nice" (e.g. 0.02 = 1/50) or just printed rounded.
- Identity layers (weight 1, bias 0) and single-neuron chains collapse; fold them away.
- When all weights of a neuron are equal, factor them out: `w * sum(inputs) + b`.
- Rewrite `relu(w*s + b)` with w > 0 as `relu(s + b/w) * w` so the threshold `-b/w` is expressed
  in input units (here 1000) and becomes interpretable; snap that threshold to a round number or
  a half-step between integers.
- Prefer dividing by an integer constant (/50) over multiplying by a decimal (0.02) for exactness
  and readability.
- Compute the reachable input range (here total in [0, 2040]) to check whether a ReLU is ever
  inactive/always active; if one side is unreachable, the ReLU can be removed.

## tax/seed0 — agent_1 (m6eb052, 2026-10-27)

- Use the input bounds to find dead ReLUs: if sign(a*x+b) is fixed over the whole domain the
  unit is either always 0 (drop) or always linear (fold into the output affine part).
- In 1-D, normalise each unit to `slope * relu(x - kink)`; kinks = -b/a, slopes = w_out*a.
  The function is then a piecewise-linear spline; read it as segments.
- Snap cumulative slopes (the slope of each segment) to simple fractions rather than snapping
  the individual increments; the segment slopes are what the original rule most likely used.
- Snap kinks to integers or half-steps; small output biases relative to the output range snap to 0.
- Check the snap tolerance (fraction of output range) first to know how aggressive snapping can be,
  and verify numerically on a dense grid before running the checker.

## tax/seed1 — agent_1 (m135dc5, 2026-10-27)

- For a 1-input net, compute each unit's hinge point -b/w and its contribution to the slope
  (output weight * input weight); the net is then a list of (kink, slope delta) pairs.
- Drop units that are inactive over the whole input range (sign of w and b both keep the
  pre-activation negative on [lo, hi]).
- Merge hinges that are close together into one kink; drop hinges whose slope delta is tiny.
- Hinges very close to the domain edge with small effect are usually fitting artifacts; check the
  error from removing them against the tolerance.
- If the slopes are increasing along the axis the function is convex: write it as max(line, ...)
  (decreasing slopes -> min). This removes all relu terms and makes intercepts readable.
- Snap slopes/intercepts to simple values (1/8, 0.4, integers) and re-verify; the check's snap
  tolerance (2% of output range here) tells how much freedom there is.

## tax/seed2 — agent_1 (mb93ac6, 2026-10-27)

- For a 1-D input, turn the net into a list of kinks (-b/w per hidden unit) and segment slopes;
  that is the whole function.
- Drop units whose kink lies outside the input range on the inactive side (always zero), and fold
  units whose kink lies outside on the active side into the linear term.
- Merge kinks that are close together (20.4 and 21.6) into one round breakpoint; treat a cluster of
  kinks near the domain edge with tiny net effect as fitting noise around the boundary.
- Read slopes as rates: look for simple ratios between segments (here each doubles) and round to
  simple decimals.
- Check key points (breakpoints, ends) against round numbers to confirm the intercept (here 0).
- Write the result as tiers (min/relu clamps) so the business meaning (tiered pricing/tax) shows.
- The snap tolerance is a fraction of the output range; a small slope error times a long
  interval is where the largest deviation shows up (the domain end).

## triangle/hand — agent_1 (mf5cccc, 2026-10-27)

- For a 1-D input, read each hidden ReLU as a knot at -bias/weight and the output as a sum of slope
  changes; accumulate slopes left to right to get the piecewise-linear shape.
- Output weights of the pattern (+a, -2a, +a) on equally spaced knots k-d, k, k+d form a hat:
  `a*d*relu(1 - abs(x - k)/d)`. Patterns (+a, -a) on two knots form a clamp/saturating ramp
  (`a*min(max(x - k1, 0), k2 - k1)`).
- If the slopes sum to zero the function is constant beyond the last knot - check the tail value to
  decide whether a bump (returns to 0) or a step (plateau) is meant.
- Prefer abs/min/max formulations over raw ReLU sums: they make centre, width and height explicit.

## window/hand — agent_1 (m74389d, 2026-10-27)

- A pair of ReLUs with opposite input weights (`relu(w*x+b)`, `relu(-w*x-b)`) and equal output weights is `abs(w*x+b)` times that output weight. Collapse it into one `abs`.
- A pair with opposite input weights and opposite output weights is just linear: `relu(z) - relu(-z) == z`.
- For a binary output, write the score in a form whose sign is easy to read, like `c - abs(x - m)`. It reads directly as "within c of m", and you can name the centre and the half-width.
- Before snapping, check whether the weights are already exact simple values. If they are, aim for `PASS (exact)` with an algebraically equivalent formula rather than a rewritten decision rule.
- Disagreements that sit exactly on the decision boundary with near-zero scores are float noise, not errors in the formula.

## window/seed0 — agent_1 (mbaccce, 2026-10-27)

- First check each hidden neuron's pre-activation range over the input box; neurons that are
  never positive are dead and can be deleted (and always-positive ones are linear).
- For a binary output, the magnitude of the score is irrelevant: find the decision boundary
  (zero crossings of the piecewise-linear score) and re-express it directly, rather than
  snapping individual weights.
- In 1-D, a piecewise-linear net is fully described by its kinks; between kinks solve the
  linear equation for the zero crossing.
- Several ReLUs with nearly equal kink points and the same sign of slope act as one ramp; merge them.
- A positive bias minus ramps going both ways = an interval/band; if the two boundaries are
  roughly opposite, express it with abs() (|x - c| < r).
- Snap boundaries to the nearest simple value (here ±0.5, i.e. a half-step) and accept
  disagreements that fall only inside the network's low-confidence zone.

## window/seed1 — agent_1 (mb71939, 2026-10-27)

- First compute each hidden unit's activation threshold (-b/w for 1-D inputs, or the bias range
  over the input box) and delete units that are never (or always, then linearise) active on the
  domain. Small output weights (|w| << other weights) are another hint of a vestigial unit.
- Look for pairs of units with opposite-sign slopes of similar magnitude and similar biases and
  equal output weights: relu(a*x - b) + relu(-a*x - b) = relu(a*|x| - b) when b >= 0, i.e. an
  abs-value feature.
- For binary outputs, don't snap individual weights first; solve for where the output crosses the
  threshold, then snap those crossing points (here +/-0.5), and rewrite as a simple signed margin
  (e.g. `c - abs(x)`). Overall scale of the output is irrelevant.
- When two boundaries are nearly symmetric (-0.508 vs 0.496), merge them into one shared threshold
  and use abs().
- Ignore large magnitudes inside units (slopes of 8): steep ReLUs are just the network's way to
  make a sharp step; what matters is where the step happens.

## window/seed2 — agent_1 (mfa4ceb, 2026-10-27)

- For each ReLU, compute its pre-activation range over the input bounds; drop neurons that are
  never active (or never inactive -> they are linear and fold into the bias/other terms).
- Group neurons whose kinks (x = -b/w) are close and whose output weights share a sign; they
  behave as a single ramp.
- For binary outputs only the decision boundary matters: solve output = 0 piecewise per linear
  region rather than simplifying the score itself, then snap the boundaries.
- A positive bias plus negative-weighted ramps on both sides of 0 is a "band" detector:
  rewrite as `c - abs(x - m)` with snapped centre m and half-width c.
- Mirror-symmetric boundaries (+0.5004 / -0.5004) strongly suggest abs().
- Accept PASS (snap) rather than encoding the noisy exact boundaries.

## xor/hand — agent_1 (m416985, 2026-10-27)

- With bool inputs and few of them, list every input and the raw score; the truth table often
  names the function (here XOR) straight away.
- If a hidden neuron sees only a sum s of inputs, rewrite the network as a piecewise-linear
  function of s. `relu(s) - 2*relu(s-1)` is a "tent" shape, which is `1 - abs(s-1)`, and on bools
  is `abs(x0 - x1)`.
- When the score takes only two values (±c), write it as `indicator - c` with a closed-form
  indicator (abs, min, max) so the result matches exactly, not just in sign.
- A bias of -0.5 on a 0/1-valued expression is the usual "half-step" threshold: the output is
  `expr > 0.5`, i.e. `expr == 1` for integers.

## xor/seed1 — agent_1 (m6fec39, 2026-10-27)

- When the input space is small (few booleans / small integer ranges), enumerate it fully and read off
  the truth table first; that often identifies the function directly (XOR, AND, majority, ...).
- For a binary output only the sign matters: drop scale factors and replace the score with a boolean
  expression that has the same sign pattern (comparisons yield 1/0, and 0 maps to class 0).
- Hidden units with weights (+a, -a) on two inputs and ~0 bias are relu(x_i - x_j) "i and not j"
  detectors; several units with proportional weight vectors are duplicates and can be merged by
  summing their output weights.
- relu(a - b) + relu(b - a) = |a - b|; on booleans |a - b| == (a != b).
- A unit with a large positive bias that is always active acts as an affine term (bias + linear
  correction), not a real nonlinearity; fold it into the output bias/linear part.
- Tiny biases (1e-5 .. 0.04) relative to weights of ~3 are noise: snap them to 0.

## xor/seed2 — agent_1 (m663a8c, 2026-10-27)

- For small discrete (bool) input spaces, enumerate every input and read the truth table directly;
  that beats snapping weights one by one.
- Check each ReLU's max pre-activation over the input box (sum of positive weights * hi + bias);
  if it is <= 0 the unit is dead and can be dropped. If its min is >= 0 the ReLU is the identity.
- When all live units are always-on, the net collapses to one affine function; then only the sign
  pattern (for classification) matters, and tiny coefficients that never change a sign can be dropped.
- For binary outputs, rescale the final score to small integers (e.g. +-1) with a margin around the
  threshold rather than reproducing tiny raw magnitudes.
- Beware of a checker that treats near-zero scores as ties: a trivial program can "pass"; confirm
  zero disagreements, not just the PASS label.

## iris_petal/seed0 — agent_1 (m90c92a, 2026-10-27)

- For 2-input argmax nets, first print an ASCII class map. It shows the number of regions, which
  classes are adjacent, and whether boundaries look straight.
- Find dead hidden units by sampling the input box and checking which preactivations are ever > 0.
  Drop the units that never fire.
- Bisect the class boundary along one axis for many values of the other axis and fit a line. Zero
  residual means the boundary sits in one ReLU activation pattern, so an exact linear inequality exists.
- Get the exact inequality by folding `(W2[i]-W2[j]) * mask` through layer 1 for that pattern, not
  from the fitted line. Then divide by one coefficient so it becomes 1, to make it readable.
- Only the score differences of adjacent classes matter. Classes that never share a boundary need no
  comparison, so argmax becomes an if-chain of half-plane tests.
- Snap per coefficient and check each snap against check.py. Test the extreme corners: a slope
  error grows along the far edge of the box (here at x1=2.5).
- Domain ranges are a strong hint to a known dataset (here Iris petal features), which suggests
  names and the expected structure.

## iris/seed0 — agent_1 (mf58628, 2026-10-27)

- Compute each first-layer neuron's pre-activation range on the box corners (exact for linear
  pre-activations): always-negative neurons are dead, always-positive ones are linear.
- Also test "mostly on" neurons by linearising them and counting confident disagreements; ReLUs that
  only clip in a far corner usually don't matter.
- Ablate neurons one by one and measure decision changes; drop the ones that only affect the unsure zone.
- For argmax outputs only differences of outputs matter: subtract one output and think in terms of
  pairwise margins. Try to rewrite argmax as an ordered decision list (`class A if oA > oB else ...`)
  and check it empirically.
- When a single ReLU remains, check which side of it each class lives on; often the class boundary
  only exists on one side, which turns `relu` into either `0` or the identity there and makes the
  boundary a plain linear inequality.
- Snap whole inequalities (not individual weights): normalise by one coefficient, try several
  normalisations, grid-search simple values, and keep the candidate with zero confident
  disagreements; if half-steps fail, go to quarter-steps. A snap that appears under multiple
  normalisations is more likely the "true" simple boundary.
- Re-check with different sampling seeds to make sure a snap isn't passing by luck.

## iris_petal/seed1 — agent_1 (m8e659b, 2026-10-27)

- For argmax nets with few inputs, skip the weights: plot the class map on a grid, then bisect the
  boundaries along one axis and fit lines to the boundary points. A low-input classifier is usually
  just a few half-planes.
- Check whether `boundary + k*other_input` is constant for a simple k (1, 4/3, 1/2...). If it drifts,
  look for a kink (two slopes) rather than forcing one line.
- A kink of a ReLU boundary is a max/min of two lines -> write it as `or`/`and` of two half-planes
  (union when the region is convex-outward toward the corner, intersection otherwise; test via which
  line gives the larger/smaller boundary value on each side of the kink).
- Emit a decision list ordered by the regions that are cleanly separable; the middle class is the
  fallback `return`.
- When snapping a threshold, try the round value and the half-step neighbours; the midpoint of the
  observed range of the boundary constant is often the only one that passes.
- Unsure-zone disagreements are fine for PASS (snap); concentrate on removing "net confident" ones,
  which point exactly to the badly fitted segment.

## iris_petal/seed2 — agent_1 (m116501, 2026-10-27)

- For argmax or classification nets with few inputs, skip the weights. Map the decision regions
  on a grid first. The region layout (bands, wedges, nesting) tells you the shape of the program.
- If the classes form ordered bands, write a decision list: peel off one extreme class, then
  the other, and make the middle class the default.
- Find boundaries by bisecting along one axis at many fixed values of the other. Constant
  differences give the slope. A sudden change in slope means a ReLU kink, and that usually
  becomes an `or`/`and` of two half-planes (convex or concave corner) rather than a single line.
- Normalise each boundary line to `x0 + k * x1 <op> c`, then snap k to a simple fraction (1, 1.1,
  9/4 and so on) and c to one decimal. Try the simplest candidate first, but expect to keep a
  second significant digit when the slope is far from an integer. The checker's confident zone
  is narrow enough that 2.2 vs 2.25 can matter.
- The worst counterexamples sit at the domain edges (x1 = lo/hi). That is where a wrong slope
  produces the largest offset, so check the line's fit at both ends of the range.
- Small regions near domain corners, where the net bends at the very edge, can usually be ignored
  because they fall inside the unsure zone.

## iris/seed2 — agent_1 (m815a5b, 2026-10-27)

- First find dead neurons (max of pre-activation over the input box < 0, via interval arithmetic) and
  always-on neurons (min > 0); the latter make the layer linear and can be folded into the next layer.
- Enumerate activation patterns on a big random sample, tabulate class counts per pattern; often one switching
  neuron separates which classes can co-occur, turning argmax into a decision list of pairwise boundaries.
- For argmax nets, compute pairwise differences o_i - o_j per pattern; those are the real decision surfaces.
- Normalise a boundary by its largest coefficient (or a natural one) before snapping; look for equal-magnitude
  pairs (x2 - x1) that suggest a combined feature.
- Snap with a small grid search scored only on confident points (the checker ignores the unsure zone), then
  prefer the coarsest lattice that still gives 0 errors. Coarse 0.1 lattices failed here; 0.05 worked.
- Steep boundaries (large scale of o_i - o_j) need more precise coefficients than shallow ones.
- When a single error remains, inspect where it is: usually a corner where a secondary neuron (here h1_1)
  tilts the boundary; shift the threshold toward the safe side rather than adding a branch.
- Validate with several checker seeds, not only the default one.

## iris/seed1 — agent_1 (m410c85, 2026-10-27)

- Bound each hidden unit's pre-activation over the input box with interval arithmetic
  (positive weight -> hi, negative -> lo). Units with max <= 0 are dead; units with min >= 0
  are linear and can be folded.
- For argmax outputs, look at score differences between classes rather than raw scores; units
  with nearly equal output weights across classes do not affect the decision.
- For small classifiers, label a dense uniform sample with the net, drop the points inside the
  checker's unsure zone (top-2 gap < margin), and test whether a one-vs-rest decision list of
  linear cuts separates the remaining points (LinearSVC / LP). Order the list by the most
  isolated class first.
- Snap coefficients by fixing the largest coefficient to 1 (or to a nice integer scale), solve
  a max-margin LP for the feasible region, then enumerate simple fractions and keep the
  simplest set whose confident-point gap stays positive; put the threshold in the middle of
  the gap (rounded to a half-step or quarter-step).
- If simple fractions fail at one scale, rescale the whole inequality (e.g. divide by the
  smallest coefficient) - ratios like 1:2:2.75:5 can be much cleaner than the x2=1 form.
- Verify on a larger sample than the checker uses and with other random seeds, since the
  confident gap for a linear cut can be narrow.

## iris_petal/seed4 — agent_1 (m586bdd, 2026-10-27)

- For low-dimensional argmax nets, map the decision regions on a grid first; the region picture
  tells you the shape of the program (decision list, linear cuts, kinks) far faster than the weights.
- Locate each boundary by bisection at many points along the other input; fit a line, check
  residuals for kinks (none here), then snap slope and intercept.
- Ordered classes in bands -> a chain of `if score < t: return k`, with the last class as the default.
- Try the simplest slope (1) first; if check.py reports confident disagreements clustered at one end
  of the boundary, the slope is wrong, so move to the next simple fraction (here 1.1) rather than
  shifting the threshold.
- Among thresholds that pass, prefer the one closest to the fitted boundary (fewer unsure-zone
  disagreements) while keeping it a round/half-step value.
- Hidden neurons with tiny output weights or that are inactive over the input box (here several
  all-negative-weight neurons with negative bias) are dead code; a region-level rewrite removes them
  all at once.

## iris_petal/seed5 — agent_1 (m68f02a, 2026-10-27)

- For low-dimensional argmax nets, map the decision regions first (coarse grid print) and then
  locate each boundary precisely by bisection along one input; this is far cheaper than algebra on
  hidden units, most of which only shape the logits away from the boundary.
- Measure the width of the "unsure" band (where top-two logit gap < the checker margin) along each
  boundary; it tells you how much slope/threshold error a snapped line can afford.
- For a nearly straight boundary, snap the slope to a simple value (1, 7/4, 2, ...) and pick the
  threshold as the midpoint of the range that a·x0 + b·x1 takes along the true boundary; accept the
  snap if half that range is below the band half-width, otherwise try the next simple slope.
- If two classes never share a boundary, a short if-chain of linear tests reproduces argmax.
- Recognising the dataset (here iris petal measurements) helps name variables and sanity-check the
  class order, but the network's extrapolation outside the training data still has to be matched,
  because the checker samples the whole input box.

## iris_petal/seed3 — agent_1 (mf6cdb2, 2026-10-27)

- For low-dimensional argmax nets, read the decision boundaries rather than the weights.
  Scan a grid, bisect each row, and fit `x0 + k*x1 = c` per segment (look at how x0 + k*x1
  drifts to find k).
- Watch for slope changes along a boundary. A small change can be merged into one line. A
  large one (here 0.7 vs 1.1) needs two pieces, joined with `or` when the region is convex
  toward it (a union) and with `and` when it is concave (an intersection). Check which by
  comparing each piece's prediction against the measured boundary at both ends.
- Order the tests by class band: return the extreme classes first and fall through to the
  middle class.
- Try coarser snaps one at a time with the checker. Keep the simplest one that has 0
  disagreements where the net is confident. The size of the unsure zone (5% of the 95th
  percentile logit gap) sets how much a boundary may move.
- Input ranges often identify a known dataset (here iris petals), which gives meaningful
  variable names.

## iris/seed3 — agent_1 (m10fc6a, 2026-10-27)

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

## iris/seed5 — agent_1 (m6ca160, 2026-10-27)

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

## iris/seed4 — agent_1 (m579e70, 2026-10-27)

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

