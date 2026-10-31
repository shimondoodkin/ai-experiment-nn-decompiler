# Heuristics learned by Claude Code

Appended by `nndc agent collect`, one section per refactoring attempt.

## bump/seed0 — agent_1 (m6b1950, 2026-10-31)

- First tabulate the net on a coarse grid and compare it with familiar shapes (Gaussian bump,
  |x|, products, min/max). Check the ratios at radius 0.5, 0.71 and 1 against exp(-r²):
  0.78, 0.61, 0.37.
- Before snapping anything, compare the maximum error of the clean formula with the snap tolerance.
  For a smooth target built from many sigmoids, the net's own ripples in the flat regions can be
  larger than 2 % of the range. The formula is then the right reading of the net, but no
  program can reach the net with it. Report the formula and keep a snapped net.
- If the residual against the formula looks like noise with no structure, global corrections
  (refit width, tilt, offset) will not help. Test this quickly with a minimax fit before spending
  time on it.
- Pruning: remove one unit at a time and refit the output layer by **minimax LP**, not least
  squares, because the checker measures the maximum error. Units whose removal costs almost nothing
  after a refit are redundant even if their output weight is not tiny.
- Snap in stages. Snap the hidden layer coarsely (0.1), refit the output layer, then snap the
  output weights **one at a time with re-solves**. Rounding everything at once can turn a
  passing fit into a failing one.
- Keep a safety margin. Check the maximum error on a grid finer than the fitting grid. The
  checker adds random and boundary points, so aim at about 80 % of the tolerance or less.

## bump/seed1 — agent_1 (mbf30d8, 2026-10-31)

- First evaluate the net on a coarse grid and print it; a recognisable shape (here a radial bump
  with exp(-r²) values) gives the intended formula quickly. Then check the formula's max error
  against the tolerance before investing in it: a net's own approximation error (tail ripple)
  can be larger than the snap tolerance, and then the formula alone cannot pass.
- For sigmoid layers, describe each unit as (steepness |w|, normal angle, boundary distance -b/|w|,
  output weight). Units with equal steepness and distance at many angles are walls of a polygonal
  bump. Very shallow units (|w| · box radius ≲ 1) are near-linear and are the first candidates to drop
  or replace by linear terms.
- Normalise units with `sigmoid(z) = 1 - sigmoid(-z)` so that all boundaries face outward; the sign of
  the output weights then separates "walls" from "corrections", and the constants fold into one.
- After pruning or snapping hidden weights, refit only the output layer, and refit by **minimax (LP)**
  rather than least squares. The acceptance test is a max-error test, and minimax roughly halved the error here.
- Keep more digits on output weights than on hidden weights when there are many units: rounding
  errors in the output layer add up linearly across units.
- Snap hidden weights greedily one value at a time, to coarser grids (0.5, then 0.25), with an
  error budget below the tolerance (about 60% of it) to keep margin for the checker's denser and boundary samples.
- When the ripple has to be kept, a fresh small fit (fewer units, or a formula plus a few units)
  rarely beats pruning the existing units: the existing units already encode the ripple.

## bump/seed2 — agent_1 (m23ed3c, 2026-10-31)

- First evaluate the net on a coarse grid and look for a closed form (here the radius-to-value
  pattern 0.78 / 0.37 / 0.10 gave exp(-r²) at once). Then check the formula against the net.
  The net can be close to the formula and still fail because of training ripple. Measure the
  ripple's max against the snap tolerance before spending time on formula variants.
- If the ripple is spread out (many local extrema, not one region), corrections made from a few
  sigmoids will not fix it. Simplify the network itself instead.
- For sigmoid nets, pruning with an output-only LP refit is pessimistic. A full nonlinear
  minimax refit (smooth max / log-sum-exp with rising temperature, L-BFGS, warm start from the
  original weights) removed 4 of 16 units where the LP refit could remove only 2.
- Snap one parameter at a time, nearest-to-grid first, with a refit after each snap. Keep a snap
  only while the error stays under about 85% of the tolerance, so the checker's denser and
  boundary points still pass.
  - Coarse grid first (0.25). Then a finer grid (0.05) for the parameters that refused.
  - Output weights last, using a fast LP that has the snapped weights as fixed bounds.
- Look for near-duplicate units (similar weight vector and bias). They merge or prune first.
- Name sigmoid units geometrically: normal angle, offset -b/|w| and slope |w|. Then group by the
  sign of the output weight. In a bump-shaped net, units with the same offset and opposite
  output-weight signs are the "in" and "out" walls of a plateau.
- Keep long fits in the background and write their logs to scratch files. A full refit of
  12 sigmoids on about 5k points takes 1–2 minutes.

## circle/seed0 — agent_1 (mbcdc07, 2026-10-31)

- For 2-input binary nets, first draw the decision region on a grid (ASCII is enough). Its shape (disk,
  band, quadrant, XOR pattern) usually names the formula before any weight is read.
- For a blob around a centre, bisect the boundary along rays from the centre. Near-constant radius means a
  circle `x0*x0 + x1*x1 < r*r`; a radius that varies with angle as 1/|cos| means a polygon or box.
- Read each sigmoid unit as a line: normal = (a, c)/|(a, c)|, signed distance = -d/|(a, c)|, steepness =
  |(a, c)|. Many steep units at the same distance with normals spread around the circle, all with large
  negative output weights, are tangent lines of a circle (a polygon approximation of a round region).
- Units with tiny input weights (|(a, c)| < 0.3) and tiny output weights are constants; units whose line
  lies outside the box (or with the whole box on one side) are constants on the domain.
- Snap the radius (or r^2) to a nice number using the median of the bisected radii, not the unit lines'
  distances: the polygon's tangent lines sit inside the circle (here about 1.8 for radius 2).
- Return a signed score (`r*r - x0*x0 - x1*x1`) instead of 1/0, so the program also gives the margin.

## circle/seed1 — agent_1 (m7b5369, 2026-10-31)

- **Constant units:** a sigmoid unit whose input weights are ≈ 0 (≪ the others) is a constant
  sigmoid(b); fold it into the output bias.
- **Distance-from-origin test:** for every unit compute |b|/‖w‖ (the line's distance from the origin)
  and the unit direction w/‖w‖. If many units share one distance and their directions spread around
  the circle, the net is drawing a disk/ring (polygon approximation of x0²+x1² vs r²). Opposite-sign
  output weights with opposite-sign biases are the same "outside the polygon" test written two ways.
- **Ray bisection:** for a radial shape, bisect the decision boundary along evenly spaced rays from
  the centre (the score's maximum point). Mean radius → snap to a nice value; the spread tells you whether
  it is really a circle (small, unstructured spread) or an ellipse/polygon (systematic spread).
- The radius the net learns sits slightly beyond the per-line distance (1.75 → 2 here), because the
  steep negative units only win once they are well past half-on. Use the measured boundary, not the line
  distances.
- For binary nets, return a smooth score (`r0² − r²`) instead of a hard 0/1 comparison. The decision is the
  same, and the score keeps the margin information.

## circle/seed2 — agent_1 (m962442, 2026-10-31)

- For 2-input binary nets, plot the decision on a grid first. The shape (disk, band, half-plane,
  quadrant) often names the formula right away.
- Estimate the size of a round region from the class-1 area fraction (area = pi r^2), then confirm
  it by bisecting the boundary along rays. If the radius is nearly constant, it is a circle. Snap r^2 to
  an integer.
- A ring of hidden units with similar |w| and similar |bias / |w||, whose weight directions are
  spread around the circle, is a polygon approximating a circle of radius |bias| / |w|.
- Units with weights near 0 are constants; fold them into the output bias.
- Write squares as `x*x` when `**` is not allowed. Return a signed margin (`r^2 - x0*x0 - x1*x1`)
  rather than a 0/1 comparison, so the score is meaningful near the boundary.

## dist/seed0 — agent_1 (m4d4fdb, 2026-10-31)

- Scan the output on a coarse grid first; a radial or other symmetric shape names the function
  (here a distance |x|) even when the program has to stay a sum of units.
- Compare the best closed-formula error with the snap tolerance before investing in it. A smooth
  sigmoid net approximating a cone has ripple of a few percent of the range; if a minimax fit of the
  formula family sits at the tolerance, the formula will not pass robustly.
- Sigmoid units with small input weights over the whole box are almost linear; they are the first
  to prune (their effect folds into a constant/linear term, which may then turn out unnecessary).
- Select units greedily by minimax error (LP refit of the output layer), not least squares: the
  checker judges max error, and least-squares greedy picks poorly.
- After pruning, snap input weights and biases first (one decimal kept error well under tolerance
  here; 0.25 or 0.5 steps did not), then refit the output weights by minimax and round them last.
- `-a*sigmoid(z) = a*sigmoid(-z) - a` turns negative-weight units into positive outward steps; with
  directions atan2(w1, w0) and offsets b/|w|, the units of a radial function group into rings.
- Keep a margin: aim for about half the snap tolerance on a dense grid, since the checker also probes
  boundary and corner points.

## dist/seed1 — agent_1 (md4aa70, 2026-10-31)

- Print the net on a coarse grid first. Here the grid showed the radial shape right away
  (3 on the axes at radius 3, 4.24 at the corners).
- Compare the max error of the best smooth formula with the snap tolerance. If a minimax-fitted
  formula family levels off above the tolerance and the residual is unstructured wiggle, the net's
  own error makes a closed form impossible. Stop searching for formulas and reduce the net instead.
- For sigmoid nets, units with small input weights are near-linear over a bounded box. Replace them
  with one free linear term `c + a*x0 + b*x1` instead of pruning them outright. Here that term let
  10 units fit to 0.013, where 10 units alone fit only to 0.039.
- Backward elimination with a minimax LP refit of the output layer (`scipy.optimize.linprog`,
  minimize t with |F w - y| <= t) finds the smallest unit set that passes quickly. The tolerance is a
  max-error bound, so minimize the max error, not the least-squares error.
- Snap the hidden weights first, then refit the output weights on the snapped units, then round the
  output weights. Rounding the output weights to 0.01 cost about 0.016 of max error here; keep
  margin for that.
- Include grid, edge and corner points (and the origin) in the fit set. The checker tests corners
  and boundary points, where sigmoid nets often deviate most.
- A tiny-looking coefficient (here -0.03 on x1) can still matter after a refit. Test each removal
  with the checker rather than assuming.

## dist/seed2 — agent_1 (m309d79, 2026-10-31)

- Print the net on a coarse grid (for example 13x13 over the box) before you look at the units.
  A shape that is symmetric in sign and grows linearly with radius is a norm. Check the corners
  against sqrt(2) times the edge values to tell L2 from L1 (corners = 2x edges) or L-inf
  (corners = edges).
- Smooth-activation nets cannot make kinks, so they round off |x|, norms, max and min near the
  kink. Model this with `sqrt(t^2 + c)`, where sqrt(c) is about the net's value at the kink,
  instead of trying to match the sharp formula there.
- When the tolerance is a fraction of the output range, compute the minimax (not least-squares)
  error of the best clean formula first. An LP over the linear coefficients, with a 1-D scan over
  the one nonlinear constant, is fast and gives the true floor. If that floor is close to the
  tolerance, coefficient rounding has almost no slack. Keep a few more digits rather than adding
  units back.
- Map the residual of the clean formula. If it is an unstructured wobble with many local bumps,
  it is fitting noise, and no extra closed-form term will remove it cheaply. Small linear tilts,
  such as 0.007*x1 here, are the cheapest terms that buy margin.
- Check snapped candidates on a much finer grid than the fitting grid (601x601 here). The max
  errors sit on edges and corners, so make sure the grid includes the box boundary.

## product/seed0 — agent_1 (m402373, 2026-10-31)

- Look at the hidden units' weight directions first. If they cluster on a few directions (here
  x0+x1 and x1-x0), the net is a function of those combinations. Sum and difference directions
  together with a near-zero output on both axes point to a product, via x*y = ((x+y)^2 - (x-y)^2)/4.
- Print a coarse grid of outputs over the input box. Low-order polynomials (products, squares)
  are easy to read off a table.
- Fit a small polynomial basis by least squares, then snap the coefficients: keep terms with a
  coefficient near a simple value and drop those near 0.
- Expect the largest errors at the edges and corners of the box, where sigmoid units saturate and
  the net flattens the function's growth. If those errors stay inside the snap tolerance, the clean
  formula is the better program; don't add correction terms only to follow the saturation.

## product/seed1 — agent_1 (mce1e5a, 2026-10-31)

- **Group hidden units by weight direction first.** Normalise each unit's (w0, w1) and cluster. If the units
  fall on only a few directions (here x0+x1 and x0-x1), the net is a sum of 1-D ridge functions along those
  directions. Recognise each ridge separately.
- **Directions x0+x1 and x0-x1 with opposite-sign quadratic ridges mean a product:**
  x*y = ((x+y)^2 - (x-y)^2)/4. Sigmoid/ReLU nets have no multiply, so this polarization identity is how they
  build one. Expect it whenever the units sit on the two diagonals.
- **Tabulate on a small integer/half-integer grid before fitting anything.** A 9x9 table often shows the
  formula directly (signs, zeros on axes, symmetry, values that are products or sums of the labels).
- **Expect errors at the box corners.** Sigmoid ridge sums saturate at the edges of the training box, so the
  net undershoots a convex/growing target there. If the residual is largest at the corners and small inside,
  that points to the clean formula, not a correction term.
- **Drop the output bias when it is about 0.1% of the output range** and the formula is zero where the bias
  would matter (here f(0, 0) = 0).
- **Don't chase PASS (exact) for a smooth regression net with many sigmoids.** With a tight atol, exactness
  requires every unit. A clean formula within snap tolerance is the better result.

## product/seed2 — agent_1 (me8445b, 2026-10-31)

- **Look for shared weight directions first.** If every unit's weight vector is parallel to one of a few
  directions (here (1,1) and (1,-1)), the net is a sum of 1-D functions of those projections. Rewrite it in
  u = x0+x1, v = x0-x1, and so on.
- **Diagonal pair → product.** Units on (1,1) and (1,-1) that build even curves of opposite sign mean
  x0·x1 = ((x0+x1)² - (x0-x1)²)/4. The same pattern with matching signs would be x0² + x1².
- **Print a coarse grid before simplifying unit by unit.** On a 2-input net a 9×9 table often shows the formula
  outright (sign pattern, symmetry, values at integer points), and is faster than merging 16 sigmoids.
- **Expect edge saturation in sigmoid nets that fit polynomials.** Errors that grow only at the corners of the box
  are a sign the formula is right. Don't add terms to chase them.
- **Check the error by region.** Max error on the inner box against the full box tells a correct formula with edge
  artifacts apart from a wrong formula.

## bump/seed0 — agent_1 (ma8a976, 2026-10-31)

- Evaluate the net on an integer grid first and compare it with simple closed forms. Values like
  1, 0.368, 0.135 point straight to exp(-r^2).
- Before snapping, check the *formula's* error against the tolerance and map where it fails. If
  the failures are scattered ripple, mostly at the domain edges, the net is "formula + training
  noise", and no small symbolic tweak (amplitude, width, offset) will fix it.
- Keep the formula as a basis column and refit the net's hidden units around it. Refitting the
  output layer by minimax LP after snapping the hidden weights recovers most of the rounding
  error. Hidden weights rounded to 0.1 work with a refit, but fail badly without one.
- Let the formula's coefficient float in the refit. Forcing it to 1 left the tanh units unable to
  absorb the whole residual here.
- Use minimax (LP), not least squares, when the acceptance test is a max-error bound. Include
  dense points on the domain border: tanh nets often deviate most at the box edges.
- Greedy forward and backward unit selection with LP refits quickly shows whether the residual is
  low-rank (a few units) or spread over the whole layer.
- HiGHS dual simplex can fail on large LPs. Fall back to `highs-ipm`.
- Leave margin under the tolerance (about 10%). The re-check uses more counterexamples than a
  single local run.

## bump/seed1 — agent_1 (m5bcc3a, 2026-10-31)

- Print the output on a coarse grid first. For 2 inputs, a 13x13 table is enough to recognise a
  bump, ridge, plane or step, and to read off its centre and width (here value 1/e at radius 1 means
  `exp(-r^2)`).
- Test the guessed formula against the snap tolerance using the **max** error, not the RMS. An RMS of
  0.014 looked good, but the max error was 3x the tolerance.
- If the residual (formula minus net) is an unstructured ripple across the whole domain, it is the
  network's own fitting error. It cannot be snapped away, and modelling it costs as many units as the
  net. In that case, ship the snapped net with the formula as documentation.
- Snapping recipe for tanh/sigmoid nets: round the first layer (weights and biases) to the coarsest
  precision that still passes, then **refit the output layer by linear least squares** on the
  rounded hidden units, then round the output weights. The refit absorbs most of the rounding error.
  Here it allowed 2-decimal inputs and 3-decimal outputs at 1/3 of the tolerance.
- Greedy unit pruning with an output-layer refit is a cheap test of redundancy. If dropping the best
  unit already exceeds the tolerance, stop.
- Nonlinear refits with numerical Jacobians are very slow. Use an analytic Jacobian and a few
  thousand points.

## bump/seed2 — agent_1 (m4b873a, 2026-10-31)

- **Evaluate on a grid first and guess the closed form.** Values like 1, e^-0.25, e^-0.5 and e^-1 at
  (0,0), (0.5,0), (0.5,0.5) and (1,0) identify exp(-r^2) at once.
- **Measure the formula against the tolerance before reaching for corrections.** If the residual is
  structured noise (zero mean, high-frequency, no global trend), a handful of smooth correction
  terms will not fix it. Then fall back to cleaning up the net's own mechanism.
- **Canonicalise tanh units by the sign of their output weight** (tanh(-z) = -tanh(z)). Then sort by
  direction angle and by b/|w|. Hidden geometry shows up: here, equally spaced half-planes at a common
  distance from the origin, plus a separate group of far-out units.
- **Radial bumps from tanh nets** are built as N half-planes around the centre (an N-gon), plus a few
  far-out units that cancel the leftover tail. Expect ripple of about the output weight divided by N.
- **Snap shared structure (slope, offset) before outputs, then refit the outputs by minimax LP**,
  not least squares. The checker's criterion is a max error, and LS fits leave avoidable peaks.
- **Round output weights finely (0.001) when there are many units.** Rounding errors add across units.
  Coarse snapping belongs in the shared hidden parameters.
- **Snapping to exact symmetry can fail** even when the structure is clearly symmetric. The net's
  per-unit output weights compensate for its irregular angles, so snap angles to rounded direction
  cosines, not to the ideal 360/N grid.
- **Check how much margin a passing candidate has.** The checker adds boundary-bisected points; a 10%
  margin under the tolerance on your own dense grid plus random points was enough here.

## circle/seed0 — agent_1 (m5e99df, 2026-10-31)

- Units with near-zero input weights and |bias| > 3 are saturated constants. Fold
  `w_out * tanh(bias)` into the output bias. This alone is often exact.
- For a 2-input binary net whose active units are lines at about the same distance from one
  point, with output signs that all favour that point's side, suspect a disk (or a convex
  polygon). Compute each line's distance `-b/|w|` and its angle.
- Find the decision boundary by scanning rays from the suspected centre. A nearly constant
  boundary radius means a circle: snap r to a nice value and return `r^2 - (x0^2 + x1^2)`. For
  binary outputs only the sign matters, so a quadratic score is fine.
- The tangent lines sit inside the true circle (1.7 vs 2). Their tanh softness pushes the
  boundary outward, so read the radius from the boundary scan, not from the line distances.
- Keep a mechanically folded version as an exact fallback, and use the closed-form formula when
  it passes snap.

## circle/seed1 — agent_1 (mab579d, 2026-10-31)

- **Constant units:** a unit whose |w|·(input range) is tiny compared with |b| (here |w| ~ 0.009, |b| ~ 4) is
  saturated over the whole domain. Replace it with v·tanh(b) (or v·sigmoid(b)) and fold it into the output bias.
- **Line distance:** for a tanh/sigmoid unit, the switching line is at distance -b/|w| from the origin along the
  direction w/|w|. When many units share the same distance and |w|, and their normals are spread around 360
  degrees, the net is approximating a circle (or a convex polygon) of that size.
- **Measure the boundary directly:** bisect the net's decision along many rays from a point inside the class region.
  A near-constant radius means a disc. Snap the radius to a round number inside the measured min..max range, and
  check both sides (r^2 slightly larger or smaller) to see which side the net's confident points allow.
- **Edge distance is not the boundary radius:** the decision boundary lies outside the units' line distance (2.0
  against 1.73 here), because the output threshold is crossed only after several units have partly switched.
  Use the bisected boundary, not -b/|w|, for the final constant.
- For a binary output, any score with the same sign works. A polynomial like r^2 - (x0^2 + x1^2) is the simplest
  readable form, even when the net builds it out of many linear edges.

## circle/seed2 — agent_1 (md34a30, 2026-10-31)

- **Constant units:** if |w|·(input range) is much smaller than |bias| and |bias| > ~3, the unit
  is a constant (±1 for tanh, 0/1 for sigmoid). Fold it into the output bias.
- **Antiparallel pairs:** look for pairs of units whose normalised weight vectors are parallel or
  antiparallel and whose biases mirror each other. Together they form a band (window) indicator
  `[|n·x - c| < d]`, and the output weights show the band's height.
- **Rotational symmetry:** when the bands' directions are evenly spaced (180°/k) and they all sit
  the same distance from one centre, test for a circle or ellipse instead of a polygon. Bisect
  the decision boundary along rays from the centre and check whether the radius is constant.
- **Thresholded counts:** with equal band heights, the output is a thresholded count ("in at
  least m of k bands"). Use the constant offset divided by the band height to find m. That count
  describes the network exactly, but the smooth shape it approximates (here a disc) is usually
  the intended formula.
- **Snapping radii:** round a measured radius or r² to a simple value (2, so 4) and confirm with
  the checker. A radius that varies by ±1% with angle is polygon or tanh-softness noise.
- **Exact vs snap:** for binary outputs where the true shape is curved, an exact pass would mean
  reproducing the polygon. A snap pass with no confident disagreements is the useful result.

## dist/seed0 — agent_1 (mb3252b, 2026-10-31)

- **Look at the output first.** A coarse grid and radial/angular profiles show the shape (here: radial,
  like a norm). Then test the obvious closed form against the tolerance before doing any structural work.
  If its error is several times the tolerance, keep the net's structure and simplify that instead.
- **Units with tiny input weights are polynomials.** When |w|·(input range) is small relative to the
  tanh/sigmoid scale, a unit is nearly affine over the domain. Sum all such units with the output bias and
  fit a low-degree polynomial (a plane is usually enough). Fit it jointly with the remaining units, not
  separately: the joint refit absorbs leftover curvature.
- **Ablate with refitting, not plain ablation.** To test whether a unit is needed, drop it and re-solve
  the output layer (a linear least-squares problem) before measuring error. If that still fails, the unit
  is needed.
- **Canonicalise the signs of odd activations** (tanh(-z) = -tanh(z)) so that every bias has the same
  sign. Mirror pairs (same |w|, swapped sign on one input) then become visible.
- **Snap in two stages.** First round the inner weights and biases to a coarse grid (0.05 here worked;
  0.1 did not) and re-solve the output layer by least squares. Then snap the output weights greedily, each
  to the coarsest grid that keeps the error under about 2/3 of the tolerance, refitting the free ones
  after each snap. Leave margin: the checker samples more points than your grid, including the box corners,
  where errors tend to peak.
- **Symmetrising pairs is worth one try, not forcing.** Trained nets for symmetric targets are usually
  only approximately symmetric. If the shared-coefficient version uses most of the tolerance, keep the
  separate coefficients.

## dist/seed1 — agent_1 (m62331b, 2026-10-31)

- In tanh/sigmoid nets, units with tiny input weights (argument range across the input box well
  under 1) are close to linear; replace their weighted sum with one least-squares affine (or
  quadratic) fit and fold it into the output bias and a linear term.
- Look for mirror pairs: units whose input weights are equal up to the sign of one input, with
  similar biases and output weights. Tie them (shared parameters) and the output becomes exactly
  symmetric in that input; any leftover linear term in that input usually drops to ~0.
- After pruning, refit the reduced model to the net (not to a guessed target) before snapping.
  The refit absorbs what the removed units did.
- Snap one parameter at a time, coarse grid first, refitting the free ones after each fix. Snap
  Keep a safety margin (here about half the tolerance) so later snaps still fit.
- Snap order changes the result. Run more than one order: first-layer weights first, and output
  weights or linear terms first. Keep the run whose values are roundest overall. Here, snapping
  the first-layer weights first let them round to 2 decimals. The output weights absorbed the
  difference in the refit.
- Before trying to drop a unit, check the refit error against the tolerance: a unit whose removal
  uses most of the tolerance budget leaves no room for snapping and should stay.
- Check a candidate closed form at a few points (origin, axis ends, corners) before fitting it; a
  smooth net trained on a radial-looking target can be far from the exact formula at the corners.

## dist/seed2 — agent_1 (m37857c, 2026-10-31)

- **Find groups of units that share a direction and are nearly linear.** Normalize each unit's input weight
  vector. Units with the same direction and a small pre-activation range (|w|*input range << 1)
  together form a smooth 1-D function of one projection. Replace the group with a low-degree polynomial in
  that projection (here a quadratic, error 0.03). Often only the constant and the linear term survive once
  the other units' readout is refitted.
- **Refit the readout after each structural change.** It is a linear least squares (or a minimax LP via
  `linprog` for the L-infinity tolerance the checker uses), which is cheap and exact. Snapping the hidden
  weights first and then refitting the outputs works much better than rounding everything at once.
- **Look for symmetries of the output** by comparing the net on axes and diagonals (f(x0, x1) against f(-x1, -x0),
  f(x1, x0), f(-x0, -x1)). If units map onto each other under the symmetry (use tanh oddness to
  match signs), tie their weights. This halves the parameters and makes the structure readable.
- **Test the obvious target formula against the net first, not against the task.** Here hypot is clearly
  the intended shape, but the net differs from it by 0.18 to 0.6. That decides early that a pure formula cannot
  pass, and the work should go into compressing the net.
- **Snap to a coarse grid (tenths) for hidden weights.** tanh units tolerate it, because the readout refit
  absorbs the change. Output coefficients are more sensitive (a 0.05 change times an output range of 2 is
  already 0.1), so keep two decimals there unless a grid search over nice values passes.
- **Before keeping a unit, check whether it can be dropped:** refit with that unit removed and all other
  weights free. If the error stays far over tolerance, the unit is real.

## product/seed0 — agent_1 (m4960de, 2026-10-31)

- Look at the hidden units' weight directions first. Units that come in groups with |w0| ~ |w1| along
  (1, 1) and (1, -1) suggest a function of x0 + x1 and x0 - x1; for a regression output, the product
  x0 * x1 = ((x0 + x1)^2 - (x0 - x1)^2) / 4 is the first candidate.
- Print the net on a coarse grid (for example 9x9 over the box) before snapping any weight. Zero rows
  and columns on the axes plus slopes equal to the other input identify a product immediately.
  Recognising the function from the grid is much faster than simplifying 16 smooth units one by one.
- With smooth (tanh, sigmoid) nets, an exact pass is usually impossible for a clean formula; aim for a
  snap pass and report the residual's size and where it is largest (usually the box corners, where
  the units saturate).
- Do not fit the net's error (small constant offsets, corner bends): check that the clean formula
  already passes, and keep it if it does.
- Compare the max error with the snap tolerance. If the margin is small, look at where the error
  lies; if it is only at the box edges, it is the net's approximation error, not a missing term.

## exp/seed0 — agent_1 (md7aa87, 2026-10-31)

- For a 1-input ReLU net, first list each unit's kink position -b/w (keep the ones inside the
  domain) and its slope change w*v. Units with kinks outside the domain are either constant
  there or linear there, and fold into the affine part.
- Look at the sign pattern of the slope changes. If they are all (or nearly all) positive,
  the function is convex. If they grow steadily, the curve is smooth, such as exp, x^2 or
  softplus, and not a few hinges. A few large changes with gaps between them point to a
  genuine piecewise function such as abs, clamp or max.
- Compare a few landmark values against well-known functions: f(0), f(1), f(-1) and the ends
  of the domain. f(0)=1 with f(1)=e and f(2)=e^2 identifies exp. The ratio f(x+1)/f(x)
  staying constant (about 2.72 here) confirms an exponential.
- Use the language's built-in smooth functions (exp, log, sqrt, sigmoid, tanh) as
  candidates before trying to snap the hinges. A smooth target fitted by ReLUs shows up as
  many evenly spread small kinks, and a single formula usually passes with a large margin.
- Check against the net and see where the largest error falls. Here it sat between two kinks,
  where the net's chord is furthest from the curve. This is expected for a piecewise-linear
  fit and is not a sign that the formula is wrong.
- Near-zero weights (|w| < 1e-2 with a small output weight) and kinks at the very edge of the
  domain are training leftovers. Drop them.

## exp/seed1 — agent_1 (me72b2c, 2026-10-31)

- For a 1-input ReLU net, list the kinks (-b/w) inside the domain and the slope of each segment. The
  slope sequence is the derivative, sampled. Monotone rising slopes mean a convex function. Slopes that
  are proportional to the function's value mean an exponential. Slopes that grow linearly mean a square.
- Before fitting, evaluate the net at a few landmark inputs (0, ±1, the domain ends) and compare the
  values with known constants (1, e, e^2, pi, sqrt 2, ln 2). f(0)≈1 together with f(1)≈2.72 points
  straight at exp.
- Kinks whose -b/w falls outside the input box are just linear terms (or zero) on the domain. Kinks
  that fall very close together (for example 1.657, 1.668 and 1.672 here) are one kink split across
  several units. Both can be dropped or merged when you write a compact version.
- When the net looks like a smooth function, try the smooth closed form first (the language has exp,
  log, sqrt, tanh). Its error against the net is usually the net's chord error, which stays well
  inside the snap tolerance.
- The worst disagreement between a smooth formula and a ReLU net sits in the middle of the widest
  segment where the curvature is highest. Check there first.

## exp/seed2 — agent_1 (m1c4775, 2026-10-31)

- With one input, a ReLU net is a 1-D linear spline. List its knots (-b/w that fall inside the
  domain) and the slope of each segment, then look at the shape: constant, linear, |x|, convex or
  concave, steps.
- Before you go through the units, delete the ones that are dead on the domain (relu argument <= 0
  at both ends). Fold the ones that are active on the whole domain into one affine term.
- If the slopes keep growing and the slope is about equal to the value (f' ~= f), try exp(x). Look up
  the values at simple points too: f(0)=1, f(1)=e~2.718 and f(-1)=1/e~0.368 point to exp. The same
  checks work for others: f(1)/f(0) for scaled exps, f(x)^2 for sqrt, f(x) ~= x^2 for squares.
- A smooth target fitted with ReLU always leaves an error at the knots (here about 0.02 between knots
  where the curve bends most). That error is the normal cost of the spline approximation. Accept the
  smooth formula when the max error is a small fraction of the snap tolerance. Don't snap toward the
  spline.
- Small blips in the slope sequence (one segment that goes against the trend, or very short segments
  at the edge of the domain) are noise. Ignore them when you guess the formula.

## log/seed0 — agent_1 (mdf4e99, 2026-10-31)

- For a 1-input ReLU net, compute each unit's kink -b/w. Units whose kink falls outside the
  input box are linear or zero on the domain. Fold the always-on ones into one affine term and
  drop the always-off ones. That alone gives an exact, smaller program.
- Before snapping individual weights, tabulate the net at landmark inputs (0.1, 0.5, 1, 2, 4,
  the bounds) and compare with standard functions (log, sqrt, 1/x, x^2, exp). A positive domain
  that starts near 0 (here [0.1, 5]), with net(1) close to 0 and net(2x) - net(x) roughly
  constant, points to log.
- Kink density shows curvature. Many kinks crowded near the low end of the domain, with large
  output weights, mean the function is steep and curved there (log, 1/x, sqrt). Knots spread
  evenly suggest a quadratic.
- The largest error against a smooth formula usually sits at the most curved end of the domain,
  where the piecewise-linear fit is coarsest. Judge the formula by the max error relative to
  the output range, not by single points.
- Check whether a constant offset helps by looking at the mean residual, not the max. A tiny
  mean residual means no offset.

## log/seed1 — agent_1 (mfb1032, 2026-10-31)

- For 1-input nets, first sample the output on a grid and compare it with common functions (log, sqrt,
  exp, 1/x, x^2, abs, sin) at landmark points: f(1) = 0 suggests log, f(0) = 0 with slope blow-up
  suggests sqrt. Then check the best candidate directly with the checker.
- The error of a smooth formula against a ReLU approximation is largest where the curvature is highest,
  usually at a domain edge. If the formula fails only there, the net is still that formula, and the
  tolerance decides whether it passes.
- Classify each first-layer unit by its kink k = -b/a: outside the domain it is a constant 0 or a pure
  line, so fold it into one affine term. Inside the domain it is a knot with slope change c*a.
- Rewrite every in-domain unit in "left-anchored" form: a single line valid at the left edge plus
  `d_i * relu(x - k_i)` for the knots, sorted. This is exact and makes the shape readable: the signs of
  d_i give convexity or concavity, and the slope on each segment lets you test f'(x) against candidates
  (here f' ~ 1/x, so log).
- Tiny slope changes (|d| < 1e-2) at nearly the same knot as a larger one (0.472/0.482/0.483 here)
  are training noise; merge or drop them if the exact form must be shorter.

## log/seed2 — agent_1 (m73dfc7, 2026-10-31)

- For a 1-input regression net, evaluate it on a dense grid first and compare it with common
  functions (x, x^2, sqrt, log, exp, 1/x, sin). Concave and increasing, with a zero at 1, points to log.
- Fit `a*f(x) + b` for the candidate f. If a ≈ 1 and b ≈ 0, snap them and drop the network entirely.
- Kink positions -b/w that crowd where the curvature is highest point to a ReLU approximation of a smooth
  curve, not a piecewise rule. A smooth formula is the right refactoring then.
- The largest error against the smooth formula sits at the domain edge with the highest curvature (here
  x0 = 0.1). Check that point first when estimating whether snap tolerance holds.
- Units whose kink is outside the input box are linear (or zero) over the domain. Fold them into an
  affine term rather than reading meaning into them.

## recip/seed0 — agent_1 (mae44d7, 2026-10-31)

- For a 1-input regression net, first tabulate the output on ~15 grid points and compare it with a small
  library of candidate functions (x, x^2, sqrt, 1/x, log, exp, abs, sin) and simple affine variants
  (a*f(x)+b). Products like x*y(x) ~ const or ratios y(2x)/y(x) ~ const quickly identify 1/x and powers.
- Range of the input domain is a hint: a box starting at 0.5 (not 0) suggests a function singular at 0
  (1/x, log x, sqrt behaviour near 0).
- List each unit's knot -b/w and classify it as dead / always-on / kinked inside the box. Dead units can be
  dropped, always-on units fold into a linear term, and the density of interior knots shows where the target
  curves most (dense knots where |f''| is large).
- The sign pattern of slope contributions (w_in * w_out) per knot tells convexity: all one sign of change in
  slope across knots = convex/concave function; decreasing and convex over a positive domain is the
  signature of 1/x or exp(-x).
- Expect the formula's largest disagreement with the net at domain edges and where curvature is highest;
  check that the max error there is well under the snap tolerance before accepting the formula.
- Prefer the clean closed form at PASS (snap) over an exact spline reproduction when the error is a small
  fraction of the tolerance (here about 30%).

## recip/seed1 — agent_1 (m3c6970, 2026-10-31)

- For a 1-input net, evaluate it on a grid first and look at the values at round inputs (0.5, 1, 2, 3, 4).
  Recognisable values (2, 1, 1/2, 1/3, 1/4) give the formula away faster than reading weights.
- List each unit's kink position -b/w and its slope change w*v, and keep only kinks inside the input box.
  Units with kinks outside the box are constants or straight lines; drop or fold them in.
- When all slope changes have the same sign, the function is convex (positive) or concave (negative).
  When they shrink steadily as x grows, the curve flattens: think 1/x, log x, sqrt x or exp(-x). Test each
  with products and differences: x*f(x) constant means 1/x, f(2x) - f(x) constant means log, f(x)^2
  linear means sqrt.
- Kinks that crowd where the curvature is high match where a fitted spline needs them. Their density tells
  you where the target curve bends most.
- Try the plain formula first, with no fitted scale or offset. A ReLU spline of a convex curve sits
  slightly off the curve between knots, and that error is far below the 2% snap tolerance. Fitting a*f+b
  only adds noise coefficients.
- Expect PASS (snap), not exact, when replacing a spline with a smooth formula. The largest gap sits at the
  first, strongest kink.

## recip/seed2 — agent_1 (md5ee2f, 2026-10-31)

- For a 1-input regression net, first tabulate the output at simple points (0.5, 1, 2, 3, 4) and
  compare it with candidate elementary functions (1/x, sqrt, log, exp, x^2). Values like 2, 1, 0.5, 0.333 at
  x = 0.5, 1, 2, 3 identify 1/x right away.
- Kink density tells you the curvature: the kinks of a ReLU net cluster where the target bends most, and
  the sign of each slope change (a_i * c_i, adjusted for which side is active) gives convexity or concavity. Dense,
  same-sign kinks at one end of the domain point to 1/x or log (convex/concave near small x).
- Units with |a*c| far below the output tolerance, or with kinks outside the domain, carry no structure.
  Ignore them when reading the shape.
- When a smooth elementary function passes snap, prefer it over a piecewise-linear refit. The leftover
  error is the net's own approximation error, and it is largest where the curvature is largest
  (here at the left end).
- Check the error budget early: the snap tolerance is 2% of the output range. Compute max |f - net| on a
  dense grid before running the checker, to see how much room a candidate formula has.

## sqrt/seed0 — agent_1 (m60bb32, 2026-10-31)

- **1-input nets: compute the piecewise-linear form first.** Kink positions are -b/w. Drop units whose
  kink is outside the input box, since there they are constant or linear. Evaluate the net at each kink
  and print the slope of each segment. A list of 15–20 segment slopes is much easier to recognise than
  32 units.
- **Look at the slope sequence.** If the slopes only decrease or only increase, the function is concave
  or convex, which points to a smooth function (sqrt, log, x², exp), not a logic or step function. Kinks
  crowded at one end of the range show where the curvature is largest. For sqrt that is next to 0.
- **Test candidate formulas at landmark inputs:** 0, 1, the endpoints, and perfect squares or powers of 2.
  f(1) = 1, f(4) = 2 and f(0.25) = 0.5 together pin down sqrt.
- **Compare the end slope with the derivative** of the candidate (here 1/(2·sqrt(x)) at x = 4 gives 0.25,
  against 0.26 measured). That confirms the guess cheaply.
- **The largest error is usually where the target is not smooth or has an infinite derivative** (here x =
  0 for sqrt). Check that point against the snap tolerance before trying to fix the middle of the range.
- **Most of the units are padding.** About half the hidden units have no kink in the domain, and several
  in-domain kinks change the slope by less than 0.01. Expect the real structure to be much smaller than
  the unit count.

## sqrt/seed1 — agent_1 (m1ecf17, 2026-10-31)

- For 1-input ReLU nets, list the kinks (-b/w) inside the domain and evaluate the net at them and
  at simple points (integers). Units with kinks outside the domain collapse into one affine term.
- All slope changes with one sign means convex/concave; values at integer points then usually
  identify the function (here f(1)=1, f(4)=2 -> sqrt).
- Kinks that crowd towards one end, with spacing growing away from it, point to a function with
  steep or singular slope at that end (sqrt, log, 1/x).
- Where the true function has an infinite slope or a singularity, the net's error is largest
  there. Rather than fitting the residual, add a simple clamp (max with the net's value at the
  endpoint) so the formula stays readable and passes the check.
- Try the plain formula first with the checker: its failing inputs show where a fixture is needed.

## sqrt/seed2 — agent_1 (m46a58f, 2026-10-31)

- For a 1-input regression net, first sort the hidden units by kink position -b/w. Drop units whose
  kink lies outside the domain (they are constant-on or constant-off, so they fold into an affine
  term). Then tabulate the net on a grid next to a few candidate elementary functions (sqrt, log1p,
  x^2, exp, 1/x, sin). A direct match at integer inputs (1, sqrt 2, sqrt 3, 2) identifies the
  function fast.
- How the kinks are spread hints at the curvature. Kinks bunched near one end of the domain mean
  the curvature is concentrated there (sqrt or log near 0, exp near the top end).
- A smooth target with unbounded slope (sqrt at 0, log at 0) gets an approximation error at that
  point, and the net usually sits above the true value there. Model it with a floor
  (`max(f(x), c)`) or a small shift inside the function (`f(x + eps)`) instead of fitting more
  pieces. Pick c from the net's value at the endpoint, rounded to a simple number.
- Check the max error per region instead of only overall. Here all of the error came from the
  first 0.03 of the domain.

## square/seed0 — agent_1 (mf8e26e, 2026-10-31)

- For a 1-input regression net, first sample the output on a grid (for example 41 points) and look at
  the values at the integers and at 0. Values like 0, 1, 4 at 0, +-1, +-2 show a square straight away.
- List the kinks (-b/w) inside the box and the slope change at each (w * out_weight * sign(w)).
  If they are all the same sign and spread evenly with similar sizes, the net is a convex (or concave)
  piecewise-linear interpolation of a smooth curve. Fit a low-degree polynomial instead of keeping
  the kinks.
- Units whose kink is outside the input box are linear (or zero) across the whole domain. They can
  be folded into one affine term or dropped.
- Least-squares fit of a small polynomial basis (x^2, x, 1) over a dense grid, then round the
  coefficients to simple values (here 1, 0, 0), then check. Rounding is fine when the fit
  residual is well inside the snap tolerance (2% of the output range).
- The largest error of such a snap is usually at the box edge, where the net has fewer kinks to
  follow the curve.

## square/seed1 — agent_1 (ma68cc9, 2026-10-31)

- For a 1-input ReLU net, first list each unit's kink `-b/w` and its slope change `w*v*sign(w)`
  sorted by position, and sample the output on a coarse grid. That turns the net into a readable
  spline in a few lines of code.
- Drop units whose kink lies outside the input box (they are linear there and fold into the bias
  and slope) and units whose weights are ~1e-5 or smaller (dead).
- Evenly spaced kinks with near-equal positive slope changes mean a convex quadratic: slope change
  per unit length ~ 2a gives a*x^2. Negative changes mean concave. Changes that grow or shrink
  along the axis point to cubic or exponential shapes; one big kink means abs/relu/clamp.
- Check symmetry by comparing f(x) and f(-x) on the grid; a symmetric net with f(0) ~ 0 and f(1) ~ 1
  is very likely x^2 (or |x| if the slope is constant on each side).
- Before writing a formula, compute the max error of the candidate against the net over a dense grid
  and compare it with the snap tolerance (2% of the output range). The largest error is usually at the
  domain edges, where the network has the fewest knots.

## square/seed2 — agent_1 (mfe23f4, 2026-10-31)

- For a 1-input ReLU net, compute each unit's kink x = -b/w and its slope change w*v (signed by
  which side the unit is active on). Sort by kink. This turns the net into a readable spline.
- Drop units whose kink is outside the input box (they are constant or linear on the domain) and
  units with |w*v| much smaller than the others.
- Evenly spaced kinks with equal-sign, roughly equal slope changes mean a smooth convex/concave
  curve; a slope rising linearly from -k to +k over a symmetric domain is a parabola. Sample
  f(0), f(+-1), f(+-2) to read off the coefficients (here f(x) = x^2).
- Check symmetry f(x) vs f(-x) on a grid: even functions suggest x^2 or abs(x), odd ones x or x^3.
- Test the simple formula with the checker before trying to fit corrections; with a 2%-of-range snap
  tolerance, a spline approximation of a smooth function usually passes as the formula itself.

## exp/seed0 — agent_1 (mfbc40b, 2026-10-31)

- For a 1-input ReLU net, compute each unit's pre-activation at the two box ends: if both are
  negative the unit is dead; if both are positive it is linear and folds into the bias.
- Kink positions are -b/a per unit. Kinks closer together than a few percent of the input range can
  usually be merged into one kink without leaving the snap tolerance.
- Compute the slope of every segment. If the slopes only increase, the function is convex and can
  be written as `max(line1, line2, ...)` (decreasing slopes: concave, use `min`). This is far more
  readable than a sum of ReLUs and needs no thresholds.
- Snap slopes and intercepts of the envelope lines, not the raw unit weights: errors of the lines
  are directly the output error, so the effect of rounding is easy to bound.
- Check the endpoint values of the box first; the worst error of a snapped piecewise-linear
  program is at a kink or a box end.
- Keep an exact version (dead units removed only) as a fallback before snapping.

## exp/seed1 — agent_1 (mbb6bc6, 2026-10-31)

- For 1-input ReLU nets, compute each unit's kink -b/w and which side it is active on. Drop units whose
  active side is entirely outside the domain. Fold units that are active over the whole domain into a
  linear term.
- List the segment slopes in kink order. If the slopes only increase (or only decrease), the function is
  convex (or concave), so it is a smooth curve fitted by chords. Then try exp, x^2, |x|, log and sqrt.
- Exp check: slope ≈ value at each point, f(0) ≈ 1 and f(1) ≈ e. For x^2 the slopes grow linearly
  instead.
- A smooth formula's largest error against a chord fit is usually at the domain ends, where the net has
  the fewest kinks. Measure the max error against the snap tolerance (2% of the output range) before
  you trust the formula.
- Keep an exact pruned version as a fallback when the formula's error is close to the tolerance.

## exp/seed2 — agent_1 (m88544f, 2026-10-31)

- For a 1-input ReLU net, compute each unit's kink at -b/w first. Units whose kink lies outside the
  input box are either dead or linear across the whole box; fold them into the base line.
- Then sample the slopes between the remaining kinks. If the slopes keep growing (convex) or keep
  shrinking (concave) and the hinge positions look irregular, the net is probably fitting a smooth
  curve, not a function made of hinges. Compare the values at integer points with exp, x^2, |x|,
  sqrt, log and sigmoid before trying to snap the hinges.
- Ratios between values one unit apart (here about e) and slope ≈ value (f' = f) are quick tests for
  exp.
- Check the smooth formula against the snap tolerance (2% of the output range). The largest error
  usually sits at the ends of the box, where the last linear segment cuts across the curve.
- Keep an exact hinge version as a backup: the base line on the first segment plus one
  `c*relu(x - k)` per inner kink, with c the slope change. It needs about 5 significant digits to
  pass as exact.

## log/seed0 — agent_1 (m0b0b5c, 2026-10-31)

- **Find dead units first.** On a domain with lo > 0, a unit with slope < 0 and bias < 0 is always
  0. More generally, a unit is dead when its pre-activation is ≤ 0 at both domain endpoints.
- **Find always-active units.** A unit whose kink (-b/w) lies outside the domain, with a
  pre-activation that is positive on the domain, is linear. Fold it into one slope and one
  intercept. After this, a 1-input ReLU net is a short list of (kink, slope change) pairs.
- **Drop kinks near the boundary.** A kink within a few percent of a domain bound changes the
  output only on a tiny interval. Replace the two pieces with one line fitted over the whole
  domain, a least-squares fit rather than either piece's own coefficients. Then check the max
  error.
- **Use the snap tolerance as a budget.** Measure the max error of each candidate rounding, here
  3 decimals, 2 decimals, or a fraction like 8/15. Prefer the simplest one that keeps a clear
  margin. Do not round further when no candidate is clearly meaningful.
- **Keep the exact merged form in the notes** as a fallback when the simplified form drops a
  kink.

## log/seed1 — agent_1 (m626462, 2026-10-31)

- For 1-input nets, first compute each unit's kink k = -b/w and its sign of slope; any unit whose
  active side lies entirely outside [lo, hi] is either dead (drop it) or always on (fold it into a
  base line). Kinks just outside the domain (here 0.098 vs lo 0.1) count as always on.
- Combine all output weights times unit slopes per segment to get the explicit piecewise-linear
  function; list each segment's line (slope, intercept).
- If segment slopes decrease monotonically the function is concave and equals `min` of the segment
  lines (increasing: convex, `max`). This removes every relu and the knots become implicit.
- Merge kinks that are much closer than the domain width (here 0.006 out of 4.9); the tiny segment
  between them costs far less than the snap tolerance.
- Snap line coefficients to one or two decimals and check the max error against 2% of the output
  range; adjust an intercept by 0.01 if one segment carries most of the error.
- Before settling on piecewise lines, least-squares fit candidate closed forms (a*log(x)+b,
  a*sqrt(x)+b) and compare the max error with the tolerance; a log-like shape is not proof of a log.

## log/seed2 — agent_1 (mb0cff8, 2026-10-31)

- **Classify each first-layer ReLU unit by where its kink (-b/w) lies relative to the input box.**
  - If the unit is zero everywhere on the box, it is dead: drop it.
  - If it is positive everywhere on the box, it is linear: fold it into the affine part.
  - Only units whose kink falls inside the box shape the function.
  - Kinks within about 1% of a bound behave like linear units.
- **For a 1-input ReLU net, list the segments (kinks sorted, slope per segment) and look at the slope signs.**
  - If the slopes only decrease, the function is concave: `min(line_1, ..., line_k)`.
  - If they only increase, it is convex: `max(...)`.
  - Either form is shorter and clearer than a sum of relus.
- **Compare the curve with familiar functions** (log, sqrt, 1/x, x², abs). Accept one only if its max error
  stays under the snap tolerance across the whole domain. Smooth targets often fail near the domain edge,
  where the net has few kinks: here log failed at x = 0.1. When that happens, keep the piecewise form but
  name the shape in a comment.
- **Snap each line's slope and intercept separately, then measure the max error on a dense grid** before
  running the checker. Spend the error budget on readability (one decimal place) rather than on removing
  segments. Try k-1 lines first to see whether a segment can go.
- **Fit the segment lines from net values just inside each kink**, not by adding up weight products. It is
  less error-prone, and the result is the exact line.

## recip/seed0 — agent_1 (me86c33, 2026-10-31)

- For each ReLU unit, solve w*x + b = 0 and compare the root with the input box; with the sign of w
  this tells you at once whether the unit is always off (delete), always on (fold into the linear
  part), or a real kink inside the box.
- Units with negative weight and negative bias on a positive-only input domain are dead.
- Fold all always-on units and the output bias into one line `c + m*x`, then rewrite each remaining
  unit as `a * relu(k - x)` / `a * relu(x - k)` so the kink position k is explicit and easy to snap.
- Snap the kink position first (it matters most near the kink), then the line, then hinge weights;
  check the max error of each candidate with a dense numeric sweep before running the checker.
- Compare candidate snaps against the snap tolerance (2% of the output range); choose the
  coarsest values that keep a comfortable margin.

## recip/seed1 — agent_1 (md4fe2b, 2026-10-31)

- Evaluate every first-layer kink `-b/w` against the input bounds first. A unit whose kink is outside
  the box is either dead (remove it) or always on (fold it into a linear term). Kinks just outside
  the box (here 0.496, 0.497 vs lo = 0.5) count as always-on. Check the direction, not just the location.
- After folding, a 1-input net with one surviving kink is `line + a*relu(+-(x - k))`; rewrite it as
  `max(line1, line2)` when a > 0 (convex) or `min(...)` when a < 0 (concave). That is exact and
  easier to read than the relu form.
- Express each segment by its own slope and intercept (compute the left segment as
  intercept + a*b, slope + a*w), then round digits until the exact check breaks. With atol 1e-5,
  about 5 significant digits are needed on slopes when |x| reaches 4.
- For snapping, grid-search small sets of nice values for each segment's intercept and slope, using
  max error on a dense grid against the folded exact formula. Then confirm with check.py. The error
  is largest at the domain ends and at the kink, so check those points first.
- Kink position and segment slopes interact: snapping the kink alone (1.264 -> 1.25) while keeping
  the relu form can fail where moving to a max-of-lines form with snapped lines passes, because the
  intercepts absorb the shift.

## recip/seed2 — agent_1 (md20d80, 2026-10-31)

- For a 1-input ReLU net, compute every unit's kink -b/w and drop units that are never active on [lo, hi];
  units active everywhere are linear terms. What remains is a piecewise-linear function with one piece per
  in-domain kink.
- If the slopes of the pieces increase (convex), write it as max(line1, line2, ...); if they decrease
  (concave), min(...). That is exact and removes all relu calls.
- Read off the function from a few sample values (here f(x)*x ~ 1 suggests 1/x). If the true function does
  not pass the snap tolerance itself, keep the piecewise-linear form and snap its lines instead.
- Look for symmetry of the suspected function in the lines: for 1/x, successive lines related by
  x -> 2x, y -> y/2 (intercept /2, slope /4) gives one shared snap for all lines instead of three
  independent ones.
- Brute-force snapping on a grid around each line's coefficients, evaluating the max of lines against the
  exact piecewise form, is cheap in 1-D; keep the error margin for the kinks, where it peaks.

## sqrt/seed0 — agent_1 (m34a6b3, 2026-10-31)

- For each first-layer ReLU, evaluate the pre-activation at the input box's corners (for 1 input, at lo and hi).
  Negative at every corner means always off: delete it. Positive at every corner means always on: replace
  `relu(z)` by `z`. A unit only matters if its kink `-b/w` lies strictly inside the domain.
- Negative slope with negative bias on a domain that starts at 0 is a quick sign of a dead unit; positive
  slope with positive bias is always on.
- Once the always-on units are inlined, fold everything into one affine expression and recompute the
  coefficients by multiplying out, rather than snapping each weight separately (snapping each weight
  separately adds up the errors).
- Snapping a regression line: the error of a rounded line is largest at an end of the domain, so check
  |d_offset| + |d_slope| * (max |x|) against the tolerance before trying it. If the rounded line is near
  the tolerance edge, prefer the exact coefficients, or round only the offset or only the slope.
- Count the active kinks before writing any piecewise code: a net with 8 hidden units can turn out to be
  a single line on its domain.

## sqrt/seed1 — agent_1 (mbe83fa, 2026-10-31)

- For a 1-input ReLU net, compute every unit's kink -b/w first. Units whose kink is outside the
  domain are either dead (drop them) or always active (fold w*v and b*v into one linear term).
- Rewrite the net as `c + s*x + sum d_k * relu(x - k)` with d_k = v*|w| signed by the side the unit is
  active on (a unit that switches off as x grows contributes -v*w at its kink). This form is exact
  and makes the structure obvious.
- Kinks closer together than about 1% of the domain can be merged into one kink with the summed
  slope change; the error is |d| * gap, tiny for small slope changes.
- Snap segment slopes (not the raw slope changes) to simple values, then derive the slope changes
  from them; snap the intercept last and check the worst error at the domain ends.
- Compute the snap tolerance (2% of output range) before choosing how coarse to snap; here it allowed
  rounding kinks and slopes to 0.05 steps.
- Test whether the curve is a known function (sqrt, log, x^2) by comparing at the ends and a few
  points; if it fails by more than the tolerance, keep the piecewise-linear form.

## sqrt/seed2 — agent_1 (m26e68a, 2026-10-31)

- For 1-input ReLU nets, compute each unit's kink -b/w and drop units that are dead over the whole
  input box; fold units that are always active into one affine term with the output bias.
- What is left is a 1-D piecewise-linear function with only the in-domain kinks. Write it as
  base line + sum of c*relu(+-(x - k)).
- If all the slope changes have the same sign, the function is convex (max of lines) or concave (min
  of lines). min/max of the segment lines is shorter and more readable than the relu sum, and still exact.
- Evaluate the net at integer inputs and compare with familiar functions (sqrt, log, x^2, 1/x) to name
  the function. Then check whether that formula fits within the snap tolerance everywhere, the ends of
  the domain especially. Here it failed at x = 0 because a ReLU net can't follow sqrt's infinite slope.
- Snapping segment lines to round slopes/intercepts is usually safe for regression when the error budget
  (2% of output range) is larger than the rounding. Check the worst point, which is usually a domain end
  or a kink.

## square/seed0 — agent_1 (m4b38d8, 2026-10-31)

- For each unit compute the knot -b/w and test it against the input box; units whose
  pre-activation has one sign over the whole box are constants (0 if always negative) or linear terms.
- Fold the output weight into the unit: out_w * relu(w*x + b) = (out_w*|w|) * relu(+-(x - knot)).
  Comparing the folded slopes, not the raw weights, shows which units are copies of each other.
- Look for mirror pairs (knots at +-k, same folded slope, opposite input sign): they combine into
  `s * relu(abs(x) - k)`, i.e. the function is even.
- Snap knots to round values (here tenths) and equal slopes to one shared value; check whether the
  slopes/knots coincide with chords of a simple function (x^2, |x|) to name what the net approximates.
- Snap the constant last and try a couple of candidates (0, rounded value); pick the one that
  minimises max error rather than the simplest, if both pass.

## square/seed1 — agent_1 (m6ea084, 2026-10-31)

- For a 1-input ReLU net, compute every kink position (x = -b/w), which side each unit faces
  (the sign of w) and its slope contribution (w * v). If the kinks are symmetric around a point
  and the slopes grow steadily outward, the net is a convex polynomial, most often x^2 or |x|.
  Kinks evenly spaced, with the slope increasing by about the same amount at each one, point to a
  quadratic.
- Tabulate the net on a coarse grid next to candidate formulas (x, x^2, |x|, x^3) before fitting
  anything. Recognising a familiar function is quicker than snapping weights one by one.
- Expect a net that approximates a smooth function to miss it most at the domain edges (after the
  last kink, it continues in a straight line) and at the vertex (the flat segment and the small
  bias). Those points tell you how far you are from the snap tolerance.
- When the snap tolerance is roughly twice the approximation error, take the clean formula over a
  kink-by-kink exact rewrite, which only restates the dump.
- `**` is not allowed in this language, so write x*x for a square.

## square/seed2 — agent_1 (m907a3e, 2026-10-31)

- For a 1-input ReLU net, first list each unit's kink (-b/w). Drop units whose active side never
  meets the input box: they are constant 0 there.
- Evaluate the net at the kinks and the box ends, and compute the slope of each piece. Kinks that
  are symmetric about 0 plus slopes that change sign suggest an even function such as |x| or x^2.
  Slopes that grow roughly linearly (equal slope steps at roughly equal kink spacing) mean a
  quadratic.
- Compare the net at simple points (0, +-0.5, +-1, +-1.5, +-2) with candidate formulas before
  fitting anything. Integer and half-integer inputs make x^2 easy to spot.
- A piecewise-linear net that approximates a smooth curve is least accurate at the box ends, where
  its last segment stops bending. Check the program's margin at the corners first. The snap
  tolerance is 2% of the output range.
- Prefer the clean formula (x0 * x0) over a fitted scale or offset (such as 0.98*x^2 + 0.02) when the
  clean formula passes. The fit would follow the net's interpolation error, not the function the
  net learned.

## exp/seed0 — agent_1 (md17fba, 2026-10-31)

- For a 1-input regression net, tabulate the output on a regular grid before reading weights. Compare it with
  common candidates (x, x^2, |x|, exp, log, sqrt, sin, 1/x) at landmark points: f(0), f(1), f(-1), and the
  ratio f(x+1)/f(x). A constant ratio (here e) means exp; constant second differences mean a quadratic.
- Many sigmoid units with similar positive slopes and staggered offsets, all feeding positive output weights,
  are a sum-of-steps approximation of a smooth monotone convex function. Do not snap them unit by unit; fit the
  whole curve.
- The largest net-vs-formula error is usually at the edge of the input box, where the training data ends and the
  last sigmoid saturates. Check the edges first when a formula seems close.
- If a simple formula passes the snap check by a wide margin, stop there: the remaining error is the network's
  approximation error, not structure that a refactor should keep.

## exp/seed1 — agent_1 (me3cf71, 2026-10-31)

- For 1-input regression, tabulate the net on a coarse grid first, then compare it with a short
  list of candidate functions (x, x^2, |x|, exp, log, sqrt, sin-like shapes). Key values give
  the answer away: f(0) ≈ 1, f(1) ≈ e and f(-1) ≈ 1/e mean exp.
- Fold a decreasing sigmoid into a rising one with sigmoid(-z) = 1 - sigmoid(z). Its weight
  changes sign and the constant moves into the bias. Units with nearly equal centres b/w then merge.
- If many unit centres sit at one edge of the domain, the net is building curvature there (a
  convex or concave tail). Expect a smooth nonlinear target, not a step function.
- Find the largest error of the closed formula. It usually sits at a domain edge where the
  sigmoids saturate. Check it against the snap tolerance before trying to fit coefficients:
  the plain formula may already pass.

## exp/seed2 — agent_1 (m66c83c, 2026-10-31)

- For 1-input regression nets, first tabulate the net on a coarse grid and compare with a short
  list of familiar functions (x, x^2, |x|, exp, log, sqrt, sin, sigmoid/tanh of x). A sum of
  same-direction sigmoids is just a smooth monotone curve; the formula is easier to see from the
  values than from the weights.
- Read monotonicity from signs: sign(slope) * sign(output weight) equal for all units means a
  monotone function. Steeper units placed near one end of the box signal convexity there
  (exp-like growth); units spread evenly with similar slopes signal a near-linear function.
- Check telltale values: f(0) ~ 1, f(1) ~ e, f(-1) ~ 1/e identifies exp; f(0) ~ 0 with even
  symmetry suggests x^2 or |x|.
- The worst error of a learned approximation is usually at the box edges (here x0 = 2), where the
  net saturates or undershoots; compare the formula's edge error with the snap tolerance
  (2% of the output range) before trying to fit extra coefficients.
- Prefer the clean formula over a fitted a*exp(b*x)+c when the clean one already passes snap;
  the extra constants only fit the net's training error.

## log/seed0 — agent_1 (m39cc09, 2026-10-31)

- For a 1-input regression net, first sample the output on a grid and compare it with a short list of
  standard functions (log, sqrt, 1/x, exp, x^2, sin). Use landmark values: f(1) ~ 0 suggests log, and
  f(4)/f(1) = 2 suggests sqrt. That usually finds the formula before you need to read any weights.
- When a clean formula only just fails, look at where the counterexamples are. If they all sit at a
  domain edge where the true function is steep (log or 1/x near 0, sqrt near 0), the net is underfitting
  there. A clamp on the input (`f(max(x, c))`) or on the output (`max(f(x), c)`) models that, with c a
  round number just inside the edge. Prefer this to bending the whole formula with shifts or scales.
- Compute the residual net - formula over the whole domain. If it is tiny everywhere except one end,
  fix that end and leave the formula alone.
- A sum of sigmoids with mixed signs and wide-ranging slopes (0.5 to 7.6 here) is usually a smooth fit
  of one curve, not a sum of meaningful parts. Steep units clustered at one end of the input range mark
  where the target curves most (here, the left end of log).

## log/seed1 — agent_1 (mccd557, 2026-10-31)

- For 1-input regression nets, first tabulate the net on a grid next to standard functions
  (identity, x^2, sqrt, log, exp, 1/x). A wide-range smooth monotone curve with net(1) ~ 0 and
  net(2) ~ 0.693 is a log.
- A sum of sigmoids is bounded and smooth, so it cannot follow a function that diverges at a
  domain edge (log near 0, 1/x near 0). Expect the largest error at that edge; fit the formula
  on the interior and fix the edge with a clamp (`max`/`min` at a constant near the net's edge
  value) rather than distorting the formula (shifting the argument made the interior worse).
- Measure the error profile (where |formula - net| exceeds tolerance) before adding terms;
  often the failure is confined to a tiny sliver of the domain and one clamp fixes it.
- Snap the clamp constant to a round value with margin under the tolerance, not to the
  value that only just passes.

## log/seed2 — agent_1 (mfd43c2, 2026-10-31)

- For a 1-input regression net, sample it on a grid first and compare it with the common
  elementary functions (log, sqrt, exp, x^2, 1/x). Positive-only domains like [0.1, 5] are a
  hint for log, sqrt or 1/x.
- The net-minus-formula residual tells you where the net is underfit. Large errors at a domain
  edge where the formula is steep (log near 0, 1/x near 0, exp at the top) are typical:
  a finite sum of sigmoids cannot follow an unbounded slope.
- Fix edge underfit by clamping the formula's argument (`f(max(x, c))` or `f(min(x, c))`) and
  scan c on a fine grid. The clamp copies the net's flattening and leaves the interior unchanged.
- Find the corner where the error peaks (here x = lo) and check it directly. Snap errors in
  smooth 1-D fits peak at the domain ends.
- Prefer a rounder snap constant when it still passes with margin. If a rounder value fails,
  keep the less round one that passes.

## recip/seed0 — agent_1 (m14f8ad, 2026-10-31)

- For 1-input regression nets, first tabulate the net on a coarse grid and compare against a short list
  of elementary functions (1/x, log, sqrt, x^2, exp, sin, abs). A domain that avoids 0 (here [0.5, 4])
  hints at 1/x or log.
- Many sigmoid units all with negative input slopes and positive output weights (a decreasing,
  convex sum of decreasing steps) suggest a decreasing convex function such as 1/x or exp(-x).
- The largest error of a smooth-function fit is usually at the domain edge where the target is steepest;
  check there first against the snap tolerance.
- Don't chase PASS (exact) by pruning sigmoid units: every unit contributes a non-constant term well above
  1e-5, so exact equality needs the full net. Prefer the closed formula if it passes snap.

## recip/seed1 — agent_1 (m42b72d, 2026-10-31)

- For a 1-input regression net, sample about 15 grid points first and compare the output with
  candidate formulas (1/x, sqrt, log, exp, x^2) before you look at single units.
- Test invariants: if x*y, y/x, y^2, y - log(x) or e^y is nearly constant, the formula follows
  from that directly. Here x*y ≈ 1 gave 1/x.
- A sigmoid net with many units that all have the same sign of slope and positive output weights
  is usually a smooth monotone curve. Expect a closed-form function, not a piecewise rule.
- The errors of a smooth fit are largest at the domain edges (the net flattens there). Do not
  let edge values pull the snapped constants: 1.979 at x=0.5 is still 2.
- Once a closed formula passes with snap, stop. Matching the net's ripple only adds noise.

## recip/seed2 — agent_1 (mef5f40, 2026-10-31)

- For a 1-input regression net, look at the curve first (values at a handful of points), then
  fit `a*f(x) + b` for a small library of shapes (1/x, 1/x^2, sqrt, 1/sqrt, log, exp, x^2) and
  pick the one with the smallest max error. Snap a and b to nice values and re-check.
- A sigmoid layer with many units of mixed sign and no obvious structure is usually just a basis
  for a smooth curve; don't try to interpret units one by one when a whole-net formula fits.
- Domains that start at a positive lower bound (like [0.5, 4]) hint at functions that blow up or
  are undefined at 0: 1/x, log, sqrt.
- The largest disagreement between a smooth formula and a sigmoid net is typically at the domain
  edge where the target is steepest; the net underfits there. Check the residual at the edges
  before concluding the formula is wrong.
- Read the residual (net - formula): a small oscillation around zero means the formula is right
  and the rest is fitting noise; a systematic trend means a missing term or a wrong shape.

## sqrt/seed0 — agent_1 (ma8c181, 2026-10-31)

- For 1-input regression nets, first tabulate the output at integer and simple inputs (0, 1, 2, 3, 4) and
  compare with common functions (sqrt, log1p, x^2, exp, 1/x, sin). Values like 1, 1.414, 1.732, 2 identify sqrt
  immediately.
- When the target function has a singular slope (sqrt at 0, log at 0, abs at its kink), the net smooths it, and
  the error concentrates at that point. Patch it locally with a clamp, `max(f(x), c)` or `min(...)`, rather
  than changing the formula globally. `sqrt(x + c)` shifts the error elsewhere and fails.
- Check that a patching term stays inactive away from the problem region. A line `a + b*x` used inside `max`
  with sqrt overtakes it for larger x; a constant floor cannot.
- Compute the residual net - formula on a fine grid to see where the formula is weak before reaching for
  extra terms. Here it was below 0.007 everywhere except x < 0.05.
- Pick the roundest constant that passes, within the snap tolerance.

## sqrt/seed1 — agent_1 (mf88249, 2026-10-31)

- For a 1-input regression net, tabulate the output first and compare with a short list of
  standard functions (x, x^2, sqrt, log, exp, 1/x, sin). Matching values at integer points
  (here 1, 2, 4 -> 1, 1.41, 2) identifies the function quickly.
- Sigmoid nets cannot reproduce infinite or very large slopes at the domain edge (sqrt at 0,
  log near 0). Expect the error to sit at that edge; patch it with a floor/ceiling via
  `max`/`min` rather than distorting the whole formula with shifts.
- Choose the patch with the fewest constants that passes, and prefer a simple fraction for it;
  report the alternative that is more accurate but bigger.
- Groups of hidden units with nearly the same slope and staggered offsets are one smooth ramp
  piece, not separate features; they usually disappear into the closed form.
- Check the max error location reported by the checker: if all failures cluster at one edge, the
  core formula is right and only the edge needs handling.

## sqrt/seed2 — agent_1 (m651d36, 2026-10-31)

- For a 1-input regression net, tabulate it first and compare it against a short list of
  standard functions (x, x^2, sqrt, log, log1p, exp, 1/x, abs, sin). A whole sigmoid layer often
  collapses into one such function.
- Check the endpoints of the domain separately: smooth nets cannot follow a function with an
  infinite slope (sqrt at 0, log near 0) or a pole, so the net "rounds off" there. A floor or
  ceiling with `max`/`min` repairs the endpoint without disturbing the rest of the curve; a shift
  inside the function (sqrt(x + c)) usually spreads the error over a wider range and fails.
- Scan the repair constant over a handful of simple values and pick one with margin, not the
  first one that barely passes: the empirical checker samples many points and bisects onto
  boundaries, so a borderline constant can fail on points you did not try.
- Compute the snap tolerance (2% of the net's output range here) before choosing between
  candidates; it tells you how much smoothing near the edges you can ignore.

## square/seed0 — agent_1 (m791db9, 2026-10-31)

- For 1-input regression nets, tabulate the output on a grid first and compare it with simple
  candidates (x, x^2, |x|, exp, sin, cosh) before you look at individual units. The shape is
  often obvious from 20-40 points.
- Check symmetry (f(-x) vs f(x)) and values at 0, +-1, +-half-steps. Round values at integer
  inputs (0, 1, 4) point straight at a polynomial.
- A one-layer sigmoid net whose units come in two families (positive slopes with offsets on
  one side, negative slopes mirrored on the other) is usually a smooth piecewise fit of a
  convex/even function. Read the units as spline pieces, not as features.
- Expect the largest error at the box edges: a sum of sigmoids saturates there and
  underestimates fast-growing functions. Edge sag smaller than the snap tolerance is not a
  reason to add correction terms.
- Try the simplest formula against the checker before building a piecewise version. If it
  passes with a comfortable margin (here ~36% of the tolerance), stop.

## square/seed1 — agent_1 (m1c0f5d, 2026-10-31)

- For a 1-input regression net, first tabulate the output on a coarse grid. Compare it with candidates
  (x², |x|, exp, cosh, sin, sqrt) and check its symmetry (f(-x) vs f(x)) before reading any weights.
- Fit low-degree polynomials by least squares. If the coefficients sit near small integers and higher
  terms are tiny, snap them and check.
- A sum of sigmoids that approximates a convex function loses accuracy at the edges of the training box,
  usually by under-shooting. Expect the largest snap residual there, not in the middle.
- Prefer the clean formula when it passes snap with a comfortable margin (here about 60% of the
  tolerance). Unless the net is itself piecewise simple, an exact match costs as many terms as the net.

## square/seed2 — agent_1 (m32485a, 2026-10-31)

- For a 1-input net, tabulate the output on a fine grid before reading the weights. Values at the
  integers and half-integers (0, ±1, ±2) often show the formula right away (here 0, 1, 4 means x²).
- Check symmetry: f(-x) ≈ f(x) points to an even function (x², |x|, cos), f(-x) ≈ -f(x) to an odd one.
- A sigmoid net with many units of mixed sign that adds up to a smooth curve is a basis expansion.
  Fit candidate closed forms to the whole output, not to single units. Snapping the units one by one
  keeps all the clutter.
- Fit errors are largest at the edges of the domain, where the net has few units to bend the curve.
  Read the max error's location: if it is at a bound and the mean error is near 0, the formula is
  right and the net is the inexact one.
- Compare the max error with the snap tolerance (2% of the output range). A clean formula that uses
  about half of it is a better result than a fitted polynomial that reaches exact.

## exp/seed0 — agent_1 (m1ed0cd, 2026-10-31)

- **Sample at integer and half-integer inputs and compare against known functions** (exp, x², |x|,
  sqrt, log, sin) before analysing units. For 1-input regression, the values at the integers often
  name the function directly (here 0.37 / 1 / 2.72 / 7.39 = powers of e).
- **Units whose argument barely moves over the input box are polynomial terms.** If
  |slope|·(hi − lo)/2 is small (< about 0.2) and the bias is not near 0, tanh/sigmoid is almost linear
  there. Fold all such units into one constant + linear (+ small quadratic) term by least squares.
- **A run of tanh units with increasing centres and increasing output weights builds a convex,
  growing function.** Suspect exp (or a power) when the steep end of the box gets the largest weights.
- **The net undershoots at the edge of the box where the target is steepest.** Fitting the true formula
  leaves its largest error there. Compare that error with the snap tolerance (2% of the output range)
  rather than trying to match the net's edge error.
- **Prefer the closed form over a cleaned-up copy of the units** when the closed form passes snap and
  an exact match would need nearly every unit.

## exp/seed1 — agent_1 (m21c084, 2026-10-31)

- For a 1-input regression net, sample it on a fine grid first and look at the shape before simplifying units. Check
  ratios between equal steps (constant ratio = exponential), constant differences (linear) and second differences
  (quadratic), and look at the values at 0, 1 and the endpoints for 1, e, pi, sqrt 2 and so on.
- Check the residual against the candidate formula over the whole domain. The largest errors are usually at the domain
  edges, where tanh/sigmoid units saturate and the net cannot follow a fast-growing function.
- Tanh units with |slope × domain half-width| well below 1 and a sizeable bias act as near-linear terms. Several of
  them together form a constant plus a gentle linear and curved term, and are not features of their own.
- The bend points of the steep units, at -bias/slope, show where the target's curvature changes. For a convex
  increasing target they bunch toward the high end.
- When a closed-form function from the allowed set (exp, log, sqrt, abs, min, max) passes the snap check, prefer it
  over a fitted polynomial or a pruned net. It is shorter, and it is probably the function the net was trained on.

## exp/seed2 — agent_1 (m680f63, 2026-10-31)

- For a 1-input regression net, first print a table of values on a grid. Compare the table with
  common functions (x, x², |x|, e^x, log, sqrt, sin, step) at landmark inputs: 0, ±1 and the edges
  of the domain. f(0) ≈ 1 and f(1) ≈ e point to exp.
- Measure the largest error of a candidate formula on a dense grid before calling the checker,
  and compare it with the snap tolerance (snap_tol_frac × output range).
- Smooth nets fit their curve worst at the edges of the domain, where training data is sparse
  and tanh units saturate. Expect the largest error at the edges (here at x0 = 2), not in the
  middle.
- Tanh units with a tiny slope (|w| much smaller than the input range implies) act as a
  constant plus a linear term. They are padding, not structure.
- A sum of several tanh units that gives a monotone convex curve is often a smooth elementary
  function such as exp or x². Try the function before simplifying unit by unit.

## log/seed0 — agent_1 (m012fbf, 2026-10-31)

- For 1-input regression nets, first tabulate the net on a grid next to standard functions (log, exp,
  sqrt, 1/x, x², sin) over the given domain. Reading values at x = 0.5, 1, 2 and e quickly identifies log
  (0 at 1, symmetric values at 0.5 and 2).
- A positive domain that starts slightly above 0 (here 0.1) hints at log, 1/x or sqrt: functions whose
  domain or behaviour breaks at 0.
- Many smooth tanh/sigmoid units with mixed signs and no clear role usually build one smooth curve
  together. Look at the whole input-output curve instead of simplifying unit by unit.
- Expect the largest deviation where the target is steepest or most curved, usually at a domain edge. A
  large edge error with a small interior error means the net is under-fitting there, and the formula
  should still be the clean function.
- Once a clean closed form passes the snap check, stop. Do not add fitted correction terms to chase the
  net's approximation error.

## log/seed1 — agent_1 (me7a853, 2026-10-31)

- For 1-input regression nets, first tabulate the output on a grid alongside candidate functions (log,
  sqrt, 1/x, x^2, exp, sin) and look for a value match (here net(1)=0, net(2)=ln 2, net(e)=1). This
  beats trying to simplify units one at a time.
- An input range starting just above 0 (like [0.1, 5]) hints at log, 1/x or sqrt.
- Nets fitting functions with a steep end (log near 0, 1/x near 0) lag the true function at that
  corner. When the clean formula fails only at a box corner, clamp the input (or output) just inside
  the corner instead of adding terms; measure the error by region to see this.
- Check the snap tolerance (2% of output range) against the max error before adding complexity: a
  tiny fix at one edge is usually enough.

## log/seed2 — agent_1 (md3d98f, 2026-10-31)

- For 1-input regression, tabulate the net on a grid and print candidate elementary functions next
  to it (sqrt, log, 1/x, x^2, exp). Fixed points like f(1) = 0, f(2) = 0.693 identify log at a glance.
- Domains that start just above 0 (e.g. [0.1, 5]) hint at log, 1/x or sqrt, which are undefined or
  steep at 0.
- When the obvious formula fails only at a domain edge, look at where the function is steepest: a
  small smooth net under-shoots steep regions. Clamp the formula there (max/min with the net's edge
  value) instead of hunting for a different formula.
- Compare error against the snap tolerance (2% of output range) early; it tells you how much edge
  deviation can be absorbed and whether a fix is needed at all.
- Prefer the fix that keeps the true formula intact over most of the domain (a floor) over one that
  shifts it everywhere (an additive offset inside the log), even if the shift has slightly lower error.

## recip/seed0 — agent_1 (m08d07b, 2026-10-31)

- For 1-input regression nets, tabulate the output on a grid next to a few candidate
  functions (x, x^2, sqrt, log, exp, 1/x) before simplifying unit by unit. A smooth tanh/sigmoid sum
  is often just a smooth approximation of one elementary function, and spotting that function beats
  merging units.
- Read off values at integer inputs (f(1), f(2), f(3), f(4)): ratios such as f(1)/f(2) = 2 and
  f(1)/f(3) = 3 identify 1/x right away; constant differences point to log, constant ratios to exp.
- Expect the largest error at the domain edges, and most of all where the target is steepest.
  Small nets fit the steep end poorly, so a clean formula that is off only at an edge is still the
  right answer.
- Try the clean formula with no fitted constants first. Add a scale or offset only if the check
  fails; here it passed with room to spare.

## recip/seed1 — agent_1 (m1ea677, 2026-10-31)

- For 1-input regression nets, first tabulate the output on a coarse grid next to candidate
  elementary functions (x, x^2, sqrt(x), log(x), exp(x), 1/x). A positive domain that excludes 0
  (here [0.5, 4]) is a strong hint for 1/x, log(x) or sqrt(x).
- Read values at "nice" inputs (1, 2, 4): f(1) ~ 1, f(2) ~ 1/2, f(4) ~ 1/4 identifies 1/x
  immediately; f(1) ~ 0 would suggest log, f(4) ~ 2 sqrt.
- Confirm a guessed basis function g with a linear least-squares fit of a + b*g(x); if a ~ 0 and
  b ~ 1 (or another simple number), snap them.
- Check the residual's shape: an oscillation with several sign changes is fitting noise of the
  hidden units and can be dropped; a systematic trend means a missing term.
- Expect the largest error where the target has the most curvature (the steep end of 1/x near the
  low bound); smooth-activation nets underfit there. Check that this edge error is still within
  the snap tolerance rather than adding correction terms.
- Individual hidden units of such a net are not meaningful one by one (they overlap and cancel);
  look at the total function rather than trying to simplify unit by unit.

## recip/seed2 — agent_1 (m4adf96, 2026-10-31)

- For a 1-input regression net, first evaluate it on a grid and compare with simple
  functions (1/x, sqrt, log, exp, x^2, |x - c|) before you look at single units. With
  smooth (tanh/sigmoid) hidden layers the units are usually a basis, not features.
- A positive input range that starts away from 0 (here [0.5, 4]) and outputs that fall
  steeply and then flatten point to 1/x. Check f(1) = 1 and f(2x) = f(x)/2.
- The biggest fit error tends to be at the edge of the domain where the target is steepest.
  When the edge error stays under the snap tolerance, the clean formula is the right answer.
- Measure max |formula - net| against the snap tolerance (2% of the output range) yourself
  before you run the checker. That tells you at once whether a closed form is good enough or
  needs a correction term.
- Do not chase PASS (exact) when the leftover error is a small wiggle that changes sign. That
  is approximation noise from training, not structure.

## sqrt/seed0 — agent_1 (m577b06, 2026-10-31)

- For 1-input regression nets, plot or tabulate the output before touching the weights, and compare the
  table against a few standard functions (sqrt, log1p, x^2, exp, 1/x, sin). A smooth monotone curve with
  value f(4) = 2, f(1) = 1, f(2) = 1.414 is sqrt; checking 2-3 "landmark" inputs identifies it quickly.
- Smooth activations (tanh/sigmoid) cannot follow singular behaviour such as sqrt's vertical slope at 0,
  log at 0 or 1/x near 0. Expect the formula to fit everywhere except a small boundary region, and patch
  that region with a clamp (`max(f, c)` / `min(f, c)`) or a short linear piece, rather than distorting the
  formula (e.g. sqrt(x + c)) everywhere.
- Pick the clamp constant from the net's value at the boundary (here 0.112), then round to a simple
  fraction that leaves margin under the snap tolerance (1/8 rather than 0.1).
- The snap tolerance is 2% of the output range; read it from the first FAIL report and use it to
  budget the error of each simplification.

## sqrt/seed1 — agent_1 (m9a3d7c, 2026-10-31)

- For a 1-input regression net, tabulate it first and compare with common functions (sqrt,
  log1p, x^2, exp, abs, steps). Whether it is concave, and its value and slope at the domain edges,
  usually tell you which function it is.
- Smooth nets cannot follow singular slopes (sqrt at 0, log near 0, the corners of abs). Expect
  the error against the ideal formula to sit at that point, and handle it on its own instead of
  dropping the formula.
- To patch a singular point, first try a C1 tangent-line extension at a round knot (here 1/16,
  giving 1/8 + 2x). Its constants come from the formula itself, so you don't fit any free numbers.
  A constant floor (`max(f, c)`) is simpler but less accurate.
- Check the simplest formula first. The checker's max error and where it occurs show which region
  needs a patch.
- The snap tolerance is 2% of the output range. For a function with a large range, a small
  rounded-off region near a singularity often fits in it, so a short piecewise patch is enough.

## sqrt/seed2 — agent_1 (m6e4417, 2026-10-31)

- For a 1-input regression net, tabulate it at round inputs (quarters, perfect squares) and compare
  with candidate functions (sqrt, log1p, x^2, exp). Values like 0.497, 0.998, 1.498, 1.996 at 0.25, 1,
  2.25, 4 point straight to sqrt. Then fit `a*f(x) + b` by least squares to read off scale and offset.
- Smooth activations (tanh, sigmoid) cannot follow an infinite slope (sqrt at 0, log at 0). Expect the
  largest error at the end of the domain where the target's slope blows up, not in the middle.
- Fix such an end with a clamp (`max(f(x), c)` or `min`), not by shifting the argument: a shift moves
  the whole curve near the end, a clamp changes only the region where the net flattens.
- Choose the clamp level between the net's value at the boundary and its value where the clamp meets
  the formula; that balances the two worst errors. A round number in that range (here 0.15) keeps the
  formula readable.
- Look at the error over the whole domain before snapping: if the formula is within a few percent
  everywhere except one end, only that end needs a special case.

## square/seed0 — agent_1 (me6ccd5, 2026-10-31)

- First, print the net on a dense 1-D grid (or along axes). Reading the shape (symmetry, values at
  0, ±1, ±2) often gives the formula faster than reading the weights.
- Hidden units whose slope × input range is much smaller than 1 are effectively constants. Fold them
  into the bias before you reason about the rest.
- Pairs of units with opposite-sign slopes and mirrored offsets, and output weights of the same sign,
  point to an even function (|x|, x², bowls). Matching odd pairs point to an odd function.
- In tanh/sigmoid nets, the largest error of a snapped formula is usually at the box edges, where
  the units start to saturate. Check the corners first.
- Try the simplest polynomial (x*x, |x|, x) before a fitted one. The snap tolerance (2% of the output
  range) often accepts it.

## square/seed1 — agent_1 (m373803, 2026-10-31)

- **Units with tiny input slope are lines.** If |w| * (input range / 2) is small (< ~0.1), the
  unit is linear over the whole domain; replace all such units by one fitted line `a + b*x` and
  fold `a` into the output bias. Check whether the slope `b` cancels against the remaining units
  before keeping it.
- **Plot / tabulate the output first.** A few sample values (here 0, 1, 4 at x = 0, 1, 2) often
  reveal the target (x^2), but test the simple formula against the tolerance before accepting it:
  nets saturate at the domain edges, and the edge error is usually what breaks a clean formula.
- **When the clean formula misses at the edges, keep the net's own basis.** Two opposite tanh
  steps at +-s give a smooth bowl or bump `A*(tanh(k(x-s)) - tanh(k(x+s))) + C`. Fit A, k, s, C,
  then snap to nice values with a small grid search over (A, k, s, C) and pick the lowest max error.
- **Symmetrize pairs.** Two units with near-equal |slope| and near-opposite positions usually
  encode an even (or odd) function; give them one shared slope, one shared position magnitude and
  one shared weight. Use tanh(-z) = -tanh(z) to put both in the same orientation first.
- **Prefer snapped k*s products that are nice too** (here 5/6 * 1.8 = 1.5), so the code reads as
  `tanh(k*x +- c)` with simple constants.
- **Error budget:** compute the snap tolerance (2% of output range) up front and aim for well under
  half of it, since the checker probes the domain corners and boundaries.

## square/seed2 — agent_1 (ma36559, 2026-10-31)

- **Near-constant units:** a unit with a tiny slope (|w|·input range much less than 1) is a
  constant, or at most a small linear term, over the box. Fold all such units into one
  constant-plus-line, and check the line's slope before dropping it.
- **Mirrored pairs:** two units with opposite-sign slopes of similar size, similar offsets and
  same-sign output weights form an even bump or bowl. Snap them to a shared |slope|, offset and
  weight: `c − b·(tanh(o + kx) + tanh(o − kx))`. For a 1-input regression, a bowl shape like this
  usually means x², and a bump means something like 1/(1+x²) or exp(−x²).
- **Sample before modelling:** evaluate the net at integer and half-integer inputs and compare it
  with obvious candidates (x², |x|, x³, …). Here the values were close to 0.25, 1, 2.25 and 4.
- **When the plain formula just misses the tolerance:** look at the residual. A smooth loss that
  grows toward the box edges is tanh saturation. A higher even power with a round denominator
  (x⁶/1000) fixes it without any fitted-looking constants. An interior wobble the size of the
  residual means the net itself is noisy, so stop snapping further.
- **Tolerance:** snap tolerance is 2% of the output range. Compute it (here 0.078) before you choose
  between the bare formula and a corrected one.

## bump/seed0 — agent_1 (mf83a1c, 2026-10-31)

- **Look first:** print the net's output on a coarse grid. A radial pattern points to a function of
  `x0² + x1²`, and here exp(-r²) fits by eye.
- **Fit the formula against the net, not the guess:** compute the minimax error of the candidate formula,
  including a fitted scale, rate and offset. If it exceeds the snap tolerance, the net's own approximation
  ripple is too large. Stop snapping to the smooth formula and keep piecewise-linear structure instead.
- **Remove dead units exactly:** a unit is dead if it is never active over the input box, corners included.
  Unit distance from the origin is `-b/|w|`. Compare it with the corner radius (3·√2 ≈ 4.24 here), not the side.
  Units with tiny |w| and tiny output weight are also negligible.
- **Prune with a refit:** use greedy backward elimination and refit the output weights each time, with a
  free affine term so removed always-active units are absorbed. Use an IRLS-weighted least squares to
  approximate the minimax error. Stop with some margin, for example 75 to 85% of the tolerance, because the
  checker probes near activation boundaries.
- **Round after pruning:** round the hidden weights to 2 decimals, then refit the output weights. That gives
  short numbers for a small error cost.
- **Use one canonical ramp form:** use `relu(z) = z + relu(-z)` to make every ramp zero at a reference point,
  such as the bump centre. The ramps then split into "falling" ramps (near the centre, negative weight) and
  "flattening" ramps (further out, positive weight), and a ReLU approximation of a bump becomes readable as
  `peak - cone + floor`. Sort the ramps by direction angle.
- **Check the units' distances:** the start distances of the cone and floor units (about 0.2 and about 1.2
  here) show the bump's radial profile: a linear fall from radius 0 to about 1.2, then flat.
- **Watch the parser:** when you rewrite a dump with a script, watch for constants written as `- 0.09` with a
  space. A naive split drops the sign, and here that caused an offset of exactly 2·c.

## bump/seed2 — agent_1 (m1eeff0, 2026-10-31)

- Tabulate the net on a coarse grid first. A radially symmetric peak of height 1 that falls to
  0.37 at r = 1 is `exp(-r^2)`. Check it on 3–4 points (r^2 = 0.25, 1, 2, 2.25).
- When the clean formula fails only by a small margin, measure the best achievable smooth fit
  (a minimax fit with free amplitude, width, centre and offset). If even that fails, the excess
  is the net's own approximation noise. Stop searching for a better formula; patch the noise.
- For 2-input PWL nets, patch noise with local pyramid bumps
  `h * relu(1 - (|x0-a| + |x1-b|)/w)`, picked by a sparse (L1) LP over a dictionary of centres
  (the residual's local extrema) and widths. Reweighted L1 removes a few more. Keep the formula
  and the patch visibly separate, so a reader can drop the patch.
- Solve the LP against a margin below the tolerance (about 93% of it). Re-check on a different
  point sample before rounding: the LP overfits the points it saw, and coefficient rounding adds
  error where bumps overlap.
- Pruning units with an output-layer refit is the fallback when no formula exists. Here it gave
  an opaque net about half the size, which was worse for readability than formula + patch.
- Units with all weights about 1e-5 or smaller are dead (here h1_10, h1_36, h1_37, h1_50).
  Remove them first.

## circle/seed0 — agent_1 (m757c54, 2026-10-31)

- Split the hidden units by bias sign and weight size. Units with small weights and large positive
  biases are always on inside the box: fold them into one constant (plus a linear term if needed).
  Units with large weights and biases of similar size pointing in many directions are ridges, and a
  ring of them is a polygon approximation of a radial function (norm / circle).
- For a 2-input binary net, draw the sign on a coarse grid before reading weights. The shape
  (disk, half-plane, band, XOR quadrants) usually gives the formula at once.
- To recognise a circle, scan the zero crossing along many rays from the centre. A near-constant
  crossing distance r means `r^2 - x0^2 - x1^2`. Its spread tells you how much tolerance the snap
  needs, and the centre can be found the same way if it is not the origin.
- Snap the radius to a round value (here 2.0, from crossings at 1.99 to 2.01). With x² written
  as x*x (no `**` in the language), the squared form avoids needing sqrt.
- Units whose output weight is about 1e-3 of the others can be dropped.

## circle/seed1 — agent_1 (med716f, 2026-10-31)

- For a 2-input binary net, draw the decision region on a coarse ASCII grid first. The shape
  (disc, band, half-plane, quadrant) usually suggests the formula right away.
- Many first-layer units with similar-magnitude weights, normals spread over all angles, and
  similar biases (similar distance |b|/|w| from one centre) approximate a circle or radial
  function. Get the centre and radius by finding the boundary crossing along rays.
- Units with a large positive bias and tiny weights are always on over the input box. Fold them
  into a constant plus a linear term; often they only set the level of the score.
- Units with weights and output weight of ~1e-8 are dead; drop them.
- For a binary output the score only has to keep its sign. A quadratic like r^2 - x^2 - y^2 is a
  fine replacement for a polygonal ReLU score, and its disagreements stay near the boundary,
  where the net itself is unsure.
- Snap the measured radius to a round value (1.98-2.0 → 2); the checker's unsure zone absorbs
  the facet error.

## circle/seed2 — agent_1 (m5820dc, 2026-10-31)

- Split hidden units by |bias| vs weight norm times the input range: units whose bias keeps them
  always on (or always off) over the box are affine (or zero) and fold into the output constant.
- Many ReLU units with similar bias/weight-norm ratio, weight directions spread around the circle
  and same-sign output weights are a piecewise-linear approximation of a radial function
  (|x| or |x|^2); the boundary is then a circle/sphere centred where those lines are symmetric.
- For binary 2-input nets, print an ASCII map of the decision region first, then measure the
  boundary along rays from the apparent centre; a near-constant crossing radius means a disc,
  and snapping the radius to the nearest integer/half usually lands in the unsure zone.
- For binary outputs, return a smooth score (e.g. r^2 - x^2 - y^2) rather than the comparison, so
  the sign carries the decision and the magnitude still says how confident it is.
- Drop units whose output weight times maximum activation is far below the score margin.

## dist/seed0 — agent_1 (mfb330e, 2026-10-31)

- Look at the biases first. If nearly all ReLU biases are about 0 (and the units with large biases have
  tiny output weights), the net is positively homogeneous of degree 1. Then the function is fixed by its
  values on the unit circle, so probe f(r*cos t, r*sin t) for several r and t.
- If f is linear in r and the same in every direction, it is the Euclidean norm. If it changes with
  direction, compare with |x0|+|x1| (corners at 6) and max(|x0|,|x1|) (corners at 3), or a weighted norm.
- Many ReLU units whose weight directions are spread around the circle, all with positive output weights,
  are the sign of a convex, rounded function such as a norm. A few sharp directions mean abs/max.
- Drop units whose weights are around 1e-7 or whose output weight times the largest activation on the
  input box is far below the tolerance.
- Probe the net on a coarse grid and on rays before you snap individual weights. When a known formula
  fits the probes, try it in the checker first: it can replace dozens of units in one step.
- A small constant left at the origin (here 0.03) is the net's fitting error. Leave it out unless the
  checker needs it.

## dist/seed1 — agent_1 (m8f5789, 2026-10-31)

- Look at the output layer first. If every significant output weight is positive and most hidden biases are near
  0, the function is convex and close to positively homogeneous (f(t*x) = t*f(x)), so try norms: L1, L-infinity
  and L2.
- Tell the norms apart by the value at a unit diagonal point: L1 gives 2, L2 gives 1.414, L-infinity gives 1. Then
  check that the output is constant on circles (L2) or on squares (L-infinity).
- Many live first-layer units with directions spread around the circle, rather than aligned to the axes, point to
  a rotation-invariant function (L2). Axis-aligned units point to L1 or L-infinity.
- Units whose input weights or output weights are about 1e-5 or less are dead and can be dropped at once.
  Units with large biases and output weights of about 1e-3 are fitting corrections.
- Expect the largest error at the kink of a cone (here the origin). A ReLU net cannot reproduce the sharp tip, so
  a small offset there is approximation error, not part of the formula.

## dist/seed2 — agent_1 (md15c23, 2026-10-31)

- Before reading the weights, evaluate the net on the integer grid of the input box and compare it
  with a few candidate formulas (|a|+|b|, max(|a|,|b|), sqrt(a^2+b^2), a*b, a^2+b^2). A table of
  values usually identifies the function faster than the units do.
- Many bias-free ReLUs whose directions spread around the circle, all with positive output weights of
  similar size, point to a convex positively homogeneous function: a norm. Equal weights at all angles
  mean the Euclidean norm. Weights only on axis directions mean L1, weights only on diagonal directions mean L-inf.
- Units with weights ~1e-9 are dead. Units with tiny output weights (<0.005), or units whose kink lies
  far out with a small slope, are fine-tuning and can go once a formula is found.
- A ReLU net cannot be exactly zero at a norm's apex. Expect the largest error at the
  non-smooth point (here the origin), and compare it with the snap tolerance instead of
  adding correction terms.

## product/seed0 — agent_1 (md38a5e, 2026-10-31)

- **Evaluate the net on a coarse grid first.** For 1–2 inputs, a printed 9x9 table often shows the
  function at a glance (here, a multiplication table). That is quicker than reading 64 units.
- **Check the hidden-unit directions.** When most first-layer weight vectors are near (a, a) or (a, -a),
  the net works in s = x0 + x1 and d = x0 - x1. Hinges spread along both diagonals, with the
  output-weight signs set by the direction, mean (s² - d²)/4 = x0·x1. Hinges along one diagonal only,
  with a V shape, mean |x0 - x1| or a similar term.
- **Confirm a guess with least squares over a small basis** (the guessed term plus low-order
  polynomial terms). Snap coefficients within about 1% to integers, and drop terms whose coefficients
  are below about 0.5% of the output scale.
- **A smooth formula will not be `exact` against a ReLU net.** If the check reports `snap` with the
  largest errors at the corners or edges of the domain, that is the piecewise-linear fit running out of
  hinges, not a missing term.
- **Weights at or below about 1e-4 that also have tiny output weights are dead units.** Remove them
  before reading the rest.

## product/seed1 — agent_1 (m233f07, 2026-10-31)

- Delete units whose input and output weights are both tiny (< 1e-4); they are dead.
- Look at the directions of the first-layer weights first. If they cluster on (1, 1) and (1, -1)
  with output weights of opposite signs per cluster, suspect a product x0*x1 (polarization identity
  ((a+b)^2 - (a-b)^2)/4). Clusters only on one axis with same-sign outputs suggest a square or abs.
- Before snapping individual weights, evaluate the net on a small grid (integer and half-integer
  points). A regression net's table usually reveals the formula directly (here: a multiplication table).
- Errors of a ReLU fit to a smooth function are largest at the corners of the input box; a formula
  that misses only there is the intended one, not a wrong one.
- A smooth formula cannot pass `exact` against a piecewise-linear net; accept `snap` when the
  residual is plainly fitting noise.

## product/seed2 — agent_1 (m142a3b, 2026-10-31)

- Evaluate the net on a coarse grid (for example 9x9 over the box) before reading weights. Many
  64-unit nets are smooth textbook functions, and the grid shows the shape at once.
- Signs of a product: zero on both axes, odd in each input, rows that are linear in the other
  input, corner values equal to +-lo*hi.
- Hidden units with weights in mirrored pairs along the directions (1,1) and (1,-1), with
  output weights of opposite sign between the two families, point to x*y built as
  ((x+y)^2 - (x-y)^2)/4 from two ReLU splines of a square.
- Units whose weights are all below ~1e-3 can be dropped first; they rarely matter.
- For a smooth nonlinear target approximated by ReLUs, aim for PASS (snap) with the clean formula;
  the residual is spline error, largest at the domain corners, and it is not worth encoding.

## dist/seed0 — agent_1 (mbb53df, 2026-10-31)

- All hidden biases near 0 plus all-positive output weights suggests a convex, positively homogeneous
  function such as a norm (|x|, |x0|+|x1|, max(|x0|,|x1|), sqrt(x0^2+x1^2)). Check the value at the
  axis ends and at the corners: their ratio tells the norms apart (sqrt(2) for L2, 2 for L1, 1 for L-inf).
- Units whose directions are spread evenly around the circle, rather than lined up with the axes, point
  to a rotationally symmetric target (L2), not L1 or L-inf.
- Before snapping, fit the candidate formula with a free scale and offset (`a*f + b`). If a ≈ 1 and b ≈ 0,
  the formula is the function itself.
- A ReLU net that approximates a smooth or irrational function has its largest error where the target has a
  sharp feature (here the cone tip at the origin). Check that the snap margin there is not used up before
  adding correction terms.
- To find dead units, evaluate each unit's affine maximum over the input box (sum of |w|*bound + b). A
  negative maximum means the unit is dead.

## dist/seed1 — agent_1 (m52f251, 2026-10-31)

- Before reading the units, evaluate the net on a coarse grid and look for symmetry (sign flips, swapped
  inputs) and for known values at the edges and corners (3 and 3*sqrt(2) here). That identifies norms,
  products and similar formulas faster than going through the weights.
- Fit candidate closed forms (L2 norm, L1 norm, max-norm, product, ...) as `a*f + b` with least squares
  and compare max errors; the right one has a near 1 and b near 0.
- Smooth or non-linear targets learned by a ReLU net are wrong at their kinks or tips: the net rounds
  off a cone's apex. When the clean formula fails only there, read the net's value at that point
  (here net(0,0) = 0.1) and use it as a smoothing constant, e.g. `sqrt(r^2 + c^2)`, instead of a
  `max`/clamp or a patch with extra cases.
- When several candidates pass, keep the one with the smaller max error if it is no bigger.

## dist/seed2 — agent_1 (m1e46db, 2026-10-31)

- All hidden biases near 0 means the hidden layer is positively homogeneous: probe along rays from the
  origin. If output/radius is constant per direction, the function is homogeneous of degree 1.
- If additionally the ratio is the same in every direction, it is the Euclidean norm
  `sqrt(x0*x0 + x1*x1)`. If it varies with direction like |cos| + |sin|, try the L1 norm; if like
  max(|cos|, |sin|), try the L-infinity norm (`max(abs(x0), abs(x1))`).
- Mostly positive output weights over ReLUs of near-origin lines spread around all directions = convex
  function built from tangent planes (a norm, a distance, or a quadratic bowl). Check symmetry under
  x -> -x and swapping inputs before guessing a formula.
- A small constant at the origin (output bias plus leftover hidden biases) is a fitting artefact; drop
  it when it is within the snap tolerance rather than adding a constant to the formula.
- Sample the net in numpy from `weights.npz` directly (keys W0, b0, W1, b1) to explore quickly, then
  confirm with `check.py`.

## product/seed0 — agent_1 (m1fa0f4, 2026-10-31)

- Before reading the weights, evaluate the dump on a coarse grid. A table that is zero on both
  axes and has sign pattern +,-,-,+ by quadrant is a product `x0*x1`. Units in pairs along (1, 1)
  and (1, -1) are the ReLU form of `((x0+x1)^2 - (x0-x1)^2)/4`.
- Fit the clean formula first, then look at where the residual exceeds tolerance. If the
  residual is concentrated where |target| is largest (box corners), the net is saturating. Model
  that with an odd shrink term (`p - p^3/k`, or a scale just below 1), not with clamps or extra
  kinks.
- Prefer the candidate with margin over the one that passes by a hair. The checker's input set
  can grow with counterexamples.
- A ripple of a few percent left over from a piecewise-linear fit of a smooth function is noise.
  Do not try to reproduce individual kinks when the snap tolerance already covers them.

## product/seed1 — agent_1 (m9d749e, 2026-10-31)

- Evaluate the dump on a coarse grid of the input box first. A table of 9x9 values is often enough to
  recognise a classic function (product, abs, max, sums).
- Rewrite each unit's direction in rotated coordinates (`x0+x1`, `x0-x1`, or more generally in the
  candidate basis). Units grouped by direction, with output weights of one sign per group, reveal
  convex/concave 1-D pieces. Convex in the sum and concave in the difference means a product
  (polarisation identity).
- A unit whose `|output weight| * |input weight vector|` is tiny compared with the others is a
  constant or noise over the box. Drop it.
- When the ideal formula fails only at the corners or edges of the domain, the net is extrapolating
  badly where the training data was thin. Model that with small `relu(|direction| - k)` corrections at
  a round knot, rather than giving up the formula.
- Find the residual (net - formula) on a grid. Fix only the cells beyond tolerance, and check that
  the remaining ripple is below tolerance before adding more terms.
- Grid-search snapped corrections (round knots, coefficients in steps of 0.05) against a dense grid
  before calling the checker. Keep the simplest one that passes with some margin.

## product/seed2 — agent_1 (m6949cb, 2026-10-31)

- Before reading weights, print the output on a coarse grid. Zero rows/columns along both axes
  plus sign flips by quadrant point to a product `x0*x1`; fit `c·f + d` for a few candidate
  formulas (product, |x0±x1|, squares, sums) and keep the one with the smallest max error.
- A wide ReLU layer (16 units for 2 inputs) whose weights have no clean pattern, all of similar
  size with mixed signs, is usually a piecewise-linear approximation of a smooth function, not
  a sum of meaningful kinks; look for the smooth formula rather than snapping each unit.
- When the ideal formula fails only at the edges or corners of the input box, it is usually
  because the net under-fits the extreme values (fewer training points there). Add a clip at
  the net's own extreme value, or a small scale factor, rather than giving up the formula.
  Look at where the large errors are to choose between the two: errors only at the corners mean
  saturation (clip); errors growing with magnitude everywhere mean a scale.
- The snap tolerance is 2% of the output range, so a smooth formula that is right in shape can
  still miss by a few percent at the extremes; compute the tolerance first and compare the max
  error with it.

## product/seed1 — agent_1 (m45a85d, 2026-10-31)

- Group hidden units by direction of their input weight vector. With 2 inputs, directions
  (1, 1) and (1, -1) mean the net is a sum of 1-D functions of x0 + x1 and x0 - x1.
- Evaluate each group as a 1-D curve (subtract its value at 0) and compare with simple
  shapes: s^2/4, |s|, relu, step, linear. Check symmetric points (f(v) vs f(-v)) for even
  or odd functions.
- f(x0 + x1) - f(x0 - x1) with f quadratic is a product: a tanh net with equal and
  opposite (a, a) / (a, -a) units with opposite-sign quadratic groups is multiplication.
- Many tanh units with output weights of mixed sign and biases spread over the range are a
  spline of a smooth curve; fit the curve, not the units.
- Check the net's own constant (sum of group values at 0 plus the output bias); here it
  was ~0, so no constant term.
- Largest residuals of a snapped polynomial sit at the box corners, where tanh saturates;
  that is the net's error, not the formula's.

## product/seed2 — agent_1 (m897faa, 2026-10-31)

- Evaluate the net on a coarse grid (9x9 for 2 inputs) and print it before reading any weights.
  Products, sums, abs and max show up at once in the table.
- Get the grid's axis order right (meshgrid `indexing='ij'`: row = x0, column = x1). If a formula
  fails at every corner with the sign flipped, check the orientation first.
- Hidden weight vectors close to (a, a) and (a, -a) mean the net works on x+y and x-y. Look for even
  functions of those (squares give x*y = ((x+y)^2 - (x-y)^2)/4, abs gives max/min).
- The worst error of a correct formula sits at the corners of the box, where tanh/sigmoid units
  saturate. A residual that grows only at the box edges is a training artefact, not a missing term.
- Compare the max error with the snap tolerance (2% of the output range). If a one-line formula fits
  inside it, stop; extra terms that only chase saturation make the program harder to read.

## bump/seed0 — agent_1 (mbe96b8, 2026-10-31)

- **Dead units by distance.** For relu(w·x + b), the kink is at distance −b/|w| from the origin. If it is
  farther than the box's corner radius, and on the inactive side, the unit is dead in the box: delete it.
  If it only clips a corner and its weight times activation is below the tolerance, delete it too.
- **Look at units geometrically:** direction angle, kink offset, and output weight × |w|. Here they fell
  into two clean families: lines through the origin with negative weight (the peak) and lines at a common
  radius (~2) with positive weight (the flattening skirt). That is the signature of a radial bump.
- **relu(z) = z + relu(−z)** for units that are active on most of the domain. Their linear parts merge
  into one affine term, often with a meaningful constant (here the peak height ≈ 1). The residual relus
  are local and have small activations, so rounding their weights costs less.
- **Refit the output layer in L∞ (an LP), not least squares,** after every structural change. The
  acceptance test is a max-error test, and least squares wastes a lot of the slack. Use a coarse grid
  plus cutting-plane points (add the worst points of a fine grid and re-solve). The coarse grid alone
  under-estimates the max error.
- **Snap order matters.** Snap the high-leverage units (large weight, large activation, corner kinks)
  first, while there is slack, and the small fan units last. With the opposite order a near-zero-weight
  unit used up the budget, and an important line (x1 − x0 = 2.1) could no longer be snapped.
- **Output rounding granularity has to scale with activation size.** A unit whose activation reaches 6
  turns a 0.005 rounding into 0.03 of error. Keep the budget for it, or round those weights last with
  the LP re-solving the rest.
- **Check the slack first.** Compare the best L∞ error achievable with the current structure (output
  refit only) with the tolerance. That tells you how much snapping is possible before you try. When the
  net itself is noisy and asymmetric, an idealised formula will not pass, and the useful result is a
  snapped piecewise-linear program.

## bump/seed1 — agent_1 (m49ecbf, 2026-10-31)

- Sample the net on a coarse grid and along rays from the centre first; a symmetric bump with a
  peak and decaying sides suggests exp(-r^2) or a cone, and checking the error of those formulas
  quickly tells whether a closed formula can pass at all.
- Remove units whose pre-activation is negative on the whole input box (check the box corners:
  |w0|*3 + |w1|*3 < -b means always off).
- Replace units that are positive on (almost) the whole box by their linear part; this adds a
  linear term `a*x0 + b*x1` and changes the constant.
- Snap one coefficient at a time and refit the output layer by minimax LP (min max |error|), not
  least squares: the checker's criterion is the max error, and LS fits leave much larger max
  errors. Try coarse grids first (1, 0.5, 0.25) and fall back to finer ones.
- Keep a safety margin (about 70-80% of the tolerance) on the fitting set, since the checker
  also samples points on the kinks and corners; corners of the box are where rounding errors in
  linear terms grow largest.
- Classify units of a 2-D bump by the distance of their kink line from the centre: lines through
  the centre with negative weight build the slopes, lines at the bump's radius with positive weight
  flatten the outside.
- On a busy machine, LP-based searches (seconds each) are much cheaper than retraining with torch;
  retraining with a high-p norm from a good start can also make the max error worse.

## bump/seed2 — agent_1 (m0b28f8, 2026-10-31)

- Before anything else, find dead units by interval arithmetic over the input box: a unit with
  max(w.x + b) <= 0 on the box is always 0 and can be removed exactly.
- Sample the net on a grid and print it. It shows the shape (here a bump) and whether a closed
  formula is plausible. Compare the best formula with the net under the checker's tolerance before
  spending effort on it: if the net's own ripples are bigger than the tolerance, keep a piecewise
  linear program.
- Prune by greedy backward elimination with an output-layer refit. Use a minimax (LP) refit, not
  least squares, because the checker judges the maximum error. The LP gives a lower max error for
  the same set of units.
- Normalise each ReLU unit so its largest coefficient is +-1 (relu(s*z) = s*relu(z) for s > 0).
  This makes the coefficients comparable and easy to snap.
- Snap greedily, one parameter at a time, nicest candidate first, refit the outputs after each
  snap, and accept under an error budget below the tolerance (about 70% of it), so the boundary
  points the checker adds do not push the error over.
- When a structural search gets stuck, test how far a full retrain of k units on the teacher gets;
  it gives a lower bound on the number of units needed and tells you when to stop pruning.
- On Windows with many jobs running, set torch.set_num_threads(1); with the default the small
  training loops were 100 times slower.
- Group ReLU units by where their crease lines lie (through the centre vs at a rim), and name them
  after that; it makes a sum of ramps readable.

## circle/seed0 — agent_1 (me1a33d, 2026-10-31)

- First check which hidden units are dead or always-on over the input box (max/min of the affine part
  at the box corners). Dead units go; always-on units fold into a linear term.
- When many units share an output-weight sign and similar |bias|, with weight directions spread over
  all angles, the net is building a convex region from tangent half-planes (polygon ~ disc/ellipse).
  Try a quadratic form `c - x0^2 - x1^2` (or with shifted centre / axis scales).
- Find the boundary by bisection along rays from the region's centre; constant radius -> circle,
  snap the radius to a nice number (here 2, so the program is `4 - r^2`).
- Before writing anything, print an ASCII sign map of the output on a grid: for 2-input binary nets
  the shape is usually recognisable immediately.
- Validate a candidate threshold by sweeping it and counting disagreements outside the unsure zone;
  pick the simple value in the middle of the zero-disagreement interval.

## circle/seed1 — agent_1 (m3b7414, 2026-10-31)

- First classify units by their range on the input box: always-off (drop), always-on (fold into one
  affine term), and switching units. Large bias with small weights means always-on.
- For switching units, normalise each weight vector to unit length and list the angle and the
  normalised offset. Many units with the same output sign, directions spread over 360 degrees and
  similar offsets = a convex piecewise-linear bowl: the decision region is a polygon approximating
  a disk (or ellipse if offsets vary systematically with angle).
- Estimate the radius from the grid: max radius of positive points vs min radius of negative points,
  and the area fraction vs pi*r^2/box_area; snap r^2 to an integer.
- For binary outputs return a smooth score with the right sign (`r^2 - x0^2 - x1^2`); the checker only
  needs the decision to agree outside the unsure zone.

## circle/seed2 — agent_1 (m8df0da, 2026-10-31)

- Draw the binary decision map on a coarse grid first. The shape (disk, band, half-plane, quadrant)
  often gives the formula straight away.
- If many units have negative output weights, negative biases of similar size and directions spread
  evenly around the circle, the net is a polygon approximating a round region. Measure the boundary
  radius along rays from the centre. If it is constant, use `x0*x0 + x1*x1 < r*r` (no `**` needed).
  Snap r to a simple value.
- A unit whose bias is large compared with |w|·range is always on (or always off). Fold it into the
  constant, or drop it.
- Units with tiny input weights *and* tiny output weights are dead weight. Drop them.
- For binary outputs, returning a smooth score (`4 - r2`) is as good as an if/else and shorter. Only
  the sign matters.
- When the boundary is a curve that the net approximates with straight pieces, expect only a snap pass.
  Do not chase exactness by reproducing the polygon.

