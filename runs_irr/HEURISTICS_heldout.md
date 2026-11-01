# Heuristics learned by Claude Code

Appended by `nndc agent collect`, one section per refactoring attempt.

## bump/seed3 — agent_1 (m9aa096, 2026-11-01)

- Tabulate the output on a coarse grid first. A peak of 1 at the origin with values 0.78 / 0.37 at
  radius 0.5 / 1 means exp(-r²). Compare values at a few radii with exp(-r²), 1/(1+r²) and similar
  shapes to choose the formula.
- Before trying to beat the snap tolerance with a formula, map the residual net − formula. If it is
  a scattered ripple with no structure (sign changes all over, worst at the box edges where the net
  has no training pressure), no global reparametrisation of the formula will pass. The ripple has to
  be reproduced by the units themselves.
- "Formula + refit units" can be tested quickly: minimax-fit (LP) of the residual with the existing
  hidden units as a basis. If that stays above the tolerance, keep the net's own structure and snap it.
- Use a minimax (L∞, linprog) refit for the output layer, not least squares. The check uses max
  error, and least squares leaves the worst points at the edges.
- Sigmoid output weights are strongly collinear (sums of sigmoids are nearly constant), so rounding
  them all at once ruins the fit. Round them one at a time and refit the rest after each step, starting
  with the weight closest to the grid. This allows a step about 2× coarser.
- First-layer weights tolerate coarser rounding than output weights once the output layer is refit
  (0.05–0.1 steps here, compared with 0.02 for the output).
- Greedy unit pruning with a minimax refit gives a quick answer to whether all units are needed. Here
  none could be removed.

## circle/seed3 — agent_1 (m6f3145, 2026-11-01)

- For low-dimensional binary nets, first draw the sign of the output on a grid; the shape (disc,
  band, half-plane, quadrant) usually suggests the formula before any weight analysis.
- Group hidden units by sign of the output weight and sign/size of the bias. Many units with equal
  |bias|/|w| ratio and normals spread over all directions are the facets of a polygon approximating
  a circle of radius |bias|/|w| (here about 1.8 for the facet lines; the actual boundary sits a bit
  further out where the summed penalties cross the output bias).
- Measure the boundary directly: scan rays from a point inside for the sign change, then fit a
  circle (linear least squares on 2cx + 2cy + k = x^2 + y^2). Snap centre and radius to nice values.
- Return a signed score (r^2 - x^2 - y^2) rather than a 0/1 comparison, so the program has a natural
  margin and disagreements fall only where the net itself is near zero.
- Keep scratch scans coarse: evaluating a 16-unit sigmoid net on 360 x 200k points is slow; a few
  thousand radial samples per ray already gives the radius to 1e-3.

## dist/seed3 — agent_1 (me50417, 2026-11-01)

- **Tabulate before decoding units.** For 1–2 inputs, print the net on a coarse grid first. A
  cone, bowl or ridge often names the function (here hypot) faster than reading 16 sigmoid units.
- **Test symmetries numerically.** Compare f(-x0, x1), f(x0, -x1) and f(x1, x0) with f. If they
  match to within the snap tolerance, look for radial or `abs`-based formulas.
- **Plot the radial profile.** For a candidate radial function, print net - r as min/max/mean over
  angles at fixed radii. Smoothing at kinks and edge droop show up as clear patterns.
- **Expect smoothing at kinks.** Smooth activations (sigmoid, tanh) round off kinks and cusps
  (|x| at 0, the tip of a cone, the corner of a max). Model this with a floor or ceiling via
  `max`/`min` at the net's value at the kink, rather than a smooth `sqrt(r^2 + c)`. The
  rounded part is narrow, and a flat floor matches it within tolerance.
- **Expect edge droop.** Networks often undershoot growing functions at the far corners of the
  input box, where there is little training data. A slope just under 1 (here 0.98) with a small
  offset absorbs this better than a ceiling.
- **Spend the error budget evenly.** Choose constants by minimax over a dense grid (the checker
  is a max-error test), not by least squares. Least squares left the error at the origin too large.
- **Find which zone fails.** When the clean formula fails, find which zone fails (origin vs.
  corners) and fix only that zone, keeping the formula readable.

## ellipse/seed0 — agent_1 (ma38294, 2026-11-01)

- For a low-dimensional binary net with many hidden units, plot the decision region first. If it is one
  closed convex blob, guess a conic (circle or ellipse) instead of trying to simplify unit by unit.
- Find the centre and the semi-axes by bisecting the boundary along a horizontal and a vertical line
  through the blob: the centre is the midpoint of the two crossings, and the semi-axis is half the gap.
  Snap them to simple values (0.5 steps, small integers).
- Check the guess at off-axis rows (for example x1 = +-0.5) before running the checker. If row and
  column extents match the conic both on and off the axes, the axes are aligned with the inputs and
  no x0*x1 term is needed.
- Write the class as a signed score (`1 - sum of squared normalised offsets`), so that the binary
  threshold at 0 is the conic itself and the margin grows smoothly away from it.
- A sigmoid net with many saturated units, whose weight vectors point in many directions and all have
  similar large output weights, is a typical piecewise approximation of a curved boundary. Expect
  `PASS (snap)` rather than exact: the leftover disagreements sit in a thin band along the boundary.

## ellipse/seed1 — agent_1 (mf2bcb3, 2026-11-01)

- For a 2-input binary net, first plot the decision region on a coarse grid; the shape (one convex
  blob, band, half-plane, corners) tells you the formula family before reading any weights.
- Many steep hidden units with weights of similar size and directions spread around the circle,
  all feeding the output with large weights of the same sign, are a polygonal fence around a convex
  region: look for a circle / ellipse, not for per-unit logic.
- Units with tiny input weights (|w| < 0.1) and moderate bias are constants; fold them into the bias.
- To recover a closed boundary: bisect it along rays from an interior point, fit a general conic by
  SVD, then snap coefficients (drop near-zero cross/linear terms, round the ratio of squared terms to
  an integer, read the centre and semi-axes from the boundary extents and snap them to half-steps).
- Return the signed margin (inside positive) scaled to integer coefficients; for binary output only
  the sign matters, so scale freely to make constants nice.

## hypot1/seed0 — agent_1 (m075686, 2026-11-01)

- Before reading weights, evaluate the net on a coarse grid and look for symmetries (even in each
  input, symmetric under swapping inputs). Even + swap-symmetric strongly suggests a function of
  r^2 = x0^2 + x1^2.
- For radial functions, compare the grid values against a short list of candidates: r, r^2,
  sqrt(1+r^2), exp(-r^2), 1/(1+r^2), log(1+r^2). Matching f(0), f(1,0), f(1,1), f(2,0) is usually
  enough to pick one.
- Many sigmoid units with small weights and both signs of bias are often just a smooth basis for
  one function; do not try to simplify unit by unit when a global formula fits.
- Nets trained on a box are least accurate at the corners (fewest neighbours, saturating units).
  If the ideal formula fails only at the corners, bin residuals by radius to confirm, then try a
  single-parameter correction (scale inside or outside the formula, offset, cap) fitted by minimax
  (max error, not least squares, since the checker uses the max), and snap the parameter to a
  round value that still passes.
- Prefer putting the correction factor inside the root/argument (sqrt(1 + k r^2)) when it gives a
  lower max error than an outside scale; it bends the curve at large r while leaving f(0) unchanged.

## hypot1/seed1 — agent_1 (mb69ff4, 2026-11-01)

- **Look at a coarse grid first.** For 1-2 input regression nets, print a 9x9 table of outputs.
  Symmetric bowls and values like 1, sqrt 5, 3 identify `sqrt(1 + r^2)`-type formulas at once.
- **Check the formula's max error before polishing.** If the net's own fit error is bigger than
  the snap tolerance (here 0.09 vs 0.039), no clean formula will pass. Look at the residual map:
  wavy residuals of about the tolerance size mean the net's fit noise must be kept.
- **Use a minimax LP, not least squares, for correction terms.** The check is a max-error check;
  `linprog` over [features, error] gives the best achievable max error for a feature set exactly.
- **Collapse low-slope sigmoid units to a linear term.** With |w|*range <~ 1.5 they are almost
  linear over the box; their sum is fit by `a + b*x` to about 1e-3.
- **Prune greedily with an output refit, then refit everything.** Refitting the hidden weights
  after pruning recovers much more than refitting the output layer alone (6 units: 0.032 -> 0.008).
- **Fit max error with escalating Lp losses.** `least_squares` on `|r|^(p/2)*sign(r)` with p = 2, 4,
  8 is a cheap route to a near-minimax fit of a nonlinear model.
- **Snap greedily with refit.** Snap one parameter to the coarsest grid that keeps error under
  about half the tolerance, refit the rest, freeze it, move on. Leave the linear tail and output
  biases for last; they are the most sensitive.
- **Keep about 40% margin under the tolerance.** The checker probes about 273k points, including
  corners and edges, where nets are worst; a 401x401 grid max was a reliable predictor here
  (0.0239 on the grid, 0.0239 in the checker).

## product/seed3 — agent_1 (mf8f96f, 2026-11-01)

- Before simplifying units one by one, print the net on a coarse grid (e.g. 9x9 over the
  domain). A clean table often reveals the formula directly (here a multiplication table).
- Group hidden units by weight direction. Units along (1,1) and (1,-1) with opposite-signed
  output weights point to a function of x0+x1 and x0-x1; the classic case is the product via
  ((x0+x1)^2 - (x0-x1)^2)/4. Units along a single axis point to a function of that input alone.
- Zero along both axes plus a sign pattern of + in quadrants I/III and - in II/IV is the
  signature of x0*x1 (or an odd-odd function of it).
- Saturating activations (sigmoid/tanh) under-shoot at the domain corners; expect the largest
  error of the clean formula there, and compare it to the snap tolerance (2% of output range)
  rather than trying to model the under-shoot.
- Try the cleanest formula first with the checker; only add correction terms if it fails.

## bump/seed3 — agent_1 (mc9fc65, 2026-11-01)

- Print the net on a coarse grid first and compare it with candidate formulas in radius: values
  1, e^-0.25, e^-0.5, e^-1 at radii 0, 0.5, 0.71, 1 identify a Gaussian bump exp(-r^2) at once.
- Before giving up on a clean formula, map the residual net - formula on a fine grid. If it is a
  ripple of the same size everywhere outside the main feature, it is the net's own dirt, and
  smooth corrections (polynomials, tilts) will only chase it with tiny coefficients.
- Fit corrections by minimax (an LP over grid points), not least squares: the check is a max
  error, and least squares can be 30% worse in max error.
- Sign-normalise tanh units (make the bias negative, flip the input and output weights with it)
  and list them by direction angle, gain |w| and offset c/|w|. Groups with the same gain and offset
  at evenly spaced angles mean a polygon or radial construction: a bump is a sum of half-plane walls.
- A radial bump that is made of a few ridges keeps ripples of a few percent of its height in the
  far field (the corners of the box), because each ridge is a 1-D function and the ridges do not
  cancel exactly. On wide boxes that ripple can exceed a 2% tolerance, so the true formula may not
  pass even when it is clearly the intent.
- When the formula fails only on dirt, rounding each weight to 2 decimals while keeping the
  original output weights is a safe fallback: refitting output weights after rounding inner weights
  helps only when the refit weights are left unrounded.
- Check shared-parameter snaps (one gain, one bias per group) by refitting and then rounding: a
  snap that passes before rounding the refit weights can fail after.

## circle/seed3 — agent_1 (m175e59, 2026-11-01)

- **Constant units:** if a unit's |w|·(input range) is much smaller than its |b|, it is a constant.
  Fold `w_out * act(b)` into the output bias. Bound the error with `act(b ± |w|·range)`.
- **Ring of equal-magnitude units:** units with nearly equal |w|, |b| and |w_out| whose normals cover
  every direction approximate a round shape (a disc, or a ring when the signs are mixed). Compute
  |b|/|w| for each unit to get the line distance. Check that sign(w_out) agrees with sign(b) (all
  units "on" at the centre).
- **The decision boundary is not the lines' distance.** With an output bias, the class flips after
  k lines are crossed, where k ≈ (score at centre) / (2·|w_out|). Measure the real boundary by
  scanning rays from the centre at many angles. If the radius is constant to within about 1–2%,
  snap it to a round number, here r = 2, so r² = 4.
- **Binary output:** return a signed margin (`R^2 - r^2`) instead of a 0/1 comparison. The checker
  then shows how close the program's boundary is to the net's.
- **Folding first is a safe step:** the version with constants folded can still pass exact. Keep it
  as a fallback before replacing the structure with a closed-form formula.

## dist/seed3 — agent_1 (m17d6f3, 2026-11-01)

- **Look at the output on a coarse grid first** and compare with simple shapes (distance, product,
  sum of abs). For regression nets the snap tolerance (2% of the range) is often smaller than the
  net's own error against the obvious formula. Measure that gap before spending time on a closed
  form; if it is bigger than the tolerance, keep a reduced network and put the formula in a comment.
- **Near-flat units:** if |w|·(input half-range) is small (pre-activation moves < ~0.2), the unit is
  linear over the box. Fold all such units into a constant plus a linear term (add a quadratic if
  the linear error is too big). Many of them often share one direction, which shows the net used them
  as a cheap slope or bias.
- **Prune with a refit, not by deletion:** test each unit's removal by refitting the output layer
  (and the hidden weights, if needed) and measuring the max error. A unit is necessary only if the
  refit cannot cover it.
- **Refit before snapping, then snap greedily:** a full refit of the kept units against the net
  gives headroom (here 0.014 -> 0.003). Then snap parameters one at a time, nearest-to-round first,
  re-optimising the rest after each snap, and accept a snap only while the max error stays under
  about half the tolerance. Snapping all at once on a 0.1 grid without re-optimising used most of
  the tolerance (0.067 of 0.075).
- **Solve the output layer by least squares** inside the hidden-weight optimisation (variable
  projection). It is fast and stable for these small nets.
- **Keep about half the tolerance as headroom**, because the checker also probes boundary and
  bisected points that a grid can miss.
- **Paired tanh steps on one input** (same weight, opposite-sign output coefficients, biases on
  either side of 0) are a smooth `|x|` or a bump. Name them as such.

## ellipse/seed0 — agent_1 (m58d86a, 2026-11-01)

- **Spot constant units first:** a tanh/sigmoid unit whose |w|·(input half-range) is far below |bias|
  (here |w| < 0.02, |b| ~ 3) is saturated over the whole box. Fold it into the output bias and check
  how much it varies over the box.
- **Binary 2-input net with many steep units at spread angles = a closed convex region.** Look at the
  sign map before reading weights; the unit lines are only a polygon approximating the true curve.
- **Fit a conic to bisected boundary points.** Bisect the decision boundary along rays from an inside
  point, fit [x^2, y^2, xy, x, y, 1] by SVD (smallest singular vector), normalise one quadratic
  coefficient to 1, and snap: small cross and linear terms -> 0, complete the square to get the centre and
  semi-axes, then snap those to simple values (half-steps, integers).
- **The extents of the bisected points give the centre and axes directly** (here x0 in [-1.5, 2.5],
  x1 in [-1, 1]). That is a quick check on the conic fit.
- **For binary outputs return a signed margin** such as `1 - r^2`. Only the sign matters, and a smooth
  margin makes the checker's unsure-zone rule tolerate disagreements on the rim.

## ellipse/seed1 — agent_1 (mbecdcc, 2026-11-01)

- Saturated units (|w| tiny compared to |b|, |b| ≥ 2.5 for tanh) are constants: fold them into the bias
  before looking at anything else. In this net 9 of 16 units were such dead/constant units.
- A binary net whose live units all face outward from one region (directions spread around 360°, output
  weights all with the inward-positive sign) is a convex region; with few inputs and float inputs, try
  a conic before a polygon.
- To find the curve: bisect the decision boundary along rays from the centroid of the positive region,
  then fit a general conic by the SVD null vector of [x², xy, y², x, y, 1]. Rotation (xy) and offset
  terms near 0 snap to 0.
- Snap the conic in completed-square form (centre, semi-axes) rather than its raw constant: the fitted
  constant is biased by the polygon approximation, and "nice" values live in the centre/axes, not in
  the expanded constant (here 3.75 = 2² - 0.5², not 4 or 3.85).
- Try a couple of nice candidates with the checker; a candidate failing at only a handful of confident
  points along one side points to a slightly wrong constant or centre.
- The returned score scale does not matter for binary output; write the program in the most readable
  normalised form (1 - (u² + v²)).

## hypot1/seed0 — agent_1 (m6022f2, 2026-11-01)

- Tabulate the net on a coarse grid first (9×9 is enough for 2 inputs) and look at symmetries
  (x↔y, sign flips) and values at the origin, the axes and the diagonals. Compare those values with
  sqrt of small integers, squares and similar landmarks to guess the closed form.
- Many tanh units with tiny weights (|w| ≈ 0.05) and biases near ±1 work in the near-linear part of
  tanh. Together they add a smooth low-order polynomial along one direction. They are a strong hint
  that the function is smooth (a polynomial or root), not piecewise.
- Fit polynomials of rising degree to see whether the function is low order. Here degree 2 left 0.34
  and degree 4 left 0.06, which points to a non-polynomial such as a sqrt.
- Once a formula is close, map the residual (net − formula) on a grid. Fix any structured part
  (here the corners, where the training data runs out and the net bends inwards) with a small
  relu correction in an L1 or |x·y| coordinate, and treat the rest as the net's fitting noise.
- Fit the correction by minimax (Nelder–Mead on an L^p norm with p rising to 80) rather than least
  squares, because the check is a max-error test. Then try nearby round values and keep any that
  don't raise the max error.
- If the tolerance floor comes from unstructured ripple, stop; no clean formula can beat it.

## hypot1/seed1 — agent_1 (m0839d3, 2026-11-01)

- **Measure each unit's swing over the input box.** Take |w|·(box half-widths) for every unit. Units
  with a pre-activation swing under about 0.2 are effectively affine: fold them into one linear term plus
  a constant, fitted by least squares, and check the residual curvature.
- **Refit after pruning.** When the small units are replaced, refit the tail (linear + constant)
  jointly with the remaining big units. Adding their contributions up one by one is less accurate
  (0.003 against 0.01 here).
- **Look for symmetry among the big units.** Equal lengths and directions 120° (or 90°/180°) apart
  suggest a radial or symmetric function built from ridges. Mirroring directions (±w1) lets units share
  an amplitude and a bias, and often removes the linear tilt.
- **Keep one degree of freedom per real asymmetry.** If the fully symmetric fit misses the tolerance,
  give the odd ridge its own amplitude and bias rather than giving up on the shared structure.
- **Use a minimax fit for the snapped structure.** The checker's acceptance is a max error, so for the
  last free parameters optimise the max error (Nelder–Mead works for about 5 parameters), then round. A
  least-squares fit leaves more max-error headroom unused.
- **Snap greedily with refits.** Snap one parameter at a time to the coarsest grid that stays under a
  safety margin (here about 60% of the tolerance), refitting the rest after each snap. The order matters:
  input weights (structure) first, then amplitudes and biases, then the tail.
- **Compare the guessed closed form with the net, not with the target.** A plausible formula such as
  sqrt(1 + x0² + x1²) can still be off by 3× the tolerance where the net is lopsided.

## product/seed3 — agent_1 (m5e50d0, 2026-11-01)

- Group hidden units by the direction of their weight vector, ignoring scale and sign. If the
  directions fall into a few groups such as (1,1) and (1,-1), the net is a sum of 1-D functions
  of those projections. Examine each one separately.
- If the groups are the two diagonals and the 1-D pieces look like even (U-shaped) functions,
  suspect x0*x1 = ((x0+x1)^2 - (x0-x1)^2)/4. If only one diagonal is present, suspect
  (x0 ± x1)^2. If there is no cross direction, suspect x0^2 or x1^2.
- Before snapping individual coefficients, print the net on a coarse grid of round input values.
  Products, sums and abs values are often obvious from the table, and checking one candidate
  formula is much faster than simplifying 16 units.
- The largest errors of a smooth (tanh/sigmoid) fit sit at the domain corners, where units
  saturate. Judge a candidate formula by its interior fit, and expect a corner error of a few
  percent of the range.

## bump/seed3 — agent_1 (m06ce7f, 2026-11-01)

- First look at the output on a coarse grid and guess the target function (here a Gaussian bump).
  Then check the formula. If the net's tails ripple by more than the tolerance, the formula cannot
  pass, and the job becomes simplifying the net itself.
- Normalize each ReLU unit by its largest |coefficient| (or by its norm) and push the scale into the
  output weight. After that, offsets read as kink positions and units can be compared.
- Group units by kink distance from the origin (-b/|a|). For radial functions this gives a cone at
  the origin (negative weights) and rings at a fixed radius (positive weights) that cut the cone off.
- A unit active toward the origin, relu(n·x + R), equals n·x + R - relu(-n·x - R). Inner and outer
  ring units are the same kind of kink, apart from a linear term.
- After snapping hidden weights, always refit the output layer with a minimax LP, not least squares.
  The acceptance test is a max-error bound, and least squares can fail where minimax passes
  (least-squares pruning gave 0.03, minimax 0.018).
- Snap in steps (0.01, then 0.02, 0.05, 0.1) and measure each step. Small offsets near 0 often matter
  more than they look: zeroing them can cost more than coarse snapping of the directions.
- With a max-error tolerance about 2% of the range, a 16-unit irregular net often allows no pruning.
  Spend the effort on snapping and readable grouping instead.
- In scratch folders, do not name a script after a stdlib module (`struct.py` broke the numpy import).

## circle/seed3 — agent_1 (mbe1088, 2026-11-01)

- **Normalise each ReLU unit** to (unit direction, signed distance from origin). Units with
  similar distance and directions spread evenly around the full circle, all with the same-sign
  output weight, are a polygon approximating a circle/ellipse: look for a quadratic
  boundary instead of trying to simplify the lines.
- **Ray scan for the decision boundary**: for a binary net, find the first sign change along
  rays from a central point at many angles. A constant radius means a circle `r^2 < R^2`;
  radius varying as `1/sqrt(a cos^2 + b sin^2)` means an ellipse; an offset centre shows up as a
  sinusoidal radius.
- **Dead and always-on units**: evaluate each unit's pre-activation range over the input box
  (corners suffice for an affine function). Never positive means delete it; always positive
  means drop the `relu` and fold it into the constant/linear part.
- **Big constant plateau, many negative penalty units** is the ReLU way of writing
  "inside a convex region": the output is roughly `C - sum of distances outside each edge`.
- A binary net only needs the boundary right: a smooth formula that is right outside the
  net's unsure zone beats a faithful but unreadable copy of the polygon.
- Snap the radius to the nearest simple value (here 2, so `R^2 = 4`) and re-check.

## dist/seed3 — agent_1 (m5443aa, 2026-11-01)

- Before reading individual units, evaluate the net on a coarse grid and compare it with
  candidate closed forms (|x|, x*y, x^2, hypot, min/max). A grid that looks the same along a row
  and a column and grows with distance from the centre points to a radial function.
- If a 1-hidden-layer ReLU net has its hidden weight vectors spread around the circle, biases
  near 0 and positive output weights, it is a convex piecewise-linear surface. For an output of 0
  at the origin that grows linearly, try the Euclidean norm.
- ReLU nets round off non-smooth points such as cone tips and kinks: the output there sits a bit
  above the true value. If the clean formula fails only at such a point, add a small smoothing
  term (sqrt(r^2 + eps^2), or sqrt(x^2 + eps^2) in place of |x|). Choose eps close to the net's
  value at the tip and round it to a nice number. A floor such as max(f, c) is usually worse.
- Bin the residual (net minus formula) by the natural variable, here the radius. A flat mean
  residual away from the special point shows the formula is right, and that what is left is
  facet noise, not a missing term.
- When the error is set by facet noise spread over the whole domain, stop at PASS (snap):
  `PASS (exact)` would need the piecewise-linear net itself.

## ellipse/seed0 — agent_1 (m9680fd, 2026-11-01)

- Binary ReLU net with positive output bias and all large output weights negative: class 1 is a
  convex region (sum of convex hinges below a level). Look for a polygon approximating a disc,
  ellipse or other convex shape.
- Plot the decision on a coarse grid first; the shape is usually obvious.
- To identify a closed boundary: bisect along many rays from an interior point, then least-squares
  fit a conic (x^2, xy, y^2, x, y = 1). Read off centre and semi-axes and snap them to half-steps /
  small integers. A tiny xy coefficient means the axes are aligned with the inputs.
- Find dead units by bounding each pre-activation over the input box (sum of |w|*range + bias);
  units with tiny output weights that fire only where the score is already far from threshold can
  be dropped too.
- The checker's unsure zone (|score| < 5% of p95) is wide for these nets, so a smooth formula that
  matches the boundary to ~0.01 passes snap even though the net's boundary is piecewise linear.

## ellipse/seed1 — agent_1 (ma413dd, 2026-11-01)

- For a binary net with 2 inputs, first print an ASCII map of the decision region. Its shape
  (half-plane, strip, blob, ring) tells you which formula family to try.
- For a closed convex blob, bisect the boundary along rays from a rough centre and fit a general
  conic by least squares (SVD null vector). Then snap: drop near-zero cross and linear terms, round
  the coefficient ratios to simple fractions, and complete the square to read off the centre and
  semi-axes. Snap the right-hand side to 1 after dividing through.
- Return a signed score (1 - normalised quadratic), not a 0/1 comparison. It keeps the threshold
  at 0 and the margin is meaningful.
- To prune exactly in a ReLU net, find each unit's min and max over the input box (or near the
  decision boundary). A unit that is never active can be dropped. A unit that is active over the
  whole region that matters is linear and can be folded into one affine term. For a binary
  output, folding is still exact if the region where the unit switches off is far from the
  threshold.
- A ReLU net approximating a curved boundary can't be exact with a smooth formula. Expect
  PASS (snap) at best, with disagreements only in the unsure zone along the boundary.

## hypot1/seed0 — agent_1 (meda121, 2026-11-01)

- **Find dead units first.** For a first-layer ReLU unit, evaluate w.x + b at the box corners. If the
  maximum is <= 0, the unit is dead and can be removed exactly.
- **Check the output weights' signs.** If they are all positive (on a single hidden layer), the net is
  convex. Fold each weight into its unit, relu(v*z) = v*relu(z), so the output is `const + sum(hinges)`.
  That removes a multiplier per unit.
- **Sample a coarse grid and look at the special points**: the origin, the axes and the corners. Values
  like 1, sqrt(3), sqrt(5), 3 point straight at `sqrt(1 + x0^2 + x1^2)`. Try the formula against the
  checker before anything else.
- **Compare the formula's residual spread with the tolerance.** When the formula fails, look at
  max - min of (net - formula). If it exceeds twice the snap tolerance, no affine rescaling of the
  formula can pass, and the net's own approximation dips must be kept. Go straight to a reduced
  piecewise-linear program. Do not spend time on formula variants.
- **Prune with a full refit, not output-only refit.** Linear refit of the output weights loses
  feasibility at 11-12 units. Refitting all parameters (least squares, then IRLS reweighting by |err|
  towards minimax) reaches 10 units.
- **Snap after the fit, then nudge.** Round to a grid (0.01, else 0.005). Then do coordinate descent of
  +-1 grid step on the max error, measured on a dense grid plus random points. Plain rounding of 30+
  coefficients adds about 0.015 of error; nudging recovers most of it.
- **Supply an analytic Jacobian.** A numeric Jacobian on a dense grid made the refit loop too slow.
  The ReLU Jacobian is just the mask times (x0, x1, 1).

## hypot1/seed1 — agent_1 (m5f731d, 2026-11-01)

- Get each first-layer unit's range over the input box from the box corners (it is affine). If it is never
  positive, delete the unit. If it is never negative, fold it into a linear term.
- When an output weight is positive, fold it into the relu: `c*relu(z) = relu(c*z)`. Then units can be
  compared by their scaled direction and offset alone.
- Look for antiparallel unit pairs (w, b) and (-w, b'), with b about b':
  - `relu(z+b) + relu(-z+b)` = `b + max(b, |z|)` when b > 0;
  - it is `relu(|z| - |b|)` when b < 0.
- Units with bias about 0 are folds through the origin. With `relu(z) = (z+|z|)/2` their linear parts
  often cancel the leftover affine term, which shows the function is even.
- If every output weight is positive, the net is convex. Check for a radial or norm-like target such as
  `sqrt(1 + r^2)` or `hypot`: compare values at (0,0), (1,0), (1,1) and (2,2) with sqrt(1), sqrt(2), sqrt(3)
  and sqrt(9).
- A smooth formula can fail the check where the net's own piecewise-linear error is biggest, mostly at kinks
  such as a cone at the origin. Find that by mapping net minus formula on a grid. If the error passes the
  tolerance, keep the cleaned-up piecewise-linear form and give the formula in a comment.
- Before snapping, run greedy pruning with a least-squares refit of the output layer to see whether any unit
  is redundant. Then try rounding grids from coarse to fine, and pick the coarsest one that stays well inside
  the tolerance (here 0.01 gives 0.024 against a tolerance of 0.042, and 0.02 already gets close to it).
- Finish with the minimax constant (the midpoint of the max and min residual), not the least-squares one.

## product/seed3 — agent_1 (m4362a9, 2026-11-01)

- Before reading individual units, fit a low-degree polynomial (1, x, y, x², xy, y²) by least
  squares on random samples. A coefficient near 1 with the rest near 0 names the function at
  once (here x0·x1).
- When hidden-unit directions cluster around (1, 1) and (1, -1), suspect a product built from
  squares of the sum and the difference: xy = ((x+y)² - (x-y)²)/4.
- A smooth formula on a ReLU net can fail only at the corners of the box. Print the residual
  net - formula on a grid to find where. Corner errors usually come from the net's
  piecewise-linear approximation running straight past its last knot.
- Model a "straight past the last knot" effect as the curve minus a squared hinge. For u², the
  tangent continuation beyond |u| = a is u² - relu(|u| - a)². A clip or a max/min with a tangent
  line is wrong here, because a tangent line lies on one side of the curve everywhere, so max or
  min picks it far from the knot too.
- Check each corner on its own. Only some corners may be flattened, so a one-sided hinge
  (relu(s - a) instead of relu(|s| - a)) can be enough.
- When several knot values give the same max error, the limit is somewhere else, usually
  interior ripple. Keep the nicest value.

## bump/seed3 — agent_1 (m1f0f74, 2026-11-01)

- Sample the net on a coarse grid first and compare it with standard shapes (exp(-r^2), 1/(1+r^2), cones).
  A peak of about 1 at the origin with exp(-0.25) at r = 0.5 identifies a unit Gaussian quickly.
- Before snapping, compute the tolerance (2% of the output range) and the net's own deviation from the
  best smooth formula. If the net's facet noise is larger than the tolerance, no clean formula will pass.
  Then the useful target is formula + a small set of the net's own units, not a formula alone.
- Within-bin spread check: bin the outputs by the candidate invariant (here r). If the spread inside a bin
  is more than twice the tolerance, no function of that invariant alone can pass.
- Write ReLU units as ridge hinges (direction angle, signed offset, weight times norm). Sorting by
  offset shows the construction: rings of hinges at offsets -c, 0 and +c with signs +, -, + are
  tents `relu(c - |n.x|)`, and many directions of them make a radial bump.
- Dead-unit test for a box domain: the max of `w.x + b` over the box is `sum(|w_i| * half-width) + w.center
  + b`. If it is negative, drop the unit. This removal is exact.
- For pruning with a max-error criterion, use a linear program, not least squares: a minimax fit for the
  weights, and L1-weighted sparsity (reweighted L1) for choosing units. Put the smooth formula in as a
  free (zero-cost) column so that it absorbs what it can. Use exchange iterations: refit with the worst
  points of a large sample added. Greedy forward selection by least squares did badly here (24 units still
  at 0.072 without the formula).
- One backward-elimination step with a full LP refit per candidate is too slow at 60 units. Reweighted
  L1 gets a similar count in about 12 LPs.
- Round first-layer weights (2 decimals) before the final output-weight refit, not after. Then round the
  output weights only as far as the error budget allows, and re-check after rounding.
- Validate with the checker under several seeds before stopping, so that the result is not tuned to one
  point set.

## circle/seed3 — agent_1 (m9f0761, 2026-11-01)

- Split units into "always on inside the box" (small weights, large positive bias; check
  `|w|*range < bias`) and "edge" units. The always-on ones only add a constant (plus a small linear
  tilt); their sum is the score's base level.
- Many units with the same sign of output weight, similar bias/|w| ratio and directions spread
  evenly over 360 degrees mean a radial function: a disc / circle (or a cone `r`). Compute the
  offset |bias|/|w| per unit to estimate the radius.
- For a binary 2-input net, do a radial scan of sign changes from the centre (or a grid scan) to
  read the boundary shape directly before analysing the weights. A near-constant crossing radius
  means `r² < R²`; snap R to the nearest integer / simple fraction and centre to the origin.
- Write the score as `R² - (x0² + x1²)` rather than `sqrt`: it has the same sign, is simpler, and
  avoids sqrt.
- A ReLU net can only make polygonal boundaries, so a curved true boundary will only ever give
  PASS (snap); don't chase exact by adding corners.

## dist/seed3 — agent_1 (m81fa7b, 2026-11-01)

- Sort units by |v_i| * ||w_i|| (output weight times input weight norm). The units with tiny products
  are dead and can be dropped together.
- Many ReLU units with bias ~ 0, output weights of the same sign and directions spread around the full
  circle (or sphere) mean a convex, homogeneous function of degree 1: a norm. Check the L2 norm
  `sqrt(sum x_i^2)` first. If the directions cluster on the axes or diagonals, check L1 (`abs` sums)
  or Linf (`max(abs(..))`) instead.
- Before analysing units one by one, evaluate the net on a coarse grid and print it. Symmetric
  patterns (equal values at (±a, ±b) and at (a, b) vs (b, a)) and familiar numbers (1.414, 2.236,
  3.162, 4.243) point straight at the formula.
- A small constant offset at a kink (here the net gives 0.03 at the origin) is approximation error of
  the net. Don't add it to the formula. The snap tolerance absorbs it.

## ellipse/seed0 — agent_1 (m081e56, 2026-11-01)

- For a binary net on 2 inputs, first draw a coarse sign map of the whole domain; the shape (half-plane, band,
  blob, ring) tells you which formula family to try before reading any weights.
- A ReLU layer where nearly all output weights are negative and biases negative, plus a few units with large
  positive bias that are always active, means "constant minus a sum of walls": a convex region, typically a
  polygon approximating a circle or ellipse.
- To recover a curved boundary, bisect the boundary along rays from an interior point and fit a general conic
  (`A x^2 + B xy + C y^2 + D x + E y = 1`) by least squares. Normalise by the largest quadratic coefficient,
  snap small cross/linear terms to 0, snap ratios to simple fractions, then complete the square to read off a
  centre and radii and snap those too.
- For binary outputs only the sign of the score matters: scale the inequality so all coefficients are integers
  (e.g. multiply `(dx/2)^2 + y^2 < 1` by 4).
- Disagreements that all lie in the unsure zone along the boundary mean the formula is right and the net is a
  piecewise-linear approximation of it; don't try to fit the net's facets.

## ellipse/seed1 — agent_1 (m6db502, 2026-11-01)

- For a binary net on 2 inputs, first print an ASCII map of the decision over the box. The shape class
  (half-plane, strip, blob, ring) is usually visible at once.
- For a closed blob, bisect the decision boundary along many rays from a point inside it, then fit a
  general conic by least squares (normalise the x^2 coefficient to 1). A small cross term and near-integer
  ratios mean an axis-aligned ellipse or circle. Complete the square to read off the centre and the
  semi-axes, then snap them to simple values (halves, small integers).
- Cross-check the snapped conic against the boundary's min/max extent along each axis: centre =
  (min+max)/2, semi-axis = (max-min)/2.
- Many units in a wide ReLU net that draws a curved boundary are just facets of the polygon. Do not try to
  simplify them one by one. Replace the whole net with the smooth formula, and expect PASS (snap), not
  exact.
- Return a signed score (1 - quadratic form) instead of a 0/1 comparison. The decision `raw > 0` is the
  same, and near the boundary the checker sees small magnitudes, which fall in the unsure zone.

## hypot1/seed0 — agent_1 (m2f30af, 2026-11-01)

- Check the signs of the output weights first. If they are all positive (or all negative) in a single
  hidden ReLU layer, the net is convex (or concave). Then look for a smooth convex formula
  (norms, x^2, exp, sqrt(c + r^2)) rather than a piecewise rule.
- Before reading 64 units, evaluate the dump on a small grid (9x9 over the box) and look for
  symmetries: x0 <-> x1, x -> -x, and radial dependence. A radial bowl points to a function of
  x0^2 + x1^2.
- Read off values at simple points (origin, axis points at integers, the corner) and compare them with
  square roots of integers, squares and exponentials. Values like 1, 1.414, 2.236 and 3 at r^2 = 0, 1,
  4, 8 identify sqrt(1 + r^2) directly.
- Units whose weights are all below about 1e-2 and whose output weight is tiny are dead. Ignore them
  when reading the structure.
- A many-unit ReLU net that approximates a smooth function will never pass exact against the
  formula. Measure the residual (mean, max, where it is largest); if it is well inside the snap
  tolerance and has no structure beyond a small bias and the apex error, the formula is the answer.

## hypot1/seed1 — agent_1 (m221943, 2026-11-01)

- One-layer ReLU net with all (significant) output weights positive and hidden directions spread
  over the circle: the function is convex. Try radial/norm-like formulas first.
- Tabulate the net on a coarse grid before reading the weights. Symmetry (x -> -x, x0 <-> x1)
  and values at the origin, edge midpoints and corners usually identify the formula quickly
  (here 1, sqrt 5, 3 at radius² 0, 4, 8 gives sqrt(1 + r²)).
- Fit `a*f(x) + b` for a handful of candidate features (r, r², sqrt(1 + r²), L1 norm,
  exp(-r²)) by least squares, and compare max errors. A fit with a ≈ 1, b ≈ 0 means the
  formula is right as written; snap it.
- Units with all weights below about 1e-4 are dead or negligible; drop them before any analysis.
- A ReLU net approximating a smooth curved surface has its largest error at the corners of the
  input box, where the training data is sparsest. Expect snap rather than exact there, and judge
  the formula by the fit coefficients, not by those corner errors.

## product/seed3 — agent_1 (m534213, 2026-11-01)

- Before reading 64 units one by one, tabulate the net on a coarse grid (integer and half-integer
  points). Symmetries (f(-x0, x1) = -f(x0, x1), zero along both axes) identify products and similar
  formulas at once.
- Many first-layer units sharing two directions, (1, 1) and (1, -1), with biases spread over the
  range, are a ReLU spline of a function of x0 + x1 and of x0 - x1. Opposite signs of output weights
  on the two families plus convexity suggest the polarisation identity x*y = ((x+y)^2 - (x-y)^2)/4.
- Units whose weights and output weight are ~1e-3 or below are dead; delete them first.
- For a smooth target fitted by ReLUs, the error is largest at the box corners and edges (fewest kinks
  there); expect PASS (snap), not exact, and check that the max error sits well below the snap
  tolerance rather than trying to fit the kinks.
- Try the clean formula before any fitted scale factor: here 1.0 * x0 * x1 already passed with
  a 2.7x margin.

## decay/seed0 — agent_1 (m1b94d2, 2026-11-01)

- For a 1-input regression net, sample the output on a dense grid first and look at the
  shape (monotone? convex? sign of slopes and output weights) before working on units.
- If all hidden units slope the same way and all output weights have the same sign, the
  output is monotone and usually a smooth elementary function; skip per-unit simplification
  and fit the whole curve instead.
- Test for an exponential by taking ratios y(x)/y(0) at equally spaced x: a constant ratio
  per step (here e^(-1/2)) means a*exp(b*x). Check that y(x+d)/y(x) is constant across the
  range and that an added constant c fits to about 0.
- Fit a*exp(b*x)+c with three free parameters, then snap a, b, c to nice values (integers,
  halves) and confirm with the checker; the snap tolerance is 2% of the output range, so
  compare the max error against that.
- Errors of a clean formula against a sigmoid-sum net are largest at the domain edges, where
  the net's basis runs out; that is expected and not a reason to add correction terms.

## decay/seed1 — agent_1 (m3cd548, 2026-11-01)

- For a 1-input sigmoid net whose units all have the same slope sign and whose output weights all
  have the same sign, the output is monotone. Tabulate it at integer inputs before analysing units.
- Check ratios f(k+1)/f(k) at equal steps. A constant ratio means an exponential a*exp(c*x), with
  c = log(ratio). Check differences too: constant differences mean a line, constant second
  differences a quadratic.
- Fit the candidate family by least squares with an extra constant term. If the fitted constant is
  about 0 and the parameters land near simple values (3, -1/2), snap them, and check that the
  snapped max error is about the same as the unsnapped fit's. If it is, the leftover error is the
  net's own approximation error and there is nothing more to recover.
- The largest disagreement of a smooth net with its formula is usually at the edges of the input
  box, where training data is sparse and the sigmoids saturate.

## exp/seed3 — agent_1 (m3a9a6b, 2026-11-01)

- For 1-input regression nets, tabulate the output on a coarse grid first and check the
  landmark values: f(0), f(1), f(-1), f(+-edge). Values like 1, e=2.718, 1/e=0.368, e^2=7.39
  identify exp immediately; similarly check x^2, |x|, sqrt, log, 1/x, sin.
- Check ratios between consecutive grid points: a constant ratio f(x+h)/f(x) means an
  exponential (here ~1.105 per 0.1 = e^0.1).
- Sums of many sigmoid units with mixed signs and similar slopes are usually a smooth fit of
  one elementary function; do not try to simplify unit by unit, fit the whole curve.
- The net's error against the true function is largest at the domain edges (sigmoid fits
  flatten out there), so expect the snap residual to peak at the boundary, not in the middle.
- If the clean formula is within the snap tolerance everywhere, prefer it to any
  "corrected" formula with fitted coefficients.

## log/seed3 — agent_1 (me2aa3c, 2026-11-01)

- For 1-input regression nets, tabulate the net on a grid first and compare it with standard
  functions (log, exp, sqrt, 1/x, x^2). Check values at "landmark" inputs (net(1) ~ 0 points to
  log; net(0) ~ 1 to exp; net(1) ~ 1 to powers).
- Sigmoid networks fitting a function with a steep end (log near 0, 1/x near 0, sqrt near 0)
  usually underfit that end: the net is flatter than the true function there. Write the clean
  formula, then patch only the failing end with a clamp (`max`/`min` on the output, or on the
  input inside the function), with a round constant.
- Look at where the max error is and how it compares with the snap tolerance; fix the region
  that breaks it, not the whole curve.
- Prefer a round clamp constant (-2.2) over a tuned one when both pass; it reads better and
  says the same thing.

## logshift/seed0 — agent_1 (m11dc4c, 2026-11-01)

- For a 1-input regression net, sample it on ~20 points first and read off landmark values:
  the zero crossing, the value at 0, the value at the domain ends. Match them against
  standard functions (log, sqrt, exp, 1/x, x^2) with a simple shift or scale.
  A zero crossing at an integer x = a together with f(a+1) ~ 0.693 points to log(x - a + 1).
- Concave, increasing and unbounded-looking over the range, with slope decreasing like 1/x:
  try log or sqrt. Check the slope ratio between ends: for log(x+c) it equals the inverse ratio of x+c.
- Many sigmoid units with mixed signs and similar slopes are usually a smooth-curve fit,
  not separate features. Don't simplify unit by unit; fit the whole curve with one formula.
- Look at the residual profile, not only its max. A small oscillating residual with mean ~0
  everywhere except one domain edge means the formula is right and the net is the one that
  is wrong near a singularity or steep part (here log near x = -2).
- The largest disagreement of a smooth formula usually sits at the domain edge closest to a
  singularity; compare that error with the snap tolerance (2% of the output range) early.

## logshift/seed1 — agent_1 (mc0ff72, 2026-11-01)

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

## recip/seed3 — agent_1 (md5e550, 2026-11-01)

- For 1-input regression nets, first tabulate the net on about 15 points and compare it with a
  short list of simple functions: 1/x, sqrt, log, exp(-x), x^2, x*(c-x), and |x-c|. A smooth
  sigmoid or tanh layer with a linear readout often fits one of these, and the unit-by-unit
  structure is just a basis expansion that carries no meaning.
- Check the domain bounds against the candidates. A domain that starts at a positive value such
  as 0.5 points to functions that are singular or undefined at 0: 1/x, log x, sqrt x.
- Look at the signs of the readout weights. If most units fall with x and the readout weights are
  positive, the function is decreasing. Small units of the opposite sign usually only correct
  curvature.
- Expect the largest error at the steep end of the domain (here the small-x end of 1/x). The net
  underfits there, so judge the candidate by its error in the interior, and use the snap
  tolerance (2% of the output range) to see whether the edge gap is acceptable.
- Prefer the exact, simple formula when it passes snap. Fitting extra affine corrections
  (a/x + b) to chase the network's edge errors adds coefficients without adding meaning.

## sqrt/seed3 — agent_1 (m903b00, 2026-11-01)

- For a 1-input regression net, tabulate it at integer and simple inputs first, and compare against
  candidate functions (sqrt, log, log1p, exp, x^2, 1/x). Values at perfect squares (1, 4) that land
  on integers (1, 2) point to sqrt.
- Find where the candidate formula fails (|error| > snap tolerance). If the failures sit in a small
  region where the true function has a singularity or an infinite slope (sqrt at 0, log at 0, 1/x
  near 0), the net is smoothing it. Patch only that region with the simplest guard: a constant floor
  or ceiling (`max(f, c)` / `min(f, c)`). Pick c from the net's value at the edge and snap it to a nice value.
- Try several nice values for the patch constant and keep the one with the smallest max error. Do
  not use a guard line `a + b*x`: it can cross the formula a second time and break the fit far away.
- Sigmoid units with tiny output weights (here |w| ≈ 0.001) or a near-linear range over the
  domain only add curvature. Look at the whole net's curve, not unit by unit, when the net
  approximates a smooth elementary function.

## square/seed3 — agent_1 (m49f0f9, 2026-11-01)

- For a 1-input regression net, first tabulate the output on a fine grid over the input box and compare
  it with candidate formulas (x^2, |x|, exp, log, sqrt, 1/x, 2^x) before reading the weights at all.
- Check symmetry first: if f(-x) is close to f(x), look at even functions (x^2, |x|, cosh-like), and use
  ratios such as f(2x)/f(x) (4 means x^2, 2 means |x|) to tell them apart.
- Sigmoid nets that fit a smooth convex curve split the work: units with slopes of one sign build each
  branch, and a unit with a negative output weight shapes the bottom. Don't try to simplify the units one
  by one. Fit the whole curve.
- Expect the largest error at the box edges, where the sigmoid units saturate and the net falls short of
  the true curve (here 3.964 vs 4 at x=2). An edge error like this does not mean the formula is wrong.
- Compare the max error with the snap tolerance (2% of the output range). If the clean formula fits with
  margin, prefer it to a fitted polynomial with odd coefficients.

## decay/seed0 — agent_1 (m8edbbb, 2026-11-01)

- For 1-input regression nets, sample the dump on a grid first and read the shape before looking at weights.
  Units of a smooth-activation net are usually a basis, not features.
- Test for elementary functions with simple transforms. A constant ratio f(x+1)/f(x) means an exponential.
  Then fit log(f) linearly and snap the scale and rate to simple numbers. A constant difference means a line.
  A constant log-log slope means a power.
- Look at the values at integer inputs (f(0), f(1), ...). They often show the constants directly
  (f(0) = 3 here).
- Expect the net to be least accurate at the edges of the input box. The max error of the clean formula
  is usually there, so judge a snapped formula by the residual in the interior.

## decay/seed1 — agent_1 (ma16615, 2026-11-01)

- For 1-input regression nets, sample the net on a grid first and look at the shape (monotone?
  convex? sign?) before touching individual units; the sum of many small-slope tanh/sigmoid units is
  usually just a smooth textbook function.
- If the output is positive and convex-decreasing (or increasing), fit a line to log(y): a straight
  line means a*exp(b*x); snap a and b (here 3.0004 -> 3, -0.4997 -> -1/2).
- Try other linearising transforms the same way: log(y) vs log(x) for powers, y vs 1/x for
  reciprocals, y^2 vs x for square roots. A near-perfect straight line with nice coefficients is the
  formula.
- Tanh units with |slope| * input-range well below 1 are nearly linear over the box; many such units
  are a sign that the net is approximating a smooth nonlinear function, not a piecewise one.
- Largest residuals of a snapped formula tend to sit at the domain edges, where the net extrapolates
  worst; compare max error to the snap tolerance (2% of output range) rather than chasing exactness.

## exp/seed3 — agent_1 (m9c7f48, 2026-11-01)

- For a 1-input regression net, tabulate it on a fine grid before reading the weights. Values at
  x = 0, 1, -1 and the sign of the curvature quickly identify common functions (exp: 1, e, 1/e;
  2^x: 1, 2, 1/2; x^2: 0, 1, 1; log, sqrt, 1/x by their domain behaviour).
- Compare the table against a few candidate closed forms side by side. When one matches to within
  about 1% of the range everywhere, try it directly in the checker before fitting anything.
- Small tanh/sigmoid nets fitting a smooth function are usually a basis expansion. Nearly flat,
  saturated units (|slope| about 0.1, |bias| > 1) add up to a constant plus a slight tilt. Don't
  try to give them meaning one by one; identify the overall function instead.
- Expect the largest error at the edges of the input box, especially where the target is steepest
  (here the right end of exp). Check that error against the snap tolerance before deciding the
  closed form is wrong.
- If the formula is the net's intended function, keep it unscaled (no fitted 0.99·exp(x)). A clean
  formula within snap tolerance is better than a refit that only chases the net's training error.

## log/seed3 — agent_1 (m1fe7e0, 2026-11-01)

- For a 1-input regression net, sample the output on a grid first and put it next to standard
  functions: log, sqrt, 1/x, exp(+-x), x^2. Clues that can identify the function quickly:
  - net(1) ~ 0 and net(2) ~ 0.693 point to log;
  - net(4) ~ 2 points to sqrt.
- Steep functions (log near 0, 1/x near 0) are where a small net bends away from the true
  formula, at the steep end of the domain. If the clean formula fails only there, patch only the
  edge instead of dropping the formula:
  - a clamp `max(f, c)` / `min(f, c)` with a round c near the net's edge value;
  - or a small argument shift `f(x + c)`.

  Pick the patch by max error first (it must be within the snap tolerance), then by RMS error,
  and prefer the patch that leaves the rest of the domain untouched.
- In a 1D tanh net, units with large |slope| (here -4.1, -3.1, -2.0) and centres near the low
  edge are what bend the curve there. Units with small slopes (|w| < 0.35) together make up the
  slow log-like trend.

## logshift/seed0 — agent_1 (m6c958a, 2026-11-01)

- For 1-input nets, tabulate first. A table of about 20 points usually shows the function family before any unit-level work.
- Find anchor points: zeros, the values at integers, and values that match known constants (ln 2 = 0.693, ln 6 = 1.792,
  sqrt 2, pi, e). A zero at x = -c+1 plus f(-c+2) = ln 2 identifies log(x + c).
- Read the shape: concave, rising and slowing down suggests log or sqrt; convex and growing suggests exp or a square.
  Compare the ratios of successive differences to tell log from sqrt.
- The net's error is largest at the domain edges, most of all where the target curves most (near log's singularity).
  Judge a candidate by its interior fit, and expect a snap pass rather than exact.
- Try the clean closed form before fitting extra scale or offset coefficients. If it passes snap, the network's small
  bias (about 1e-3 here) is training noise, not structure.

## logshift/seed1 — agent_1 (m3e0368, 2026-11-01)

- For a 1-input regression net, sample about 20 points first and compare the values with known
  functions before touching the weights. Check where the output is zero (here x0 = -1, so the
  shift is +1 from log's root) and where it equals ln 2, ln 3 and so on, or 1, 2, 3 for sqrt and
  squares.
- Concave, increasing and slowing down like a log means `log(x + c)`; find c from the zero crossing
  (`log(x + c) = 0` at x = 1 - c).
- Expect the biggest error at the domain edge nearest a singularity of the true function (log near
  its pole, sqrt near 0). Smooth tanh/sigmoid sums underfit there. That error usually stays inside
  the snap tolerance, so it doesn't mean the formula is wrong.
- A closed-form formula that passes snap beats a tanh sum that passes exact, for readability. Only
  chase exact when the net is piecewise linear or very small.

## recip/seed3 — agent_1 (m8b1bfc, 2026-11-01)

- For 1-input regression nets, tabulate the output on a grid and compare side by side with
  candidate elementary functions (1/x, log x, sqrt x, exp(-x), x^2); reading off f(1), f(2),
  f(4) often identifies the function immediately (here f(2) ~ f(1)/2, f(4) ~ f(1)/4).
- Compute the residual net - candidate across the domain. If it is small everywhere except at
  one edge, the formula is right and the net is just failing to follow a steep or curved region
  near the domain boundary (saturating tanh/sigmoid units cannot follow 1/x's blow-up).
- Fix edge droop with a clamp (`min`/`max` against a round constant near the net's edge value)
  rather than by distorting the whole formula with fitted coefficients.
- Sum-of-tanh nets fitting a smooth target show a small oscillating residual (ripple) of
  roughly constant size; do not try to reproduce it, it is fit error, not structure.
- Pick the clamp value from the tolerance budget: the error peaks where the clamp starts
  (residual of the net at that point), so choose the constant where the net's residual is
  comfortably inside the tolerance.

## sqrt/seed3 — agent_1 (m0a1318, 2026-11-01)

- For a 1-input regression net, tabulate the output on a grid and compare it against a short
  list of elementary functions (sqrt, log, log1p, exp, x^2, 1/x, |x|) before reading weights.
  Matching values at integer inputs (1, sqrt 2, sqrt 3, 2) identify the function fast.
- Smooth nets cannot follow singular slopes (sqrt at 0, log at 0, 1/x near 0). Expect the
  largest error at the domain edge where the target's derivative blows up, and patch only that
  edge instead of refitting the whole function.
- A good edge patch is the tangent line of the target at a simple point (here x = 1/16): it is
  continuous in value and slope, so the error stays small on both sides of the switch point.
  For sqrt, the tangent at x = a is sqrt(a)/2 + x/(2 sqrt(a)); a = 1/16 gives 1/8 + 2x.
- A shifted argument, f(x + c) with a simple c, is a one-term alternative for an edge offset;
  scan c over simple fractions and keep the one with the most tolerance margin.
- Do not take a max/min of a concave target with a line to patch an edge: the line can cross
  the curve again far from the edge. Use an explicit `if x < k` switch.
- Check margin, not only pass/fail: prefer the candidate whose max error is well below the
  snap tolerance, because the checker samples empirically.

## square/seed3 — agent_1 (mf5ac9b, 2026-11-01)

- For a 1-input regression net, sample the output at integer and half-integer inputs first.
  Values close to 0, 1, 4 at 0, ±1, ±2 (or other recognisable sequences) point to a simple
  formula; try that formula directly with the checker before trying to simplify units.
- Check symmetry: f(-x) ≈ f(x) suggests an even function (x², |x|, cosh-like); f(-x) ≈ -f(x) an odd one.
- Tanh units with small slopes (|w| ≲ 0.15 over a domain of width 4) are almost linear in the
  domain; pairs with near-equal weights and opposite-sign readout cancel or merge into a line.
  Their sum is mostly a linear/quadratic correction, not a feature.
- The worst error of a snapped formula is usually at the domain edges, where the net extrapolates
  poorly; compare the margin there against the snap tolerance before accepting.
- When the snap-tolerance margin is comfortable (here 0.059 vs 0.079), prefer the clean formula
  over a refit with fitted constants.

## decay/seed0 — agent_1 (mba2792, 2026-11-01)

- For a 1-input ReLU net, compute each unit's kink t = -b/w first. Units whose kink is outside the box
  are either dead (box on the off side: delete) or always on (box on the on side: fold into the
  linear part). Only units with kinks inside the box shape the function.
- Rewrite each live unit as `|c*w| * relu(t - x)` or `relu(x - t)` so its weight and its kink are
  separated; then the coefficients are directly "slope change at the kink", which are easy to snap.
- If all live units point the same way and their output weights have the same sign, the function is
  monotone and convex/concave: describe it by its segment slopes, which often turn out to be a clean
  sequence (here -0.2, -0.6, -1.2, -2.0).
- Snap kinks and slope changes first, then pick the constant as the minimax offset
  (-(max err + min err)/2) for the snapped shape rather than rounding the original constant; this
  recovers most of the error budget.
- Fit with least squares on a dense grid for each candidate kink set before rounding; it shows quickly
  which kink snap is affordable (here moving -0.089 to 0 cost about 0.04 of 0.083 tolerance).
- The error of a moved kink grows linearly with distance from the kink toward the box edge, so check
  the box corners: they are where a snapped program fails first.

## decay/seed1 — agent_1 (m343f8c, 2026-11-01)

- For 1-input ReLU nets, list every kink `-b/w` and compare it with the domain. Units whose kink is outside the
  domain are either always on (fold them into one affine term) or never on (delete them). Check the sign of `w`
  to tell which.
- Kinks that cluster within about 0.01 of each other act as one kink. Merge them and sum their slope changes.
- Read the function from its segment slopes. If all the slope changes have one sign, the curve is monotone and
  convex (or concave), which suggests a smooth function such as exp, 1/x, sqrt or log, not a true piecewise
  function.
- Test candidate closed forms with simple endpoint arithmetic: f(0) gives the scale, and log(f(a)/f(b))/(b-a) gives
  the decay rate of an exponential. Then compare the max error with the snap tolerance (2% of the output range).
- A smooth formula fitted to a ReLU net shows its largest error at the domain ends and at the kinks, because the
  linear pieces are chords or tangents of the curve. Expect an error about as large as the spline's sagitta.
- Keep an exact reduced version and a snapped spline next to the formula, so whoever reads the results can trade
  readability for fidelity.

## exp/seed3 — agent_1 (m427914, 2026-11-01)

- For 1-input ReLU nets, compute every unit's kink `-b/w` first. Units whose kink lies outside the
  input box are either always on (fold into a linear term) or always off (delete). Often only a few
  units are real.
- Merge near-duplicate units (same kink, here units 0 and 5) before anything else; they often cancel
  (here weights -1.70 and -1.83 on nearly the same unit, both folded into the line).
- Rewrite decreasing hinges `relu(b - a*x)` as `a*relu(x - b/a) - (a*x - b)` so all hinges point the
  same way; a "line + sum of c_i*relu(x - k_i)" form directly gives slopes per segment and convexity
  (all c_i > 0 means convex).
- Check exact first with unrounded folded coefficients; then try a snapped version to see how much
  slack the snap tolerance (2% of output range) gives. Errors from snapping a kink grow with the
  slope after the kink, so the endpoint of the steepest segment is where snapped versions fail first.
- Convex increasing piecewise-linear curves with geometrically growing slopes are likely fits of
  exp-like targets; worth naming in the notes but not substituting, because the check is against the net.

## log/seed3 — agent_1 (mbc48c3, 2026-11-01)

- For 1-input ReLU nets, compute each unit's kink `-b/w` and the side it is active on. Units whose active side
  lies wholly outside the input box are dead, so drop them. Units active over the whole box are linear,
  so fold them into the affine part.
- Kinks closer together than a few percent of the domain can be merged into one kink with little error.
- Compute slopes between consecutive kinks. If they only decrease, the function is concave: write it as `min`
  of lines. If they only increase, it is convex: use `max`. That is much more readable than a ReLU sum and easy to snap.
- Compare the curve with familiar functions (log, sqrt, 1/x, x^2, exp) on a grid. A concave curve whose slope
  roughly halves each time x doubles points to log. The exact formula may still fail where the net is a coarse
  interpolant (near the domain edge, or at a big kink), so keep the piecewise form when it is the faithful one.
- Snap each line to nice slopes and intercepts, prefer forms that hit meaningful points (here (1,0)), and then
  check the max error against the snap tolerance (2% of output range).

## logshift/seed0 — agent_1 (m64e8fc, 2026-11-01)

- For a 1-input ReLU net, compute every unit's kink x = -b/w first. Units whose kink lies outside the
  input range are either always off (drop them) or always on (fold them into one line).
- Kinks closer together than about 1% of the input range can be merged into one; refit the lines
  after merging.
- If the slopes decrease left to right, write the curve as min(line1, line2, ...); if they increase,
  as max(...). This is shorter and easier to read than a sum of relu terms.
- Fit the lines by least squares on a dense grid, then round coefficients and check the max error.
  Slope rounding costs error proportional to the distance from the kink, so the end of the domain is
  where a too-round slope fails; round slopes less than intercepts.
- A smooth-looking curve (here like a log) is tempting, but check the formula against the net, not
  the guess; the net's ends often differ from the true function by more than the snap tolerance.

## logshift/seed1 — agent_1 (m6896bc, 2026-11-01)

- For a 1-input ReLU net, compute kinks -b/w and the segment slopes first; the whole net is that
  table. Units whose kink lies outside the input box are just linear (or zero) terms.
- Kinks that sit within ~1% of the domain width of each other are one kink split across units;
  they create tiny segments with meaningless slopes. Ignore them when reading the shape.
- Match the slope sequence against derivatives of candidate functions: slopes ~ 1/(x + c) means
  log(x + c); slopes ~ 1/sqrt(x + c) means sqrt; slopes ~ ratio-constant means exp. Pin the shift
  c by where the net crosses 0 (log) or where the slope blows up (the left edge of the domain is
  often a little above the singularity, e.g. domain starting at -1.5 for log(x + 2)).
- Check the formula against the net's values at nice points (0, ±1, integers): net(0) = 0.687
  ≈ ln 2 and net(-1) ≈ 0 confirmed log(x0 + 2) here.
- A PL net of a concave function lies on chords, so the largest deviation from the true curve is
  at the domain edge where curvature is highest; expect the snap error there.
- When exactness is wanted for a 1-D net, an if/elif chain over the kinks with
  value-at-knot + slope * (x - knot) per segment reproduces it to float precision and is easier to
  read than the sum of ReLUs.

## recip/seed3 — agent_1 (m447665, 2026-11-01)

- **Dead-unit test on a box:** for 1-D (or per-input) affine units, compute the knot -b/w and the
  sign of w. If the active side does not meet the input interval, drop the unit. With w ~ 0, check
  only the sign of the bias. Here half the hidden layer was dead.
- **Fold `c*relu(w*x+b)` into `(c*|w|) * relu(±(x - k))`.** This leaves one coefficient per hinge,
  with the knot shown directly. It makes the piecewise-linear structure (knots, slope changes)
  readable and exact.
- **A knot just inside the box edge makes a unit almost linear.** It can be swapped for a linear
  term within snap tolerance, but the swap breaks exactness. Keep it when aiming for exact.
- **Read the segment slopes.** The slope changes at each knot give the convexity and monotonicity
  at a glance, and they suggest the target family (here decreasing and convex, 1/x-like).
- **To decide whether snapping is safe, use a minimax LP over a dense grid on candidate nice knot
  sets** before snapping coefficients. Least squares underestimates how few knots will do, because
  its max error is larger than the minimax optimum.
- **Snap the constant last,** at the midpoint of the residual range, and round it to a grid
  (0.05 here). The constant is usually what sets the final max error.
- **Exact rounding:** folded coefficients survive rounding to about 5-6 significant digits within
  atol 1e-5 / rtol 1e-4. Check this rather than assuming it.

## sqrt/seed3 — agent_1 (m65268a, 2026-11-01)

- **Dead and always-on units:** for each unit, compute its kink -b/w and check the sign of the
  pre-activation over the input box.
  - A unit that is negative over the whole box is dead: drop it.
  - A unit whose kink is outside the box and which is positive inside it is linear: drop the relu
    and fold it into one line plus constant.
- **Near-coincident kinks** (gap far below the input range) with opposite on-sides make a tiny ramp.
  Merge them into one kink and drop the sliver segment.
- **Segment slopes:** compute the slope of each linear piece between the sorted kinks.
  - If the slopes decrease monotonically, the function is concave: write `min(line1, line2, ...)`.
  - If they increase, it is convex: write `max(...)`.
  - This removes all the relus and is easy to read.
- **Snapping a min/max of lines:** snap the slopes first, then refit each intercept at a
  representative x inside its own segment (for example the segment's middle or an endpoint). Snapping
  intercepts on their own moves the kinks.
- **Fewer lines:** try dropping one line from the min/max to see if fewer pieces still pass. Here they
  did not.
- **Named shape vs the net:** a recognisable target shape (sqrt, log) may not pass if the net departs
  from it near a singular point (infinite slope at 0). Check the endpoints against the candidate
  closed form before trusting it.

## square/seed3 — agent_1 (mf97a29, 2026-11-01)

- For a 1-input ReLU net, list each unit's kink `-b/w`; units whose kink is outside the input box and that are
  negative across the box are dead, and those positive across it are linear and fold into the constant/slope.
- Two kinks closer than ~1% of the domain, with a near-zero slope between them, are one kink: merge them.
- Compute the slope of every segment. If slopes increase monotonically the function is convex, and it can be
  written exactly as `max(line_1, ..., line_k)` with no ReLUs (decreasing slopes: `min`).
- Equal slope steps between equally spaced kinks mean a parabola (constant second derivative). Slope step /
  kink spacing ~ 2a gives the coefficient (here 1.29 / 0.64 ~ 2, so x^2).
- A fitted PWL of a smooth curve sits below it at the knots and above it between them (for convex curves).
  When the plain formula just misses the snap tolerance, add the constant that centres the error band,
  `(min err + max err) / 2`, rather than changing the formula.
- Always sanity-check the output range first: the snap tolerance is 2% of it, which decides whether a smooth
  formula can replace a coarse PWL fit at all.

## decay/seed0 — agent_1 (mc30bc2, 2026-11-01)

- For a 1-input ReLU net, compute the kink positions -b/w and drop the units whose kinks are outside the input
  box. Then tabulate the slope on each piece. Many kinks with slopes that change steadily and monotonically mean
  a smooth function approximated by a spline: look for a closed formula rather than snapping breakpoints.
- Find the curve family from ratios and differences at evenly spaced points. A constant ratio f(x+1)/f(x) means
  exponential (a*exp(k*x)). Constant differences mean linear, constant second differences mean quadratic, and a
  linear log-log plot means a power law.
- Fit the family's parameters with least squares (for an exponential, a linear fit of log f), then snap them to
  simple values (3, -1/2) and check the snapped formula against the net. The 2% snap tolerance is usually wide
  enough to absorb the spline's own error.
- The largest disagreement between a spline net and its smooth target is usually at the edges of the input box,
  where the net has the fewest kinks and the least training data. Check that the edge errors fit inside the
  tolerance.
- Units with tiny output weights (|v| < 0.01) or kinks outside the box hardly matter. Once a formula fits, there is
  no need to account for them one by one.

## decay/seed1 — agent_1 (m46119d, 2026-11-01)

- For a 1-input ReLU net, list the kinks (`-b/w`) inside the input box, then the slope and value on
  each piece. Units whose kink is outside the box are constants or plain lines on the domain and
  fold into the bias and slope.
- Slope that changes in one direction (convex or concave) with many small kinks means a smooth
  curve, not a real piecewise rule. Try the standard smooth families before snapping kinks:
  exp, log, sqrt, 1/x, x^2.
- To test for an exponential, look at ratios of values at evenly spaced points (f(x+1)/f(x)).
  If they are constant, the rate is log(ratio). Snap the rate to a simple fraction (here -1/2)
  and the scale to f(0) (here 3).
- Compare the smooth formula's max error with the snap tolerance (2% of the output range) before
  running the checker. A piecewise-linear net with ~20 kinks fits a smooth curve to well within 1%.
- Many tiny units (|w| ~ 1e-3 or less, or very small output weights) are training noise. Ignore
  them when you look for the shape.

## exp/seed3 — agent_1 (m3f17a9, 2026-11-01)

- For a 1-input regression net, tabulate f on a coarse grid first and compare it with standard
  functions at landmark points (f(0), f(1), f(-1), the endpoints). Values like 1, 2.718 and 7.389
  identify exp at once. Look the same way for x^2, |x|, sqrt, log and sin.
- List the kinks `-b/w` with their slope changes `w*v`, sorted by position. If every slope change
  has the same sign and they are spread evenly over the domain, the net is a convex or concave
  smooth function approximated by a spline. Look for a closed-form smooth function, not a few
  linear pieces.
- A slope that grows in proportion to the value (f' ≈ f) means exp. f' ≈ 2x means a square, and
  f' ≈ 1/x means log.
- Check the closed form directly. The snap tolerance (2% of the output range) usually covers the
  net's own approximation error, and the largest error tends to sit at the domain ends, where the
  network was trained least densely or where curvature is largest.
- Units whose weights are about 1e-9, or whose kink lies outside the input box, contribute nothing
  and are the first to drop.
- Near-coincident kinks (within about 0.01) are the signature of a spline fit. Merging them keeps
  a snap pass, but rules out exact equality.

## log/seed3 — agent_1 (m1dd514, 2026-11-01)

- For a 1-input ReLU net, first classify each hidden unit on the input box: never active (drop),
  always active (fold into a linear term), or a knot inside the box (keep). Here half the units
  were dead.
- Sample the function on a fine grid and look at the values at recognisable points (x = 1, 2,
  the bounds). net(1) ~ 0 together with a concave increasing shape and knots crowded near the
  small end of a positive domain strongly suggests `log(x)`; compare `log`, `sqrt`, `1/x`, `x^2`,
  `exp` candidates by their max error against the net.
- Knot density follows curvature: knots crowd where |f''| is large. For log that is near the low
  bound; for exp near the high bound; for 1/x near 0.
- Compare the candidate's max error to the snap tolerance (2% of the output range) before
  bothering with an exact piecewise-linear rewrite; the largest errors of a PWL fit to a smooth
  function sit at the domain edges, where the fit has the fewest constraints.
- Positive-domain lower bounds like 0.1 (instead of 0) are a hint that the target is undefined
  or singular at 0 (log, 1/x).

## logshift/seed0 — agent_1 (mbe0277, 2026-11-01)

- For 1-input ReLU nets, compute each hinge -b/w and drop units whose hinge is outside the input
  box. They are constant-sign there, so they are linear (or zero) and fold into the affine part.
- Drop units with |w_in * w_out| times the domain width far below the tolerance.
- Tabulate the slope on each segment between hinges. Monotone, decreasing slopes mean a concave
  curve. Before fitting splines, test elementary candidates: log(x+c), sqrt(x+c), 1/(x+c), exp.
- Find the shift from the derivative: if slope*(x+c) stays roughly constant across segments, the
  function is k*log(x+c). Then check that the values match k*log(x+c) at a few round points
  (x = 0, 1, the domain ends).
- A domain lower bound like -1.5 with an argument shift of +2 keeps log's argument positive
  (0.5..6). Domain bounds often hint at the shift of a log or sqrt.
- The biggest error between net and formula is where curvature is highest (here near the left
  edge). Check there first when judging whether a closed form fits.

## logshift/seed1 — agent_1 (ma923d4, 2026-11-01)

- For a 1-input ReLU net, compute each unit's kink -b/w and slope change w*v, drop kinks outside the
  input box (they are linear there) and units with negligible slope change, then read the piecewise
  slopes. The slope sequence often identifies the function: constant -> linear, slopes ~ 1/(x+c) -> log,
  slopes growing linearly -> square, slopes proportional to the value -> exp.
- Test a candidate closed form at a few "nice" points (where the formula gives 0, log 2, integers, ...)
  before running the checker; a zero crossing at an integer input (here f(-1) ~ 0) pins the shift.
- The net's error against a smooth target is largest at the edges of the input box, especially where the
  target is steepest; expect the snap residual to concentrate there, not in the middle.
- When many units are near-zero (weights ~1e-3 or 1e-9), the net is over-parameterised; the effective
  number of kinks is much smaller than the width, and a smooth formula is a better summary than a
  pruned piecewise-linear program.

## recip/seed3 — agent_1 (mc7d170, 2026-11-01)

- For a 1-input ReLU net, first sort units into three groups on the input box: dead (zero everywhere), always
  active (kink outside the box, so linear), and units with a kink inside. Fold the always-active ones into a single
  `a*x + b`, and write the rest as `s*relu(x - k)` / `s*relu(k - x)`. That gives an exact program with only the
  in-box kinks.
- Kinks that sit just outside a bound (here 0.498 against lo = 0.5) are always active, i.e. linear, inside the box.
  Check the sign at both bounds rather than trusting the kink position at a glance.
- Before snapping coefficients, evaluate the net at a few round inputs (0.5, 1, 2, 3, 4) and compare with
  common functions (1/x, log x, sqrt x, exp(-x), x^2). A positive, decreasing, convex output with f(1) ≈ 1 and
  f(2) ≈ 0.5 is 1/x.
- Kinks that crowd together in one part of the domain while all the in-box slope changes have the same sign point
  to a smooth convex or concave function. The crowding marks where the curvature is largest.
- The largest gap between a smooth formula and a ReLU net is usually at the edge of the domain where the
  curvature is highest (here x0 = 0.5). Look there first when a formula is close to the snap tolerance.
- Two units with almost the same kink (1.2200 and 1.2224) where one has a tiny output weight: the small one is
  noise. It can be merged or dropped when only snap is needed.

## sqrt/seed3 — agent_1 (mf39e70, 2026-11-01)

- For a 1-input regression net, list the kinks (-b/w inside the box) and evaluate the net at
  integer inputs. Values like 1, 1.414, 1.732, 2 identify sqrt; check the other common shapes too
  (x^2, 1/x, log, exp) against those points before fitting anything.
- When a clean formula fails only at a domain edge where it has an infinite or very large slope
  (sqrt at 0, log at 0, 1/x near 0), the net has smoothed that singularity. Add a clamp
  (`max(f, c)` with c about the net's value at the edge) or a small input shift (`f(x + eps)`),
  and pick whichever has the smaller max error.
- Look at where the remaining max error sits. If it is in the middle of the range, it is the net's
  own approximation error, and tuning the edge constant further will not help, so round the
  constant to a simple value.
- Many hidden units in a wide 1-D ReLU net are dead (kink outside the box, always negative) or
  always on (they add up to one linear term). Only the kinks inside the box carry shape.

## square/seed3 — agent_1 (m3a0e54, 2026-11-01)

- For 1-input ReLU nets, compute each unit's kink position -b/w and its slope change |w|·v
  (sign of w tells which side it is active on). Sort by kink position: this is the whole function
  as a piecewise-linear spline.
- Equal-sized, same-sign slope changes at evenly spaced kinks mean a constant second derivative:
  a quadratic. Second derivative ≈ (sum of slope changes) / (domain width). Then fix the linear
  and constant terms from f(0) and the symmetry / f(±hi).
- Before reading weights, just tabulate the net on a coarse grid and compare against simple
  candidates (x, x^2, |x|, x^3, exp, sqrt, ...); a 41-point table often identifies the function
  immediately.
- Units whose kink lies outside the input box are either constant-zero (inactive side) or pure
  linear terms (active side) and can be folded into the affine part; units with
  |w·v| tiny compared to the others are fitting noise.
- The snap tolerance for regression is 2% of the output range, so a smooth formula that matches
  the net within ~0.5% of the range passes comfortably; prefer the clean formula over chasing
  PASS (exact) by reproducing every hinge.

