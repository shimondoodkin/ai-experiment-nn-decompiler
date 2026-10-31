# Notes

## What the network computes

A single bump: about 0.97 at the origin, about 0.44 at distance 1 along the axes, about 0.12 at (±1, ±1),
and close to 0 (±0.1, noisy) further out. Inside the box it is a piecewise-linear pyramid/cone built from
two groups of ReLU lines:

- a **ridge fan** of lines through (or almost through) the origin, all with negative output weight. They
  form the concave peak;
- a **skirt** of kink lines about 2 from the origin (NE, NW, SW, two in the SE), with positive weight. They
  bend the far slopes back up to a flat ~0.

The net is noticeably asymmetric (for example 0.434 against 0.461 at (0, ∓1), and -0.08 against 0.145
on the edges), so a clean symmetric formula (a Gaussian, 1 − |x0| − |x1|, a cone) cannot stay within the
2% snap tolerance (0.023). The program keeps the ReLU structure and snaps it.

## Changes

1. Removed units 2, 4 and 12: their kink lines lie 4.3, 4.6 and 10.2 from the origin, outside the box
   (max radius 4.24), so they are never active. Exact.
2. Removed unit 13 (kink at 3.0 from the origin, active on 3% of the box, max contribution 0.02).
3. Normalised each unit so its larger input coefficient is ±1, then snapped the hidden lines greedily
   (big skirt units first, then the fan), with an L∞-optimal refit of the output layer by linear
   programming after each snap (plus cutting-plane extra points from a 481×481 grid). Snaps were kept
   only while the max error stayed ≤ 0.014.
   The skirt lines became x0 + x1 = 2.1, x1 − x0 = 2.1, 0.95·x0 + x1 = −2, x0 − 0.6·x1 = 1.7 and
   0.75·x0 − x1 = 1.8.
4. Rewrote the four mostly-active units (active on ~78% of the box) as relu(z) = z + relu(−z). Their
   linear parts and the output bias collapse into one linear `tilt = 1.03 − 0.495·x0 + 0.44·x1`. The
   constant comes out at about 1, the peak height. What is left of those units is a relu that is active
   only in a far corner (the skirt).
5. Rounded every output weight and the linear terms to multiples of 0.005, re-solving the LP for the
   remaining free weights after each rounding (largest |weight × activation| first).
   Result: max |program − net| = 0.0193, under the tolerance of 0.0229. `PASS (snap)`.

Rounding the outputs to 0.01 instead gives 0.024 (FAIL). Snapping the fan lines harder (for example
relu(0.1 − x0), relu(−x0 − 0.3·x1)) with outputs at 0.005 gives 0.022, which is too close to the tolerance.
The 12 units with only the output layer refit reach 0.0096. That leaves only about 0.013 of slack for all
the snapping, which is why some fan coefficients keep two decimals.

## Heuristics

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
