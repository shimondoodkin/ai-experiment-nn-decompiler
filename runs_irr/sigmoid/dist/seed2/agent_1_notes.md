# Notes

## What the network computes

The 16 sigmoid units combine into a cone: the output is the Euclidean distance of (x0, x1) from
the origin. On a 13x13 grid the net gives about 3.0 at (+-3, 0) and (0, +-3), about 4.2 at the
corners (3*sqrt(2) = 4.24), and about 1 at unit distance. The only clear departure from
`sqrt(x0^2 + x1^2)` is at the tip: the net is 0.36 at the origin, because sigmoids cannot make a
sharp kink. That is the same as a smoothed norm `sqrt(r^2 + c)` with sqrt(c) about 0.35.

## What I changed

- Replaced the 16 units with one formula:
  `1.0084 * sqrt(x0*x0 + x1*x1 + 0.12) - 0.0624 + 0.0071 * x1`.
- `PASS (snap)`: max error 0.0749 against a tolerance of 0.0779.
- The residual (net minus formula) is a wobble of about +-0.08 with no simple structure: bumps
  along the edges and corners and a ring near the origin. These are the net's fitting errors.
  No closed form with a few parameters gets below about 0.074 in minimax error.
- The tolerance is only a little above that, so the snap margin is thin. Simpler versions fail:
  - `sqrt(r^2 + 0.1) - 0.04`: 0.0855;
  - with the gain 1.01, constant -0.07 and the x1 tilt rounded to 0.0075 or 0.008, the error is
    0.076 to 0.078, right at the tolerance.

  So I kept the LP-optimal coefficients for c = 0.12, to 2–3 significant digits. The gain 1.0084
  and the tilt 0.0071*x1 are small learned biases, not part of the intended function. Ideally the
  program would be `sqrt(x0*x0 + x1*x1)`.
- How I fitted it: minimax (Chebyshev) fit by linear programming over the linear coefficients
  (gain, constant, x1 tilt) on a 201x201 grid, scanning the smoothing constant c. I checked on a
  601x601 grid and then with `check.py`. I also tried an anisotropic norm
  `sqrt(a x0^2 + b x1^2 + c)` with a linear term. It reached 0.0735 with a about b about 1.02,
  which was no real gain, so I kept the isotropic form.

## Heuristics

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
