# Notes

## What the network is
1 input x0 in [0, 200] -> 8 tanh units -> 1 regression output. Output runs from about -0.45 (x0=0)
to about 60.9 (x0=200), and it rises the whole way.

## What I changed and why
- **Unit census.** I computed each unit's centre (-b/a), width (1/a) and output weight.
  - Units 0, 4 and 6 have centres outside [0, 200] (-272, -11.7, -13.9). Over the domain they are
    almost constant; units 4 and 6 are saturated at -1.
  - Units 2 and 3 have centres at -6.6 and 105 with wide transitions, so they act mostly as
    near-linear ramps.
  - Units 1, 5 and 7 (centres 68, 181, 145) shape the rest.
  - All |output weights| are about 7. This is the usual "sum of many saturating tanh pieces
    approximating a ramp" pattern. The individual units carry no meaning.
- **Read off the slope profile.** The numerical derivative is about 0.15 on [0, 40]. It rises
  smoothly to about 0.40 around x0 = 95–100, then wobbles between 0.38 and 0.44 up to 185.
  Near 200 it drops (an edge artifact of unit 5's tanh centred at 181). So the function is convex
  and piecewise-linear-like: a low marginal rate, then a higher one.
- **Fit with nice coefficients.** I ran grid searches over 2- and 3-piece continuous
  piecewise-linear fits, using slopes in 0.05/0.01 steps and knots in steps of 5/10.
  - The best unconstrained minimax 3-piece fit had error 0.67. Its parameters: intercept 0,
    knots 58 and 88, slopes 0.156, 0.295 and 0.391. The remaining error comes from the net's own
    wiggles.
  - The best 2-piece fit had error 0.77 (knot 75, slopes 0.18 and 0.39, intercept -0.75), which
    is less clean.
  - I chose slopes 0.15 / 0.25 / 0.40 with knots 50 and 90 and intercept 0, continuous at both
    knots. As a max of lines it is `max(0.15x, 0.25x - 5, 0.4x - 18.5)`.
- **Result:** `PASS (snap)`. The max error is 0.997 against a tolerance of 1.228, about 1.6% of
  the output range. The worst point is x0 ≈ 117, inside the net's slope wobble.
- **Exact is not reachable** with a simpler program. Even dropping the saturated unit 4 changes
  the output by up to 6e-4 near x0 = 0, which is more than atol 1e-5 + rtol 1e-4·|y| there. Any
  exact version would be the dump itself, so I chose the readable snapped formula.

## Heuristics
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
