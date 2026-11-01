# Notes

## What I did

- The dump is a 2 -> 16 sigmoid -> 1 binary classifier on [-3, 3]^2. Reading 16 sigmoid units directly
  says little, so I printed the decision region on a 61x61 grid (`scratch/look.py`). It is one closed,
  convex blob: an oval, longer along x0 than along x1, and off-centre in x0.
- I bisected the decision boundary along rows (fixed x1) and columns (fixed x0) (`scratch/bnd.py`):
  - at x1 = 0 the boundary is at x0 = -1.49 and 2.49, so the centre is 0.5 and the half-width is 2.0;
  - at x0 = 0.5 the boundary is at x1 = -0.99 and 1.00, so the half-height is 1.0;
  - off-axis points fit the ellipse too: at x1 = 0.5 the formula gives 0.5 +- 2*sqrt(0.75) =
    -1.232 / 2.232, and the net gives -1.234 / 2.221 (row) and -1.223 / 2.230 (mirrored row).
- The program is `((x0 - 0.5)/2)^2 + x1^2 < 1`, written as a score `1 - u*u - x1*x1` with
  `u = (x0 - 0.5)/2`, so `score > 0` is class 1. All 16 units and the output layer collapse into this one
  quadratic. The many sigmoids (edge steps facing different directions) are just the net's way of
  building the curved boundary.
- Check: `PASS (snap)`, 0 disagreements where the net is confident. 4517 of 308370 points disagree, all
  in the unsure zone along the boundary, mostly near the bottom edge (x1 close to -1), where the net's
  boundary sits slightly inside the ellipse. `PASS (exact)` is not reachable with a clean conic, because
  the net's boundary is a wobbly approximation of the curve.

## Heuristics

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
