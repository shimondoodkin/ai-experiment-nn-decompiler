# Notes

## What the network computes

2 inputs in [-3, 3], binary output, 16 ReLU units. Plotting the decision region on a grid showed
one convex blob, an ellipse elongated along x0 and centred near x0 = 0.5. I bisected the decision
boundary along 360 rays from (0.5, 0) and fitted a general conic
(a x0^2 + b x0 x1 + c x1^2 + d x0 + e x1 + f = 0) by SVD. Normalised to c = 1 it gave
0.248 x0^2 + 0.0001 x0 x1 + x1^2 - 0.241 x0 - 0.001 x1 - 0.928 = 0. After snapping
(b = e = 0, a = 1/4, d = -1/4) it is 0.25 (x0 - 0.5)^2 + x1^2 = 0.99, and the right-hand side snaps to 1.

`agent.py`:

    u = (x0 - 0.5) / 2
    return 1 - u*u - x1*x1        # > 0 inside the ellipse

Check: PASS (snap), with 0 disagreements where the net is confident. All disagreements lie on
the boundary, in the net's unsure zone, because the net's boundary is a polygon of about 11
line pieces and the program's is a true ellipse.

## Exact variant (kept in scratch only)

`scratch/exact.py` gives PASS (exact) with 11 ReLUs. To get it:
- unit 7 is never active on the box (its bias is -0.38 and its weights are about 0.01), so I dropped it;
- units 4, 6, 13 and 15 have large positive biases and are active everywhere near the boundary,
  so I folded them into one linear term. Where they switch off, the output is far below 0, so no
  decision changes.

Every other unit is one edge of the polygon, so this variant can't be simplified further and stay
exact. I submitted the ellipse because it is the readable form.

## Heuristics

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
