# Agent notes

## What the network does

Binary classifier on two floats in [-3, 3]. The dump is a 2 -> 16 -> 1 ReLU net whose output
bias is positive (+10.15) and whose 12 large units all have negative output weights. So class 1 is
the region where none of the hinge planes has risen far enough: a convex polygon. An ASCII plot of
the decision showed an elongated oval around (0.5, 0).

Bisecting the decision boundary along 720 rays from (0.5, 0) and fitting an axis-aligned conic
gave centre (0.507, 0.003) and semi-axes 2.0006 and 1.0019 (the xy term was -0.012, negligible).
Snapped: centre (0.5, 0), semi-axes 2 and 1.

## The program

    u = (x0 - 0.5) / 2
    v = x1
    return 1 - u*u - v*v        # > 0 inside the ellipse ((x0-0.5)/2)^2 + x1^2 < 1

`check.py`: PASS (snap), 301015 inputs, 0 disagreements where the net is confident; all
disagreements lie in the unsure zone, right on the boundary (the net's 12-sided polygon versus the
true ellipse it approximates).

## Removed

- Units 2, 4 and 12 are dead on the whole box (their pre-activation maximum over the box is < 0).
- Unit 3 can fire only near x1 = 3, with output weight 0.034 (contributes <= 0.008), where the
  score is deeply negative anyway.
- The remaining 12 hinge units are the polygon's sides; they are replaced as a whole by the conic.

## Heuristics

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
