# Notes

## What the network does
2 inputs in [-3, 3], one binary output (class 1 when score > 0), 64 ReLU units, one layer.

- About 16 units have a large positive bias (~3-4) and tiny input weights. Over the whole
  box they are always active, so together they are just a constant plus a small linear term
  (the "fixture" that raises the score at the centre; the net outputs ~129 at the origin).
- About 46 units have input-weight norms ~2-4 and negative biases, each a hyperplane at
  distance ~0.7-0.95 from the origin, with negative output weights. Their normals point in
  all directions roughly evenly, so the sum is a polygonal "cone" that grows with |x|.
- Units 12 and 47 are near-dead (tiny weights or output weight ~1e-9).

The net is a piecewise-linear approximation of a radially symmetric bump. Scanning the
decision boundary along 16 rays from the origin gave a crossing radius of 1.980-1.997 in every
direction, so the class-1 region is the disc x0^2 + x1^2 < 4.

## Program
`return 4 - (x0*x0 + x1*x1)`, class 1 inside the disc of radius 2.

Check: PASS (snap), empirical over 440k inputs. All 5188 disagreements fall in the net's unsure
zone, on the thin ring between the net's polygon (radius ~1.98-2.0) and the true circle. An exact
pass would mean copying the 46-facet polygon, which hides the formula, so I kept the snapped one.

## Heuristics
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
