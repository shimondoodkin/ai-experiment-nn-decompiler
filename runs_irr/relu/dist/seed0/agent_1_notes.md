# Notes

## What I changed

- The dump is 2 inputs -> 16 ReLU units -> 1 output. Every hidden unit has a bias near 0 (a hyperplane
  through or near the origin), and all output weights but one tiny one are positive. A sum of positively
  weighted ReLUs of lines through the origin is a convex, roughly positively homogeneous function: a cone.
- On a 13x13 grid the output was symmetric under x0 -> -x0, x1 -> -x1 and swapping the inputs, about 0 at
  the origin, 3 at (±3, 0) and (0, ±3), and 4.24 = 3*sqrt(2) at the corners. That is the Euclidean norm.
- A least-squares fit of `a*sqrt(x0^2+x1^2) + b` gave a = 1.00, b = 0.00. Fits of `|x0|+|x1|` and
  `max(|x0|,|x1|)` were much worse (max error 0.68 and 0.80).
- `agent.py`: `return sqrt(x0*x0 + x1*x1)`. Result: PASS (snap), max |program - net| = 0.078 against a snap
  tolerance of 0.083. The largest errors are at the origin (net 0.078, true 0), where 16 kinks cannot make
  a sharp tip. Elsewhere the mean error is about 0 and the max is about 0.05, so a constant offset would not help.
- Unit 11 (`0.0107*x0 + 0.2897*x1 - 0.902`) is dead on the whole box (its maximum is -0.0016), but that
  only matters for an exact rewrite. An exact version would still need the other 15 units, so I kept the formula.

## Heuristics

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
