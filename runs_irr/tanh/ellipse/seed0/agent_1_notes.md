# Notes

## Result

`agent.py` is an inside-the-ellipse test:

```
u = (x0 - 0.5) / 2
return 1 - u*u - x1*x1        # class 1 inside the ellipse centred (0.5, 0), semi-axes 2 (x0) and 1 (x1)
```

`check.py`: PASS (snap), 0 disagreements where the net is confident, 5269 (1.88%) inside the unsure zone,
all on the ellipse rim.

## What I did

1. **Constant units.** 8 of the 16 tanh units (2, 3, 4, 5, 7, 10, 12, 14) have input weights below 0.02 and
   biases near +-3, so tanh stays at about +-0.995 over the whole box. Together they add a constant
   (about -33.3, varying by less than 0.04 over the box) to the output. They carry no shape.
2. **Boundary units.** The other 8 units are steep tanh steps (weight norm 2 to 4.2) whose zero lines sit
   0.8 to 2 from the centre at angles spread around the circle. They form a polygon of soft edges that
   cuts out a convex blob: the usual way a one-layer net approximates a closed curve.
3. **Shape.** An ASCII plot of the sign of the output over [-3, 3]^2 showed one convex blob, wider in x0
   than in x1. I bisected the decision boundary along 360 rays from inside it and fit a general conic
   (x^2, y^2, xy, x, y, 1) by SVD. It came out as 0.247 x0^2 + x1^2 - 0.005 x0 x1 - 0.252 x0 - 0.0005 x1 - 0.933,
   centre (0.511, 0.002), x0 extent [-1.49, 2.52], x1 extent [-0.99, 1.01].
4. **Snap.** xy and x1 terms -> 0; x0^2 coefficient -> 1/4; completing the square gives
   ((x0 - 0.5)/2)^2 + x1^2 < 0.995 -> 1. Centre 0.5 (a half-step), semi-axes 2 and 1.
5. The output is binary with threshold 0, so the returned score only needs the right sign; I return
   1 - (normalised radius)^2, which is positive inside.

An exact pass is not realistic here: the net's rim is a polygon of tanh steps, while the program's is an
exact ellipse. They differ by about 0.01 at a few points, all inside the unsure zone.

## Heuristics

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
