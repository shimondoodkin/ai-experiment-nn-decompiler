# Notes

## What I found
- The net is 1 input -> 32 ReLU units -> 1 output, regression on x0 in [-2, 2].
- Evaluating it on a grid gives values that match x0^2 within about 0.01 everywhere
  (f(0) = -0.0014, f(±1) ≈ 0.995..0.999, f(±2) ≈ 3.99..4.00).
- Kink analysis: every ReLU unit is a hinge at -b/w. The units with sizeable slope change
  (|w|·v ≈ 0.3–0.48) sit roughly evenly spaced over [-1.85, 1.75], with "left-opening" units
  (w < 0) for negative kinks and "right-opening" units (w > 0) for positive kinks. Equal, positive
  slope changes at evenly spaced knots = constant second derivative = a parabola. Total slope
  change ≈ 8 over the range [-2, 2], i.e. f'' ≈ 2, so the parabola is x0^2.
- Remaining units are dead in the domain (kinks outside [-2, 2] with tiny output weights, e.g.
  units 28, 18, 17, 2, 4) or tiny corrections (|slope change| < 0.04); they are noise.

## What I changed
- Replaced the whole network with `x0 * x0`.
- Check: PASS (snap), max |program - net| = 0.0109 (at x0 ≈ 1.88), tolerance 0.080.
- An exact pass would require reproducing all ~25 effective hinges with their trained
  (non-nice) coefficients, which would just be the dump again; the snapped formula is the
  meaningful program.

## Heuristics
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
