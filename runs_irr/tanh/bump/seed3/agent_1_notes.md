# Notes

## What the network computes

A 2 -> 16 tanh -> 1 regression net on [-3, 3]^2. On a grid it is a round bump centred at the
origin: 1.01 at (0,0), 0.78 at radius 0.5, 0.60 at radius 0.71, 0.37 at radius 1, 0.13 at radius
1.41, and about 0 (ripple of +-0.03) beyond radius 2. That is `exp(-(x0*x0 + x1*x1))`.

## Why agent.py is not the formula

`exp(-(x0*x0 + x1*x1))` fails the check: the net ripples by up to 0.030 near the corners of the box
((3,-3) +0.030, (-3,3) -0.026, (3,1.5) +0.026, (-2.5,-3) -0.023), and the snap tolerance is 0.0207.
I tried smooth corrections fitted by minimax (scale, offset, linear tilt, quadratic and cubic
terms, other widths exp(-b r^2)) and corner hinge terms relu(+-x0 +-x1 - s). The best got to about
0.020-0.021, with meaningless tiny coefficients (0.003 * x0, ...) and no margin, so I did not use
them. The ripple is the interference of the 16 ridges and has no compact form.

## What agent.py is

The net with its structure made readable:
- Every unit is sign-normalised so its bias is negative: `a*tanh(w.x + b)` with b > 0 becomes
  `-a*tanh(-w.x - b)`. Each unit is then the "outside a half-plane" signal `tanh(k*(u.x) - c)`.
- The units fall into two groups:
  - 11 sharp walls, k ~ 1.4, c ~ 0.63 (wall at distance ~0.45 from the origin), output weight ~ -0.3,
    with directions spread every ~33 degrees (360/11). Together they form a soft 11-gon indicator.
  - 5 soft walls, k ~ 0.7, c ~ 0.4..1.16, output weight ~ +0.27, every ~72 degrees (360/5).
- Units are sorted by angle within each group and named s0..s10, w0..w4.
- All weights and output weights are rounded to 2 decimals. Result: PASS (snap), max error 0.0116
  against tolerance 0.0207.

Things that did not work: one shared (k, c, a) per group, even refit, misses by 0.063; one shared
output weight per group misses by 0.08; one shared bias -0.63 for the sharp walls with refit output
weights passes before rounding (0.012) but fails once the output weights are rounded (0.023);
rounding to 0.05 with refit output weights leaves almost no margin (0.0185 on a grid).

## Heuristics

- Print the net on a coarse grid first and compare it with candidate formulas in radius: values
  1, e^-0.25, e^-0.5, e^-1 at radii 0, 0.5, 0.71, 1 identify a Gaussian bump exp(-r^2) at once.
- Before giving up on a clean formula, map the residual net - formula on a fine grid. If it is a
  ripple of the same size everywhere outside the main feature, it is the net's own dirt, and
  smooth corrections (polynomials, tilts) will only chase it with tiny coefficients.
- Fit corrections by minimax (an LP over grid points), not least squares: the check is a max
  error, and least squares can be 30% worse in max error.
- Sign-normalise tanh units (make the bias negative, flip the input and output weights with it)
  and list them by direction angle, gain |w| and offset c/|w|. Groups with the same gain and offset
  at evenly spaced angles mean a polygon or radial construction: a bump is a sum of half-plane walls.
- A radial bump that is made of a few ridges keeps ripples of a few percent of its height in the
  far field (the corners of the box), because each ridge is a 1-D function and the ridges do not
  cancel exactly. On wide boxes that ripple can exceed a 2% tolerance, so the true formula may not
  pass even when it is clearly the intent.
- When the formula fails only on dirt, rounding each weight to 2 decimals while keeping the
  original output weights is a safe fallback: refitting output weights after rounding inner weights
  helps only when the refit weights are left unrounded.
- Check shared-parameter snaps (one gain, one bias per group) by refitting and then rounding: a
  snap that passes before rounding the refit weights can fail after.
