# Notes

## What the network computes

A table of the net on a 0.5 grid over [-3, 3]^2 showed one round bump, peak about 1.0 at the
origin, falling off the same way in every direction. Spot values matched `exp(-(x0^2 + x1^2))`:
exp(-0.25) = 0.78, exp(-1) = 0.37, exp(-2) = 0.135, exp(-2.25) = 0.105.
So the network is a 64-unit ReLU (piecewise-linear) approximation of the unit Gaussian
`exp(-(x0^2 + x1^2))`.

## Why the formula alone is not enough

- `exp(-(x0*x0 + x1*x1))` on its own FAILs: max |program - net| = 0.0297, tolerance 0.0209.
- A minimax fit of a general Gaussian (amplitude, separate widths, a cross term, centre, offset)
  still leaves 0.0267. The rest is the net's own piecewise-linear fitting error. It is
  high-frequency noise of about ±0.03, spread over about 30 separate small spots. It is not a
  systematic shape that a better smooth formula could absorb.
- Other routes I tried and dropped:
  - Pruning hidden units and refitting the output layer. Least squares needed about 37 units;
    a minimax (LP) refit about 36. That gives no formula, just a smaller opaque net.
  - The Gaussian plus a refitted subset of the net's own units. With 20 units the error was
    still 0.0208, so it was worse than local bumps.

## Final program (`agent.py`, PASS snap, max error 0.0203 vs tolerance 0.0209)

- `gauss = exp(-r2)` with `r2 = x0^2 + x1^2`. This is the real content of the net.
- `fix`: 30 small pyramid bumps `h * relu(1 - (|x0-a| + |x1-b|) / w)`:
  - |h| ≤ 0.01; centres at the error spots, on a 0.02 grid; widths 0.3 to 1.
  - They copy the net's approximation artefacts and carry no meaning. Dropping `fix` gives the
    true function, with error ≤ 0.03 against the net.
- How the bumps were chosen:
  - A dictionary of pyramids, centred on the local extrema of the residual net − gauss, in
    several widths.
  - A sparse LP: minimise Σ|h| subject to |residual − Σ h·φ| ≤ 0.0195 on a grid plus random
    points. Then reweighted L1, which went from 36 to 32 nonzero bumps.
  - A final minimax re-solve of the heights, then rounding to 0.001.
- A tighter count does not hold. With target 0.0200 the LP chose 29 bumps, but they overfit the
  sample points, and after rounding the program FAILs (0.0218).

## Heuristics

- Tabulate the net on a coarse grid first. A radially symmetric peak of height 1 that falls to
  0.37 at r = 1 is `exp(-r^2)`. Check it on 3–4 points (r^2 = 0.25, 1, 2, 2.25).
- When the clean formula fails only by a small margin, measure the best achievable smooth fit
  (a minimax fit with free amplitude, width, centre and offset). If even that fails, the excess
  is the net's own approximation noise. Stop searching for a better formula; patch the noise.
- For 2-input PWL nets, patch noise with local pyramid bumps
  `h * relu(1 - (|x0-a| + |x1-b|)/w)`, picked by a sparse (L1) LP over a dictionary of centres
  (the residual's local extrema) and widths. Reweighted L1 removes a few more. Keep the formula
  and the patch visibly separate, so a reader can drop the patch.
- Solve the LP against a margin below the tolerance (about 93% of it). Re-check on a different
  point sample before rounding: the LP overfits the points it saw, and coefficient rounding adds
  error where bumps overlap.
- Pruning units with an output-layer refit is the fallback when no formula exists. Here it gave
  an opaque net about half the size, which was worse for readability than formula + patch.
- Units with all weights about 1e-5 or smaller are dead (here h1_10, h1_36, h1_37, h1_50).
  Remove them first.
