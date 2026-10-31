# Notes

## What the net computes

A 2-input, 16-unit tanh net that regresses on [-3, 3]^2. Evaluated on a grid, it is a round bump
at the origin: 1.01 at (0, 0), 0.78 at (0.5, 0), 0.60 at (0.5, 0.5), 0.37 at (1, 0), about 0 beyond
radius 2. That matches **exp(-(x0^2 + x1^2))** closely (minimax amplitude 0.984, best width factor 1.0).

The plain formula does **not** pass the checker: max |exp(-r^2) - net| = 0.0275, against a snap tolerance
of 0.0207 (2% of the output range). The gap comes from ripples in the net's tails (about ±0.01 rms,
peaks ±0.027 near x0 ≈ 2.25 and along the x1 = 3 edge). Global smooth corrections did not close it:
- a·exp(-k r^2) + c: 0.0266;
- adding a quadratic: 0.0223;
- 7 greedy-chosen polynomial terms: 0.0201;
- Gaussian bumps or boxes on the worst spots: 0.022.

The ripple is too fine to model with a few terms.

## Mechanism (what agent.py keeps)

Flipping each unit so its output weight is positive (tanh is odd) shows the structure:
- **11 strong units** with |w| ≈ 1.2–1.5 and b/|w| ≈ +0.45. Their directions are about 360/11 ≈ 32.7 degrees
  apart (1, 34, 66, 98, 132, 164, 199, 229, 263, 296, 327 degrees). Each is a soft half-plane
  tanh(a·(d·x + 0.45)) that is on everywhere except beyond 0.45 on the far side of its direction. Only near
  the origin are all 11 on together, so their sum is a polygonal, roughly round bump (1.73 at the origin).
- **5 weak units** (|w| ≈ 0.4–0.7, b/|w| between -1.8 and -0.1) switch on only far out. They add about -0.8 at
  the origin and cancel the ridge sum in the tails, which is still 0.1–0.5 at radius 3.

## Snapping in agent.py

- One shared slope **1.4** for the 11 ridge units, in the form `tanh(1.4 * (cos·x0 + sin·x1 + offset))`.
- Directions rounded to 0.05 per component.
- Offsets rounded to 0.05: eight of the eleven became 0.45, two 0.5, one 0.4.
- Weak-unit weights and biases rounded to 0.05.
- Output weights refit by a minimax LP (on a grid plus random points) after the snapping, then rounded
  to 0.001.

Result: **PASS (snap)**, max error 0.0184 against the 0.0207 tolerance.

What did not pass:
- one offset of 0.45 for all ridge units, refit: 0.0196 on my grid, too close to the tolerance;
- uniformly spaced angles (exactly 360/11): 0.026;
- one shared output weight for the ridge units: 0.044;
- output weights rounded to 0.01: 0.025–0.03 (16 rounding errors add up).

## Heuristics

- **Evaluate on a grid first and guess the closed form.** Values like 1, e^-0.25, e^-0.5 and e^-1 at
  (0,0), (0.5,0), (0.5,0.5) and (1,0) identify exp(-r^2) at once.
- **Measure the formula against the tolerance before reaching for corrections.** If the residual is
  structured noise (zero mean, high-frequency, no global trend), a handful of smooth correction
  terms will not fix it. Then fall back to cleaning up the net's own mechanism.
- **Canonicalise tanh units by the sign of their output weight** (tanh(-z) = -tanh(z)). Then sort by
  direction angle and by b/|w|. Hidden geometry shows up: here, equally spaced half-planes at a common
  distance from the origin, plus a separate group of far-out units.
- **Radial bumps from tanh nets** are built as N half-planes around the centre (an N-gon), plus a few
  far-out units that cancel the leftover tail. Expect ripple of about the output weight divided by N.
- **Snap shared structure (slope, offset) before outputs, then refit the outputs by minimax LP**,
  not least squares. The checker's criterion is a max error, and LS fits leave avoidable peaks.
- **Round output weights finely (0.001) when there are many units.** Rounding errors add across units.
  Coarse snapping belongs in the shared hidden parameters.
- **Snapping to exact symmetry can fail** even when the structure is clearly symmetric. The net's
  per-unit output weights compensate for its irregular angles, so snap angles to rounded direction
  cosines, not to the ideal 360/N grid.
- **Check how much margin a passing candidate has.** The checker adds boundary-bisected points; a 10%
  margin under the tolerance on your own dense grid plus random points was enough here.
