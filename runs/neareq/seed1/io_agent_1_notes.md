# Notes

## What the function is
An "x0 is approximately equal to x1" detector: class 1 iff |x0 - x1| < 0.2.

## How I found it
- Only 11% of samples are positive; the highest scores all lie on the line x1 ~= x0, so I
  rotated to D = x0 - x1 and S = x0 + x1.
- Positive samples span exactly D in (-0.199, 0.199) and the closest negative has |D| = 0.1994,
  so the band edge is at |D| = 0.2 on both sides (snapped to one shared threshold).
- Local linear fits of `out` in narrow D bins show the score is piecewise linear: about
  +82.8*D + 16.6 on the left, a nearly flat plateau ~8.6 in the middle, and -88.8*D + 17.7 on the
  right; the S coefficient is ~0 (0.06 / -0.01), so S is irrelevant. Zero crossings: D = -0.200
  and D = +0.1994 -> one symmetric threshold 0.2 on abs(D).
- Raw score is roughly min(left plane, plateau, right plane) (max error 0.07), i.e. a tent
  clipped at the top; I returned the 0/1 decision rather than the unsnapped slopes.

## Result
`python check.py agent.py` -> PASS (snap); 0 disagreements where the function is confident,
remaining ones only in the unsure zone along the band edges. Not exact (the raw slopes are
asymmetric, 82.8 vs 88.8, and not meaningful values).

## Heuristics
- Look at the top-scoring samples first: if they line up along a line such as x1 = x0, rotate to
  difference/sum coordinates (D = x0 - x1, S = x0 + x1) and test which one the output depends on.
- Fit a linear model in narrow bins along the suspected key variable; constant coefficients across
  bins reveal linear pieces of a ReLU net, and a ~0 coefficient tells you a variable is irrelevant.
- A band of positives around a line, with the score falling off linearly on both sides, is an
  `abs(a - b) < t` (closeness / equality) test.
- Get the threshold from the gap between the largest positive and smallest negative |D|, and from
  the zero crossings of the fitted planes; if both sides land near the same round number, use one
  shared threshold with abs().
- Prefer the 0/1 decision over the raw score when the score's slopes are arbitrary trained
  magnitudes; the check accepts the decision and it reads as the intended functionality.
