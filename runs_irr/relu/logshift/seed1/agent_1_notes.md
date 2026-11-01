# Notes

## What the net is

One input x0 in [-1.5, 4], 8 ReLU units, regression output. With a single input the net is a
piecewise-linear function of x0, so I listed each unit's kink (x = -b/w) and the slope of every
segment:

| segment of x0        | slope |
|----------------------|-------|
| -1.5 .. -1.115       | 1.47  |
| -1.115 .. -0.497     | 0.85  |
| -0.497 .. -0.489     | 0.16 / 0.33 (two tiny pieces, three kinks within 0.008) |
| -0.489 .. 0.326      | 0.53  |
| 0.326 .. 1.190       | 0.35  |
| 1.190 .. 2.242       | 0.28  |
| 2.242 .. 4           | 0.20  |

Unit 2 kinks at x = 27.2, outside the domain, so it is a plain linear term. Units 0, 1 and 3 kink
at nearly the same point (-0.4894, -0.4971, -0.4892): units 0 and 3 are a split copy of one kink.

The slopes fall monotonically, roughly like 1/(x0 + 2) (at the segment midpoints:
1/(−1.31+2)=1.45, 1/(−0.81+2)=0.84, 1/(−0.08+2)=0.52, 1/(0.76+2)=0.36, 1/(1.72+2)=0.27,
1/(3.12+2)=0.20), and the value at 0 is 0.687 ≈ ln 2, at -1 it is 0.0002 ≈ ln 1. So the net
approximates `log(x0 + 2)`.

## Program

`agent.py` is `log(x0 + 2)`. `check.py`: PASS (snap), max |program - net| = 0.029 (at the left
edge x0 = -1.5, where the net's first chord sits above the log curve), snap tolerance 0.049.

An exact alternative (PASS exact, max error 4e-7) is the 8-segment if/elif chain in
`scratch/pl.py`: each segment written as value-at-left-knot + slope * (x0 - knot). It cannot be
snapped (merging the three kinks near -0.49 already moves the output by ~2e-3, far above the
exact tolerance), so I kept the formula as the main version: it is one node instead of ~40 and it
names what the net computes.

## Heuristics

- For a 1-input ReLU net, compute kinks -b/w and the segment slopes first; the whole net is that
  table. Units whose kink lies outside the input box are just linear (or zero) terms.
- Kinks that sit within ~1% of the domain width of each other are one kink split across units;
  they create tiny segments with meaningless slopes. Ignore them when reading the shape.
- Match the slope sequence against derivatives of candidate functions: slopes ~ 1/(x + c) means
  log(x + c); slopes ~ 1/sqrt(x + c) means sqrt; slopes ~ ratio-constant means exp. Pin the shift
  c by where the net crosses 0 (log) or where the slope blows up (the left edge of the domain is
  often a little above the singularity, e.g. domain starting at -1.5 for log(x + 2)).
- Check the formula against the net's values at nice points (0, ±1, integers): net(0) = 0.687
  ≈ ln 2 and net(-1) ≈ 0 confirmed log(x0 + 2) here.
- A PL net of a concave function lies on chords, so the largest deviation from the true curve is
  at the domain edge where curvature is highest; expect the snap error there.
- When exactness is wanted for a 1-D net, an if/elif chain over the kinks with
  value-at-knot + slope * (x - knot) per segment reproduces it to float precision and is easier to
  read than the sum of ReLUs.
