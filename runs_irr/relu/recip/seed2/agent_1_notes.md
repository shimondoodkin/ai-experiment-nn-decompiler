# Notes

## What the network does

One input x0 in [0.5, 4], regression output, 8 ReLU hidden units.

- Units 1 (-0.238 x0 - 0.885 > 0 only for x0 < -3.7) and 6 (> 0 only for x0 < 0.038) are never active on
  the domain: removed.
- Units 0, 2, 4, 5 are always active on the domain (kinks at -3.39, 0.495, 0.497, 10.7; the two near 0.5
  are just below the lower bound 0.5): they are plain linear terms and fold into one line.
- Units 3 and 7 have kinks inside the domain, at x0 = 1.805 and x0 = 0.9075.

So the net is a 1-D piecewise-linear function with three pieces. The slopes increase
(-2.140, -0.581, -0.130), so it is convex and equals the max of its three lines exactly:

    max(2.9666196 - 2.1398938 x0, 1.5516555 - 0.5807089 x0, 0.7388997 - 0.1304706 x0)

This passes `PASS (exact)` (max error 3e-7; kept as scratch/agent_exact.py).

## Snapping

Values: f(0.5)=1.897, f(1)=0.971, f(2)=0.478, f(4)=0.217, i.e. roughly 1/x0 (but 1/x0 or a/x0+b alone
are off by 0.07-0.1, above the 0.034 snap tolerance). Each line is close to a tangent of 1/x, and the
kinks are almost exactly a factor 2 apart (0.9075, 1.805), as 1/x is self-similar under x -> 2x, y -> y/2.
So I snapped the intercepts to 3, 1.5, 0.75 (halving) and the slopes to 2.2, 0.55, 0.1375 (quartering):

    max(3 - 2.2 x0, 1.5 - 0.55 x0, 0.75 - 0.1375 x0)

Kinks become 10/11 and 20/11. `PASS (snap)`, max error 0.024 (tolerance 0.034), worst at the first kink.
A plain decimal snap max(2.95 - 2.15 x0, 1.55 - 0.58 x0, 0.74 - 0.13 x0) also passes (0.026), and the
0.125-grid chords of 1/x (3 - 2x, 1.5 - x/2, 0.75 - x/8) do not (0.16). agent.py is the self-similar version.

## Heuristics

- For a 1-input ReLU net, compute every unit's kink -b/w and drop units that are never active on [lo, hi];
  units active everywhere are linear terms. What remains is a piecewise-linear function with one piece per
  in-domain kink.
- If the slopes of the pieces increase (convex), write it as max(line1, line2, ...); if they decrease
  (concave), min(...). That is exact and removes all relu calls.
- Read off the function from a few sample values (here f(x)*x ~ 1 suggests 1/x). If the true function does
  not pass the snap tolerance itself, keep the piecewise-linear form and snap its lines instead.
- Look for symmetry of the suspected function in the lines: for 1/x, successive lines related by
  x -> 2x, y -> y/2 (intercept /2, slope /4) gives one shared snap for all lines instead of three
  independent ones.
- Brute-force snapping on a grid around each line's coefficients, evaluating the max of lines against the
  exact piecewise form, is cheap in 1-D; keep the error margin for the kinks, where it peaks.
