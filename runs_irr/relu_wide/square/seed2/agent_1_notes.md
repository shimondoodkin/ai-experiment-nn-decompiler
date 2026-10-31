# Agent notes

## What the network is
One float input x0 in [-2, 2], 32 ReLU units, one regression output. A 1-input ReLU net is a
piecewise-linear function of x0, so I listed every unit's kink position (-b/w) and its slope
contribution (w*v), and sampled the output on a grid (script: `scratch/an.py`).

- 11 units have kinks outside [-2, 2] or tiny slope contributions (|w*v| < 0.02); they only add a
  constant or a near-zero linear term over the domain.
- The remaining kinks are spread roughly every 0.25 across [-2, 2], with slope contributions of
  about -0.5 per kink on the left (x < -0.7) and about +0.45 to +0.49 per kink on the right (x > 1.2),
  that is, the slope rises steadily from about -4 to +4: a convex, symmetric curve.
- Samples: f(0) = -0.002, f(+-0.5) = 0.25, f(+-1) = 0.99..1.00, f(+-1.5) = 2.24, f(+-2) = 3.99.
  This is x0^2, approximated by a chord spline with ~0.25-wide pieces.

## What I wrote
`agent.py`: `return x0 * x0`.

Check: `PASS (snap)`, empirical, 286k inputs, max |program - net| = 0.015 against a snap tolerance
of 0.080 (2% of the output range). The largest deviations are at the edges x0 = +-2, where the net
under-shoots (3.985 vs 4). An exact version would need every kink of the spline back (essentially
the dump), so I kept the formula.

## Heuristics
- For a 1-input ReLU net, compute each unit's kink x = -b/w and its slope change w*v (signed by
  which side the unit is active on). Sort by kink. This turns the net into a readable spline.
- Drop units whose kink is outside the input box (they are constant or linear on the domain) and
  units with |w*v| much smaller than the others.
- Evenly spaced kinks with equal-sign, roughly equal slope changes mean a smooth convex/concave
  curve; a slope rising linearly from -k to +k over a symmetric domain is a parabola. Sample
  f(0), f(+-1), f(+-2) to read off the coefficients (here f(x) = x^2).
- Check symmetry f(x) vs f(-x) on a grid: even functions suggest x^2 or abs(x), odd ones x or x^3.
- Test the simple formula with the checker before trying to fit corrections; with a 2%-of-range snap
  tolerance, a spline approximation of a smooth function usually passes as the formula itself.
