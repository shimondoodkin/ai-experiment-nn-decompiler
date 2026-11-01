# Agent notes

## What the network computes

1 input on [-1, 3], 8 ReLU units, regression. I found each unit's kink at `-b/w`:

- h0 (kink -9.33) and h3 (kink -6.89) are active on the whole domain, so together they are one linear term.
- h6 (kink -2.34, active only for x < -2.34) is never active on the domain: dead, so I removed it.
- The other five kinks are at -0.217, -0.206, -0.206 (three almost on one point), 0.651 and 1.682.

The net is a decreasing, nearly convex piecewise-linear curve. The slopes after merging the three kinks near -0.21
are about -2.04, -1.36, -0.85 and -0.47. Values: f(-1)=4.895, f(0)=3.008, f(3)=0.631.
The slopes fall by a fairly constant ratio per unit of x, and f(0) is about 3 with f(-1)/f(3) about e^2.
Both point to an exponential decay: f(x) ≈ 3·exp(-x/2).

## Versions (all checked with check.py)

1. `scratch/exact.py`: dead unit removed, always-on units folded into one line. **PASS (exact)**, max err 7.7e-7.
2. `scratch/pw.py`: four-piece snapped spline
   `3 - 1.35x + 0.7·relu(-0.2-x) + 0.5·relu(x-2/3) + 0.4·relu(x-5/3)`.
   The three kinks near -0.21 are merged into one at -0.2, and the kinks are snapped to 2/3 and 5/3.
   **PASS (snap)**, max err 0.019 (the tolerance is 0.085).
3. `agent.py`: `3 * exp(-x0 / 2)`. **PASS (snap)**, max err 0.051 (60% of the tolerance). The largest
   error is at the domain ends, where the net's linear pieces cut the curve's corners.

I submitted the formula as `agent.py` because it is the most readable. If you need something closer to the net,
the spline in `scratch/pw.py` is about 3× more accurate, and `scratch/exact.py` is the exact reduction.
A minimax 2-kink spline reached only 0.077 error, so a spline needs at least three kinks to pass with margin.

## Heuristics

- For 1-input ReLU nets, list every kink `-b/w` and compare it with the domain. Units whose kink is outside the
  domain are either always on (fold them into one affine term) or never on (delete them). Check the sign of `w`
  to tell which.
- Kinks that cluster within about 0.01 of each other act as one kink. Merge them and sum their slope changes.
- Read the function from its segment slopes. If all the slope changes have one sign, the curve is monotone and
  convex (or concave), which suggests a smooth function such as exp, 1/x, sqrt or log, not a true piecewise
  function.
- Test candidate closed forms with simple endpoint arithmetic: f(0) gives the scale, and log(f(a)/f(b))/(b-a) gives
  the decay rate of an exponential. Then compare the max error with the snap tolerance (2% of the output range).
- A smooth formula fitted to a ReLU net shows its largest error at the domain ends and at the kinks, because the
  linear pieces are chords or tangents of the curve. Expect an error about as large as the spline's sagitta.
- Keep an exact reduced version and a snapped spline next to the formula, so whoever reads the results can trade
  readability for fidelity.
