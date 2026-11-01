# Notes

## What the network is

One float input `x0` in [0.5, 4], one hidden layer of 32 ReLU units, regression output.
Sampling the dump gave f(0.5)=1.969, f(1)=1.003, f(2)=0.498, f(3)=0.329, f(4)=0.252,
which matches 1/x0 (other candidates checked: log, sqrt, exp(-x); none fit).

## What I changed

- `agent.py` is `return 1 / x0`. `check.py`: PASS (snap), max |program - net| = 0.0305 against a
  snap tolerance of 0.0345. The largest gap is at the lower edge x0 = 0.5 (net 1.969, formula 2.0):
  the net cuts the steep corner of the hyperbola with a few linear pieces. Mean error is about -0.0035.
- I also built an exact piecewise-linear version (`scratch/agent_exact.py`, PASS (exact), max error
  3e-7). How I got it from the dump:
  - 17 of 32 units are dead on the whole box (negative slope and negative bias with x0 > 0). Dropped.
  - 7 units have their kink outside [0.5, 4] and are active on the whole box (units with kinks at
    0.42, 0.498 (three of them, just under the lower bound), -0.16, -1.96, 5.15). I folded them into one line,
    -0.204*x0 + 0.407.
  - 8 kinks are left inside the box: 0.66, 0.89, 1.22 (twice: one big, one with weight 4e-4),
    1.65, 2.12, 2.81, 3.93. Each is written as `s * relu(x0 - k)` or `s * relu(k - x0)`.
    The kinks get closer together where the curve bends most (small x0), which is what a linear-piece fit
    of a convex 1/x would do.
  I kept the formula as the final version because it is the readable program the net approximates and it passes
  the checker. The exact version is the fallback if an exact match is required.

## Heuristics

- For a 1-input ReLU net, first sort units into three groups on the input box: dead (zero everywhere), always
  active (kink outside the box, so linear), and units with a kink inside. Fold the always-active ones into a single
  `a*x + b`, and write the rest as `s*relu(x - k)` / `s*relu(k - x)`. That gives an exact program with only the
  in-box kinks.
- Kinks that sit just outside a bound (here 0.498 against lo = 0.5) are always active, i.e. linear, inside the box.
  Check the sign at both bounds rather than trusting the kink position at a glance.
- Before snapping coefficients, evaluate the net at a few round inputs (0.5, 1, 2, 3, 4) and compare with
  common functions (1/x, log x, sqrt x, exp(-x), x^2). A positive, decreasing, convex output with f(1) ≈ 1 and
  f(2) ≈ 0.5 is 1/x.
- Kinks that crowd together in one part of the domain while all the in-box slope changes have the same sign point
  to a smooth convex or concave function. The crowding marks where the curvature is largest.
- The largest gap between a smooth formula and a ReLU net is usually at the edge of the domain where the
  curvature is highest (here x0 = 0.5). Look there first when a formula is close to the snap tolerance.
- Two units with almost the same kink (1.2200 and 1.2224) where one has a tiny output weight: the small one is
  noise. It can be merged or dropped when only snap is needed.
