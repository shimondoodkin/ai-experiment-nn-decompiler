# Notes

## What the network does

One float input x0 in [0, 4], 32 ReLU units, one regression output. In 1-D a ReLU net is a
piecewise-linear function, so I computed every unit's kink position (-b/w) inside [0, 4] and the
net's value at each kink (script in `scratch/an.py`).

- Only 13 of the 32 units have a kink inside the domain; the rest are either always off or
  always linear there (they only add to the overall slope and offset).
- The kinks cluster near 0 (0.05, four near 0.19, 0.44) and spread out further right
  (0.64, 0.77, 1.08, 1.52, 2.06, 2.55, 3.08). Every slope change is negative: the curve is
  concave, with slope falling fast near 0 and slowly later.
- Values: f(1) = 0.999, f(2) = 1.415, f(3) = 1.733, f(4) = 2.003. That is sqrt(x).

## The program

    root = sqrt(x0)
    return max(root, 0.05)

`sqrt(x0)` alone fails only near x0 = 0: the net starts at f(0) = 0.049 (a ReLU net cannot follow
sqrt's infinite slope at 0), 0.049 off against a tolerance of 0.039. Flooring at 0.05 fixes that.
Result: PASS (snap), max |program - net| = 0.019 (near x0 = 0.019, where the net's first linear
piece undershoots the root), within tolerance 0.039. Not exact: an exact version would be the
13-piece linear interpolation itself, which is the net and not a formula.

## Heuristics

- For 1-input ReLU nets, list the kinks (-b/w) inside the domain and evaluate the net at them and
  at simple points (integers). Units with kinks outside the domain collapse into one affine term.
- All slope changes with one sign means convex/concave; values at integer points then usually
  identify the function (here f(1)=1, f(4)=2 -> sqrt).
- Kinks that crowd towards one end, with spacing growing away from it, point to a function with
  steep or singular slope at that end (sqrt, log, 1/x).
- Where the true function has an infinite slope or a singularity, the net's error is largest
  there. Rather than fitting the residual, add a simple clamp (max with the net's value at the
  endpoint) so the formula stays readable and passes the check.
- Try the plain formula first with the checker: its failing inputs show where a fixture is needed.
