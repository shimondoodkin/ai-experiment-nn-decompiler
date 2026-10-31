# Notes

## What the network computes

A 1 -> 32 -> 1 ReLU net, regression on x0 in [0, 4]. Of the 32 units, 17 have their kink inside
[0, 4]; the rest are always on or always off over the domain, so they only add linear terms. I
evaluated the net on a grid. It matches sqrt(x0) closely: 1.001 at 1, 1.416 at 2, 1.733 at 3,
2.004 at 4. The kinks are packed toward 0 (0.0003, 0.026, 0.079, 0.099, 0.238, 0.446, 0.725, ...)
and spread out further along, which is how a piecewise-linear fit of a concave curve with
falling curvature places them.

## The program

    root = sqrt(x0)
    return max(root, 0.05)

- Plain `sqrt(x0)` fails, but only at x0 near 0. A ReLU net has finite slope, so it can't follow
  sqrt's vertical tangent at 0. The net gives 0.048 at x0 = 0, just above the snap tolerance of
  0.039.
- A floor of 0.05 fixes this. Check: PASS (snap), max |program - net| = 0.0125 (at the net's
  second kink, x0 = 0.0257). Everywhere else the error is a few thousandths.
- `sqrt(x0 + 0.0025)` (= sqrt(x0 + 1/400)) passes too, with max error 0.0155. I kept the floor
  because its error is smaller and it states the artifact directly: the net is sqrt with a floor
  near 0.05.
- An exact pass would mean keeping the 17-knot piecewise-linear spline, which is just the dump
  over again. The snap pass with a closed form is the readable result.

## Heuristics

- For a 1-input regression net, first sort the hidden units by kink position -b/w. Drop units whose
  kink lies outside the domain (they are constant-on or constant-off, so they fold into an affine
  term). Then tabulate the net on a grid next to a few candidate elementary functions (sqrt, log1p,
  x^2, exp, 1/x, sin). A direct match at integer inputs (1, sqrt 2, sqrt 3, 2) identifies the
  function fast.
- How the kinks are spread hints at the curvature. Kinks bunched near one end of the domain mean
  the curvature is concentrated there (sqrt or log near 0, exp near the top end).
- A smooth target with unbounded slope (sqrt at 0, log at 0) gets an approximation error at that
  point, and the net usually sits above the true value there. Model it with a floor
  (`max(f(x), c)`) or a small shift inside the function (`f(x + eps)`) instead of fitting more
  pieces. Pick c from the net's value at the endpoint, rounded to a simple number.
- Check the max error per region instead of only overall. Here all of the error came from the
  first 0.03 of the domain.
