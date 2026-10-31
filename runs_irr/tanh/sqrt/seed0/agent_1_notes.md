# Notes

## What I did

- The dump is a 1 -> 8 -> 1 tanh net on x0 in [0, 4]. I sampled it on a grid instead of simplifying
  the units one by one. The output runs from 0.112 at x0 = 0 to 1.996 at x0 = 4, and matches sqrt(x0)
  to within 0.003 for x0 >= 0.2 (sqrt(1) = 1.000 vs 0.998, sqrt(2) = 1.414 vs 1.413, sqrt(4) = 2 vs 1.996).
- `sqrt(x0)` alone fails only near zero: the net cannot reproduce the infinite slope of sqrt at 0 and
  starts at 0.112 with slope about 2.2. The error there (0.112) is above the snap tolerance (0.0377).
- Fix: floor the root at a constant. `max(sqrt(x0), 0.1)` passes with max error 0.034; `max(sqrt(x0), 1/8)`
  passes with more margin (max error 0.021), so I kept 0.125.
- Final program: `max(sqrt(x0), 0.125)`, PASS (snap), max |program - net| = 0.021. All 8 units and their
  weights are gone; no unit is individually meaningful, they only jointly fit the curve.

## Heuristics

- For 1-input regression nets, plot or tabulate the output before touching the weights, and compare the
  table against a few standard functions (sqrt, log1p, x^2, exp, 1/x, sin). A smooth monotone curve with
  value f(4) = 2, f(1) = 1, f(2) = 1.414 is sqrt; checking 2-3 "landmark" inputs identifies it quickly.
- Smooth activations (tanh/sigmoid) cannot follow singular behaviour such as sqrt's vertical slope at 0,
  log at 0 or 1/x near 0. Expect the formula to fit everywhere except a small boundary region, and patch
  that region with a clamp (`max(f, c)` / `min(f, c)`) or a short linear piece, rather than distorting the
  formula (e.g. sqrt(x + c)) everywhere.
- Pick the clamp constant from the net's value at the boundary (here 0.112), then round to a simple
  fraction that leaves margin under the snap tolerance (1/8 rather than 0.1).
- The snap tolerance is 2% of the output range; read it from the first FAIL report and use it to
  budget the error of each simplification.
