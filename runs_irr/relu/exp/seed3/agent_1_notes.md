# Notes

## What the network is
1 input (x0 in [-2, 2]) -> 8 ReLU units -> 1 regression output.

## What I changed
- Computed each hidden unit's kink position `-b/w` and compared it with the input box [-2, 2]:
  - units 0, 1, 3, 5 have kinks at 2.0027, -4.695, -4.958, 2.0026: outside the box, and active on the
    whole box, so they are plain linear terms. Folded them (with the output bias) into one line.
  - units 4 and 7 have kinks at -12.3 and 2.60 and are inactive on the whole box: removed.
  - units 2 and 6 have kinks inside the box (1.2257 and 0.0970): these are the only real nonlinearity.
- Unit 2 is `relu(-a*x + b)` (a decreasing ReLU). Rewrote it with `relu(-z) = relu(z) - z`, moving its
  linear part into the base line, so both knees are increasing hinges `c * relu(x - k)`. The result is
  a sum "line + two hinges", which shows the curve is convex with three segments
  (slopes 0.439, 2.019, 5.108; value 0.862 at x = 0).
- Kept the coefficients unrounded: this gives `PASS (exact)` (max error 2.8e-6).
- A snapped version `0.86 + 0.44*x + 1.6*relu(x - 0.1) + 3.1*relu(x - 1.2)` gives `PASS (snap)`
  (max error 0.12 vs tolerance 0.143), but I kept the exact one since the instructions prefer exact.
- Interpretation: a convex, increasing curve with value ~0.86 at 0, ~0 at -2 and ~7.1 at 2, slopes
  roughly multiplying from segment to segment; it resembles a 3-piece fit of exp(x) (exp: 0.135, 1, 7.39
  at -2, 0, 2). I did not replace it with exp(x): that would be a different function from the net.

## Heuristics
- For 1-input ReLU nets, compute every unit's kink `-b/w` first. Units whose kink lies outside the
  input box are either always on (fold into a linear term) or always off (delete). Often only a few
  units are real.
- Merge near-duplicate units (same kink, here units 0 and 5) before anything else; they often cancel
  (here weights -1.70 and -1.83 on nearly the same unit, both folded into the line).
- Rewrite decreasing hinges `relu(b - a*x)` as `a*relu(x - b/a) - (a*x - b)` so all hinges point the
  same way; a "line + sum of c_i*relu(x - k_i)" form directly gives slopes per segment and convexity
  (all c_i > 0 means convex).
- Check exact first with unrounded folded coefficients; then try a snapped version to see how much
  slack the snap tolerance (2% of output range) gives. Errors from snapping a kink grow with the
  slope after the kink, so the endpoint of the steepest segment is where snapped versions fail first.
- Convex increasing piecewise-linear curves with geometrically growing slopes are likely fits of
  exp-like targets; worth naming in the notes but not substituting, because the check is against the net.
