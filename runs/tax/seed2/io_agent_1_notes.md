# Notes

Single float input x0 in [0, 200], regression output. Sorting the samples and printing where the
finite-difference slope changes showed three linear pieces (plus a tiny noisy wiggle near 0):

| range      | fitted slope | snapped |
|------------|--------------|---------|
| 1 .. 19    | 0.1182       | 0.12    |
| 23 .. 78   | 0.1998       | 0.20    |
| 82 .. 199  | 0.4011       | 0.40    |

Intersections of the fitted lines sit at x ~ 20.6 and x ~ 79.7, snapped to 20 and 80. The shape is a
progressive tax schedule (marginal rates 12% / 20% / 40% with bracket edges 20 and 80), so the program
is written as a base rate plus rate increments above each bracket:

    tax = 0.12*income + 0.08*relu(income - 20) + 0.2*relu(income - 80)

Intercept snapped to 0 (fitted ~ -0.2; the near-zero region shows a small negative offset of about
-0.12 that is network residue, not a meaningful feature). Result: `PASS (snap)`, max error 0.30
against a snap tolerance of 1.25, worst points at the 80 kink, where the net's corner is rounded.

## Heuristics

- For a 1-D input, sort the samples and print the slope between neighbours only where it changes:
  piecewise-linear structure (ReLU kinks) shows up immediately.
- Fit a line per clean segment (stay away from kinks), then get breakpoints from line intersections
  rather than from where the slope visibly changes; the net rounds corners.
- Snap slopes to simple values and breakpoints to round numbers; express a convex piecewise-linear
  function as `a*x + sum(delta_slope_i * relu(x - t_i))` - continuity comes for free.
- Increasing marginal slopes over a money-like range (0.12 -> 0.2 -> 0.4) suggest a tax/tariff
  schedule; name variables accordingly (income, tax, brackets).
- Ignore small offsets/wiggles near the domain edge when they are well within tolerance; try the
  clean formula first and only add constants if the checker requires them.
