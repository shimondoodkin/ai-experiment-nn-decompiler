# Notes

## What the network does

1 float input on [0.1, 5], 8 sigmoid units, regression output. Tabulating the dump on a grid
showed it tracks `log(x0)` closely: net(1) = -0.0046, net(2) = 0.6953 (ln 2 = 0.6931),
net(3) = 1.0964 (ln 3 = 1.0986), net(5) = 1.6048 (ln 5 = 1.6094). The 8 sigmoids are just a
smooth fit of the log curve; no single unit has its own meaning.

## What I changed

- Replaced all 8 sigmoid units with `log(x0)`.
- Plain `log(x0)` failed only at the far left end: near x0 = 0.1 the net gives -2.204 while
  ln(0.1) = -2.303 (error 0.098, tolerance 0.076). The net cannot follow the steep part of the
  log, so it bends flatter there. Clamping from below with `max(log(x0), -2.2)` fixes this.
- Final program: `max(log(x0), -2.2)`. Result: PASS (snap), max |program - net| = 0.065 against a
  tolerance of 0.076, empirical check over 220,973 inputs. In the interior (x0 > 0.2) the error
  is below about 0.02.
- I also tried `log(max(x0, c))`: c = 0.115 gives max error 0.054, but -2.2 is a rounder
  constant and also passes, so I kept the output clamp.

## Heuristics

- For 1-input regression nets, tabulate the net on a grid first and compare it with standard
  functions (log, exp, sqrt, 1/x, x^2). Check values at "landmark" inputs (net(1) ~ 0 points to
  log; net(0) ~ 1 to exp; net(1) ~ 1 to powers).
- Sigmoid networks fitting a function with a steep end (log near 0, 1/x near 0, sqrt near 0)
  usually underfit that end: the net is flatter than the true function there. Write the clean
  formula, then patch only the failing end with a clamp (`max`/`min` on the output, or on the
  input inside the function), with a round constant.
- Look at where the max error is and how it compares with the snap tolerance; fix the region
  that breaks it, not the whole curve.
- Prefer a round clamp constant (-2.2) over a tuned one when both pass; it reads better and
  says the same thing.
