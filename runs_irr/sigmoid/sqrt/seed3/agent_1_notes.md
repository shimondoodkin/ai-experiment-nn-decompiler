# Notes

## What the network does
One float input x0 in [0, 4], 8 sigmoid units, regression output. Tabulating the dump showed
net(1) = 1.0004, net(2) = 1.4137, net(3) = 1.7354, net(4) = 1.9940: very close to sqrt(x0).

## What I changed
- Replaced the 8-unit sigmoid layer with `sqrt(x0)`. Plain `sqrt(x0)` is off by more than the snap
  tolerance (0.037, 2% of the output range) only on [0, 0.019]. There the net cannot follow the
  infinite slope of sqrt at 0, and it starts at 0.143 instead of 0.
- Fixed that region with a constant floor: `max(sqrt(x0), 1/6)`. Floors I tried: 0.15 (max error
  0.031), 0.16 (0.026), 1/6 (0.023), 0.2 (FAIL). I kept 1/6, the simple fraction with the most margin.
- Result: `PASS (snap)`, max |program - net| = 0.023, on 218k empirical inputs. An exact pass is not
  possible with a closed form, because the net is a sum of sigmoids and not exactly sqrt.

## Heuristics
- For a 1-input regression net, tabulate it at integer and simple inputs first, and compare against
  candidate functions (sqrt, log, log1p, exp, x^2, 1/x). Values at perfect squares (1, 4) that land
  on integers (1, 2) point to sqrt.
- Find where the candidate formula fails (|error| > snap tolerance). If the failures sit in a small
  region where the true function has a singularity or an infinite slope (sqrt at 0, log at 0, 1/x
  near 0), the net is smoothing it. Patch only that region with the simplest guard: a constant floor
  or ceiling (`max(f, c)` / `min(f, c)`). Pick c from the net's value at the edge and snap it to a nice value.
- Try several nice values for the patch constant and keep the one with the smallest max error. Do
  not use a guard line `a + b*x`: it can cross the formula a second time and break the fit far away.
- Sigmoid units with tiny output weights (here |w| ≈ 0.001) or a near-linear range over the
  domain only add curvature. Look at the whole net's curve, not unit by unit, when the net
  approximates a smooth elementary function.
