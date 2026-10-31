# Notes

## What the network computes
On the input box x0 in [0.5, 4] only two of the eight hidden units ever turn on:

- h1_0, h1_2, h1_3, h1_4, h1_6 have negative slope and negative bias, so their pre-activation is
  negative for every x0 > 0: always 0, removed.
- h1_5 turns on only at x0 > 6.56, outside the box: always 0, removed.
- h1_7 turns on at x0 > 0.245, below the box: always on, so it is a plain line and folds into the
  output constant and slope: -0.2518 * (0.6955 x0 - 0.1707) + 0.8394 = 0.8824 - 0.1751 x0.
- h1_1 is the only real kink: 1.167 * relu(1.3959 - 1.1127 x0) = 1.2986 * relu(1.2545 - x0).

Exact simplified form: 0.88236 - 0.17511 x0 + 1.29856 * relu(1.25451 - x0).

## Snapping
Kink 1.2545 -> 1.25, constant 0.8824 -> 0.88, slope -0.1751 -> -0.175, hinge weight 1.2986 -> 1.3.
Result: `0.88 - 0.175*x0 + 1.3*relu(1.25 - x0)`, max error 0.0081 against a snap tolerance of 0.032,
PASS (snap). (Hinge weight 1.25 or slope -1/6 each cost too much, 0.045 / 0.026.)

The curve is decreasing and convex (slope -1.475 then -0.175) from 1.77 at x0=0.5 to 0.18 at x0=4,
so it looks like a two-segment approximation of a convex decreasing function such as about 1/x
(values 1.77, 1.04, 0.53, 0.18 at 0.5, 1, 2, 4), but it is only matched against the net, so I keep
the piecewise-linear program.

## Heuristics
- For each ReLU unit, solve w*x + b = 0 and compare the root with the input box; with the sign of w
  this tells you at once whether the unit is always off (delete), always on (fold into the linear
  part), or a real kink inside the box.
- Units with negative weight and negative bias on a positive-only input domain are dead.
- Fold all always-on units and the output bias into one line `c + m*x`, then rewrite each remaining
  unit as `a * relu(k - x)` / `a * relu(x - k)` so the kink position k is explicit and easy to snap.
- Snap the kink position first (it matters most near the kink), then the line, then hinge weights;
  check the max error of each candidate with a dense numeric sweep before running the checker.
- Compare candidate snaps against the snap tolerance (2% of the output range); choose the
  coarsest values that keep a comfortable margin.
