# Notes

## What the network does

`dump.py` is a 1 -> 8 -> 1 ReLU net on x0 in [0, 4]. Checking each hidden unit's sign over the domain:

- Units 0, 2, 3, 4, 6 have a negative slope and a negative bias, so their pre-activation is negative on all
  of [0, 4]: always 0. Removed.
- Unit 5 (`0.167*x0 - 0.723`) turns on at x0 = 0.7227/0.1669 = 4.33, beyond the upper bound 4: always 0. Removed.
- Units 1 and 7 have a positive slope and a positive bias: always on, so `relu` is the identity. Inlined.

What remains is affine: `0.29613*(0.50453*x0 + 0.52764) + 0.32622*(0.77313*x0 + 0.29980) + 0.26928`
= `0.40161178*x0 + 0.52332852`.

## Versions

- `agent.py`: `0.40161178*x0 + 0.52332852`. PASS (exact), max error 2.6e-7.
- Snapped `0.4*x0 + 0.5`: PASS (snap), max error 0.0298 against a tolerance of 0.0321 (the error peaks at
  x0 = 4). `0.4*x0 + 0.52` also passes snap, with max error 0.0098. I kept the exact line in `agent.py`,
  because it is just as short and is exact; the snapped form is in a comment.

Whatever the network was trained to approximate, on this domain it computes only a straight line; a curved
target (for example a square root or a log) would have been approximated by a single line here.

## Heuristics

- For each first-layer ReLU, evaluate the pre-activation at the input box's corners (for 1 input, at lo and hi).
  Negative at every corner means always off: delete it. Positive at every corner means always on: replace
  `relu(z)` by `z`. A unit only matters if its kink `-b/w` lies strictly inside the domain.
- Negative slope with negative bias on a domain that starts at 0 is a quick sign of a dead unit; positive
  slope with positive bias is always on.
- Once the always-on units are inlined, fold everything into one affine expression and recompute the
  coefficients by multiplying out, rather than snapping each weight separately (snapping each weight
  separately adds up the errors).
- Snapping a regression line: the error of a rounded line is largest at an end of the domain, so check
  |d_offset| + |d_slope| * (max |x|) against the tolerance before trying it. If the rounded line is near
  the tolerance edge, prefer the exact coefficients, or round only the offset or only the slope.
- Count the active kinks before writing any piecewise code: a net with 8 hidden units can turn out to be
  a single line on its domain.
