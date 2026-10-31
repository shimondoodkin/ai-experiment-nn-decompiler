# Notes

## What the network computes

The net is 2 -> 16 tanh -> 1 regression on x0, x1 in [-2, 2]. Its value on a 9x9 grid
matches `x0 * x1` to about 0.01 inside the box. The largest deviations are at the corners,
where the tanh units saturate and the net falls short of |x0*x1| = 4 by up to 0.11
(for example net(2, -2) = -3.889).

All 16 hidden units have weight vectors close to (a, a) or (a, -a). So every unit reads
either s = x0 + x1 or d = x0 - x1. The output is a sum of tanh bumps in s plus a sum in d,
which is a smooth approximation of s^2/4 - d^2/4 = x0*x1 (the identity
x*y = ((x+y)^2 - (x-y)^2) / 4).

## What I changed

The 16 tanh units and 17 output coefficients are replaced by one product:

    def net(x0, x1):
        product = x0 * x1
        return product

`check.py`: PASS (snap), empirical over 249,370 inputs, max |program - net| = 0.111 against a
snap tolerance of 0.158. No input is outside tolerance. An exact pass would need the tanh
saturation at the corners modelled too. That would add back most of the network for an error
that is a training artefact, not part of the function.

(First try: I read the grid's axes the wrong way round and checked `-x0*x1`. It failed at every
corner, which showed the sign error at once.)

## Heuristics

- Evaluate the net on a coarse grid (9x9 for 2 inputs) and print it before reading any weights.
  Products, sums, abs and max show up at once in the table.
- Get the grid's axis order right (meshgrid `indexing='ij'`: row = x0, column = x1). If a formula
  fails at every corner with the sign flipped, check the orientation first.
- Hidden weight vectors close to (a, a) and (a, -a) mean the net works on x+y and x-y. Look for even
  functions of those (squares give x*y = ((x+y)^2 - (x-y)^2)/4, abs gives max/min).
- The worst error of a correct formula sits at the corners of the box, where tanh/sigmoid units
  saturate. A residual that grows only at the box edges is a training artefact, not a missing term.
- Compare the max error with the snap tolerance (2% of the output range). If a one-line formula fits
  inside it, stop; extra terms that only chase saturation make the program harder to read.
