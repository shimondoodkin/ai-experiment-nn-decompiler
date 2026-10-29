# Agent notes

## What the network does

One float input x0 in [0, 200], regression output from 8 tanh units. Reading each unit as a
smooth step `tanh(w*(x0 - c))` with centre `c = -b/w`:

- three units (h1_3, h1_4, h1_7) have centres far outside the domain (-297, -14, -387), so they
  are almost constant over [0, 200] (total variation about 0.1) and only shift the bias;
- the other five are steps of height about +-7 with centres near 0, 73, 109, 149 and 184
  (h1_1 and h1_6 have negative input weight and negative output weight, so they also rise).

Five rising steps of similar height with growing density give a monotone, convex-then-straight
curve: about -0.5 at x0 = 0, slope ~0.16 up to x0 ~ 70, slope ~0.39 from ~100 to ~190, and a
small flattening at the right edge (the last step is centred at 184 and is still unfinished at 200).

## What I changed

Replaced the whole net with a two-piece convex linear function:

    max(0.17 * x0, 0.39 * x0 - 17)      # knee at x0 ~ 77

- I tabulated the net, then fit piecewise-linear models with one and two knots (least squares)
  to see how many pieces are needed. One knot already gave max error ~0.96 against a snap
  tolerance of 1.22 (2% of the output range ~61).
- Then I grid-searched rounded coefficients for `max(a*x0 + b, c*x0 + d)` by max error.
  Slope 0.4 fails (error 1.46 at x0 = 200, where the net bends down); 0.39 with intercept -17
  and a zero-intercept first piece fits with max error 0.92.
- Result: `PASS (snap)`, max |program - net| = 0.923 (worst near x0 = 35, where the net is
  slightly convex and the program is a straight line). `PASS (exact)` is not reachable without
  keeping all the tanh units, since dropping even the nearly constant ones moves the output by
  ~0.1, far above the exact tolerance.

## Heuristics

- For a tanh/sigmoid unit, compute its centre `-b/w` and its width `1/|w|`. Units whose centre
  lies more than ~3 widths outside the input domain are constants on the domain; fold them into
  the bias.
- A unit with negative input weight and negative output weight is a rising step; normalise signs
  before reading the shape (tanh is odd).
- Several same-height steps spread across the domain add up to a smooth ramp; their spacing
  gives the local slope (step height / gap). Denser steps mean a steeper slope, so a curve made
  of unevenly spaced equal steps is a piecewise-linear function in disguise.
- For 1-input regression, tabulate the net and do a least-squares piecewise-linear fit with 1 and
  2 knots first: it shows quickly how many pieces the snap tolerance needs.
- A convex 2-piece fit is `max(line1, line2)`; this is shorter and clearer than `relu` sums.
- When rounding slopes, check the domain ends: the last step is often only half-finished at the
  boundary, so the net flattens there and a "nicer" slope (0.4 vs 0.39) can fail at the edge.
- Grid-search snapped coefficients by max (not mean) error, since the checker uses max error.
