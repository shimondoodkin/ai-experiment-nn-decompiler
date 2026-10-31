# Notes

## What the network is
A 1 -> 32 -> 1 ReLU regression net on x0 in [-2, 2]. A 1-input ReLU net is a piecewise-linear
function, so I computed its knots and the slope on each segment from the weights (`scratch/pw.py`).
- 8 units are dead on the whole domain (3, 8, 9, 12, 21, 22, 24, 29). Units 6, 12 and 21 have weights near 0.
- 2 units are active on the whole domain, so they are just linear terms (1, 31).
- The other 22 units each add a kink inside the domain.
- The slopes rise steadily from 0.18 at x0=-2 to about 6.8 near x0=2, so the function is convex.
  Its values are f(-2)=0.13, f(0)=1.00, f(1)=2.72 and f(2)=7.37, which are e^-2, 1, e and e^2.
  The slope on each segment is about equal to the function's value there, which is how exp behaves.
- There is a small dip in the slope at x in [-0.33, -0.29] and a few tiny segments near x=2. These
  are training noise and nothing else.

## What I changed
I replaced the whole network with `exp(x0)`.
- Check: PASS (snap), 263k empirical inputs.
- Max |program - net| = 0.0199 at x0 = 1.68 (net 5.354, exp 5.374). The snap tolerance is 0.145, so
  the error is about 0.27% of the output range. The mean residual is +0.0016.
- An exact pass would mean keeping the ~22-knot spline, noise included. That is not readable, so I
  kept the formula.

## Heuristics
- With one input, a ReLU net is a 1-D linear spline. List its knots (-b/w that fall inside the
  domain) and the slope of each segment, then look at the shape: constant, linear, |x|, convex or
  concave, steps.
- Before you go through the units, delete the ones that are dead on the domain (relu argument <= 0
  at both ends). Fold the ones that are active on the whole domain into one affine term.
- If the slopes keep growing and the slope is about equal to the value (f' ~= f), try exp(x). Look up
  the values at simple points too: f(0)=1, f(1)=e~2.718 and f(-1)=1/e~0.368 point to exp. The same
  checks work for others: f(1)/f(0) for scaled exps, f(x)^2 for sqrt, f(x) ~= x^2 for squares.
- A smooth target fitted with ReLU always leaves an error at the knots (here about 0.02 between knots
  where the curve bends most). That error is the normal cost of the spline approximation. Accept the
  smooth formula when the max error is a small fraction of the snap tolerance. Don't snap toward the
  spline.
- Small blips in the slope sequence (one segment that goes against the trend, or very short segments
  at the edge of the domain) are noise. Ignore them when you guess the formula.
