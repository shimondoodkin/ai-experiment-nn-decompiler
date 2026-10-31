# Agent notes

## What the network computes

One float input x0 in [-2, 2], 32 ReLU units, regression output. A 1-input ReLU net is a
piecewise-linear function of x0, so I evaluated it on a grid and listed each unit's kink
position (-b/w) and slope contribution (w*v).

- Sampled values: f(-2)=0.141, f(0)=0.996, f(1)=2.722, f(2)=7.380, which are close to
  e^-2=0.135, e^0=1, e^1=2.718 and e^2=7.389.
- Kinks are spread across the whole domain. Their slope contributions grow steadily toward
  the right (about 0.3 near x=-0.3, 0.45 at 0.56, 0.65 at 1.04, 0.85 at 1.47, 1.07 at 1.84).
  The slope keeps increasing, so the function is convex, as a piecewise-linear fit of a
  convex curve would be.
- Some units do nothing: h1_15 and h1_10 have weights near 0. h1_18, h1_19 and h1_28 have
  their kinks at x of about 1.997, so they matter only at the very right edge. A few
  left-side units have small negative contributions, which are fitting noise.

## Program

```python
def net(x0):
    return exp(x0)
```

Check: `PASS (snap)`, empirical over 260,623 inputs. The largest difference is 0.0148 (near
x0 = 1.745, between two kinks) against a snap tolerance of 0.145. That is about 0.2% of the
output range. All 32 units and the constant are replaced by one formula.

An exact pass (atol 1e-5) would require copying the net's roughly 20 effective kinks
one-for-one. That would be the net's chord approximation of exp, not a meaningful program,
so I did not write it.

## Heuristics

- For a 1-input ReLU net, first list each unit's kink position -b/w (keep the ones inside the
  domain) and its slope change w*v. Units with kinks outside the domain are either constant
  there or linear there, and fold into the affine part.
- Look at the sign pattern of the slope changes. If they are all (or nearly all) positive,
  the function is convex. If they grow steadily, the curve is smooth, such as exp, x^2 or
  softplus, and not a few hinges. A few large changes with gaps between them point to a
  genuine piecewise function such as abs, clamp or max.
- Compare a few landmark values against well-known functions: f(0), f(1), f(-1) and the ends
  of the domain. f(0)=1 with f(1)=e and f(2)=e^2 identifies exp. The ratio f(x+1)/f(x)
  staying constant (about 2.72 here) confirms an exponential.
- Use the language's built-in smooth functions (exp, log, sqrt, sigmoid, tanh) as
  candidates before trying to snap the hinges. A smooth target fitted by ReLUs shows up as
  many evenly spread small kinks, and a single formula usually passes with a large margin.
- Check against the net and see where the largest error falls. Here it sat between two kinks,
  where the net's chord is furthest from the curve. This is expected for a piecewise-linear
  fit and is not a sign that the formula is wrong.
- Near-zero weights (|w| < 1e-2 with a small output weight) and kinks at the very edge of the
  domain are training leftovers. Drop them.
