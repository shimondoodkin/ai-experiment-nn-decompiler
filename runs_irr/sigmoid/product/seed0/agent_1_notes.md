# Notes

## What I did

- The dump is 2 inputs -> 16 sigmoid units -> 1 linear output (regression on [-2, 2]^2).
- Every hidden unit's weight vector is close to (1, 1) or (-1, 1) up to scale, i.e. each unit is a
  soft function of s = x0 + x1 or d = x1 - x0. That pattern (only the sum and difference
  directions) suggests the polarization identity x0*x1 = (s^2 - d^2) / 4.
- A 9x9 grid of the net's outputs read as a multiplication table (net(1,1) = 0.25 at x = 0.5,
  net(-2,-2) = 3.94, net(-2,2) = -3.89), and the output is ~0 on both axes.
- A least-squares fit on 20k uniform points to {x0^2, x1^2, x0*x1, x0, x1, 1} gave
  coefficient 0.9998 on x0*x1 and < 3e-3 on everything else, so all other terms were dropped.
- `agent.py` is `return x0 * x1`. Check: PASS (snap), max |program - net| = 0.110 against a snap
  tolerance of 0.157. The largest errors are at the four corners (|x0*x1| near 4), where the
  sigmoids saturate and the net undershoots the product by about 0.1 (about 2.7%).
- An exact pass would need the full 16-unit network, so I kept the formula.

## Heuristics

- Look at the hidden units' weight directions first. If they cluster on a few directions (here
  x0+x1 and x1-x0), the net is a function of those combinations. Sum and difference directions
  together with a near-zero output on both axes point to a product, via x*y = ((x+y)^2 - (x-y)^2)/4.
- Print a coarse grid of outputs over the input box. Low-order polynomials (products, squares)
  are easy to read off a table.
- Fit a small polynomial basis by least squares, then snap the coefficients: keep terms with a
  coefficient near a simple value and drop those near 0.
- Expect the largest errors at the edges and corners of the box, where sigmoid units saturate and
  the net flattens the function's growth. If those errors stay inside the snap tolerance, the clean
  formula is the better program; don't add correction terms only to follow the saturation.
