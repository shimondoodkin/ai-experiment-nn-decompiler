# Notes

## What the net computes

A least-squares fit of the net on 20k uniform points against {1, x0, x1, x0², x0·x1, x1²} gave
x0·x1 with coefficient 1.000 and every other coefficient under 0.01. So the net is a product.
Most hidden units point along (1, ~0.85) or (1, ~-0.9), which means they build piecewise-linear
squares of s = x0 + x1 and d = x0 - x1 (x0·x1 = (s² - d²)/4).

## What I changed

- I replaced the 16 ReLU units with `x0 * x1`. On its own that fails: max error 0.244 against a
  snap tolerance of 0.154. The misses are at the corners (2, 2), (-2, 2) and (2, -2), where the
  net is flatter than the product (for example -3.76 instead of -4 at (-2, 2)).
- Clipping the output (`min(max(p, -3.75), 3.875)`) or scaling it (0.95 or 0.97 times p) still
  failed, by 0.17 to 0.22. Near the corners the net is linear along the diagonal, not constant.
- The model that fits is that the net's square u²/4 continues as its tangent line beyond |u| = 3.
  That is u²/4 - relu(|u| - 3)²/4. For the product this gives
  `p - 0.25*relu(s - 3)² + 0.25*relu(|d| - 3)²`.
  - The s term needs only s > 0. At (-2, -2) the net gives 3.94, close to the unflattened value 4.
  - The knots 3, 3.2 and 3.25 all give the same max error, so I kept 3, the nice value.
- Result: `PASS (snap)`, max |program - net| = 0.136. The remaining error is the net's own
  interior ripple around x0·x1, about ±0.1 across the whole domain, which no smooth formula
  removes. An exact pass would mean rebuilding the ReLU units one by one.

## Heuristics

- Before reading individual units, fit a low-degree polynomial (1, x, y, x², xy, y²) by least
  squares on random samples. A coefficient near 1 with the rest near 0 names the function at
  once (here x0·x1).
- When hidden-unit directions cluster around (1, 1) and (1, -1), suspect a product built from
  squares of the sum and the difference: xy = ((x+y)² - (x-y)²)/4.
- A smooth formula on a ReLU net can fail only at the corners of the box. Print the residual
  net - formula on a grid to find where. Corner errors usually come from the net's
  piecewise-linear approximation running straight past its last knot.
- Model a "straight past the last knot" effect as the curve minus a squared hinge. For u², the
  tangent continuation beyond |u| = a is u² - relu(|u| - a)². A clip or a max/min with a tangent
  line is wrong here, because a tangent line lies on one side of the curve everywhere, so max or
  min picks it far from the knot too.
- Check each corner on its own. Only some corners may be flattened, so a one-sided hinge
  (relu(s - a) instead of relu(|s| - a)) can be enough.
- When several knot values give the same max error, the limit is somewhere else, usually
  interior ripple. Keep the nicest value.
