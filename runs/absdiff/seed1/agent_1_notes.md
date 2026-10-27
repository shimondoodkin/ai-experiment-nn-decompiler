# Notes

## What the network does
`net(x0, x1) = |x0 - x1|` (absolute difference).

## What I changed and why
- Hidden neurons 4 and 7 have tiny input weights (|w| < 0.02) and bias -0.366, so over the input box
  (|x| <= 3) their pre-activation is at most ~ -0.29: always zero after relu. Removed as dead.
- Neurons 0, 5, 6 all compute relu(c*(x0 - x1) - small bias) (weights equal and opposite).
  Summing their output weights times input scales: 0.449*0.263 + 0.577*0.776 + 0.775*0.561 = 1.000,
  so together they are relu(x0 - x1).
- Neurons 1, 2, 3 likewise compute relu(c*(x1 - x0)) with combined gain 1.000, i.e. relu(x1 - x0).
- relu(d) + relu(-d) = |d|.
- The small hidden biases (~-0.005 to -0.023) and output bias (+0.0205) almost cancel away from
  d = 0 (the output-weighted bias sum on each side is ~-0.0206); they only add a ~0.02 bump near
  x0 = x1, a training artifact. Snapped to 0.
- Result: `return abs(x0 - x1)`. check.py: PASS (snap), max deviation 0.0205 (at x0 == x1).

## Heuristics
- Find dead neurons first: bound each pre-activation over the input box (|b| > sum |w_i|*range_i
  with the wrong sign means always 0; the opposite means always linear).
- Group neurons by the direction of their weight vector (normalize; look for w0 ~ -w1 etc.).
  Neurons along the same direction collapse: sum of (output weight * input scale) gives the
  effective gain, which is often exactly 1 or a small integer.
- A pair of groups along +d and -d with equal gain is abs(d); unequal gains give a leaky/asymmetric
  piecewise-linear function (a*relu(d) - b*relu(-d)).
- Small hidden biases combined with an output bias that cancels their weighted sum are smoothing
  artifacts around a kink; snap them to 0 together rather than individually.
- Compare the max error of the snapped version to the tolerance; if the only residual is at a kink,
  the snap is right.
