# Notes

## What I changed and why

- Input domain is x0 in [0, 4], one hidden ReLU layer of 8 units, regression output.
- For each unit I computed its kink x = -b/w and the side on which it is active within [0, 4]:
  - h1 (kink -3.72, negative slope) and h6 (kink -0.16, negative slope) are always 0 on [0, 4]: removed.
  - h0, h2, h3, h5 have kinks below 0 with positive slope: always active, so they are plain linear
    terms. Summed with the output bias they give 0.48121 + 0.50264*x0.
  - Only h4 (kink 1.682) and h7 (kink 0.416) bend inside the domain.
- Net = 0.48121 + 0.50264*x0 - 0.73508*relu(0.41610 - x0) - 0.20201*relu(x0 - 1.68195).
- Both remaining kinks make the curve concave (slopes 1.238 -> 0.503 -> 0.301), so the whole function is
  the minimum of its three segment lines. `agent.py` writes it as `min(steep, middle, flat)`:
  PASS (exact), max error 4e-7.
- Meaning: the values at integers (0.98, 1.42, 1.72, 2.02 at x0 = 1..4) follow sqrt(x0), with a
  shifted start (0.175 at 0 instead of 0). It is a 3-piece concave approximation of a square root.
  sqrt(x0) itself is off by 0.175 at x0 = 0, beyond the snap tolerance (0.037), so it is not used.
- Snapped variant `min(1.2*x0 + 0.2, 0.5*x0 + 0.5, 0.3*x0 + 0.8)` gets PASS (snap) with max error
  0.025, and `min(1.25*x0 + 0.175, 0.5*x0 + 0.48, 0.3*x0 + 0.82)` gets max error 0.006. I kept the exact
  version in `agent.py` and put the snapped one in a comment.

## Heuristics

- For 1-input ReLU nets, compute each unit's kink -b/w and drop units that are dead over the whole
  input box; fold units that are always active into one affine term with the output bias.
- What is left is a 1-D piecewise-linear function with only the in-domain kinks. Write it as
  base line + sum of c*relu(+-(x - k)).
- If all the slope changes have the same sign, the function is convex (max of lines) or concave (min
  of lines). min/max of the segment lines is shorter and more readable than the relu sum, and still exact.
- Evaluate the net at integer inputs and compare with familiar functions (sqrt, log, x^2, 1/x) to name
  the function. Then check whether that formula fits within the snap tolerance everywhere, the ends of
  the domain especially. Here it failed at x = 0 because a ReLU net can't follow sqrt's infinite slope.
- Snapping segment lines to round slopes/intercepts is usually safe for regression when the error budget
  (2% of output range) is larger than the rounding. Check the worst point, which is usually a domain end
  or a kink.
