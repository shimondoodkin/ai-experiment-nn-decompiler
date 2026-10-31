# Notes

## What the network computes

The net is 1 input -> 8 ReLU units -> 1 output, x0 in [-2, 2], regression.

- Two units can never fire on [-2, 2]: unit 1 (kink at x0 = -3.72) and unit 4 (kink at x0 = 3.02).
  They are dead code.
- The other six kinks sit at about -1.42, -0.84, -0.28, 0.27, 0.84 and 1.43, nearly symmetric around 0.
- The slopes between kinks are -3.42, -2.26, -1.11, -0.01, 1.10, 2.26 and 3.43. That is an even,
  convex function whose slope grows linearly with x0.
- Sample values: f(0) = 0.02, f(+-0.5) = 0.27, f(+-1) = 1.00/1.01, f(+-1.5) = 2.22/2.23,
  f(+-2) = 3.93/3.94. This is x0^2, with the network's piecewise-linear interpolation of it
  (7 linear pieces).

## Program

`agent.py` returns `x0 * x0`.

`check.py` result: PASS (snap), max |program - net| = 0.0715 against a tolerance of 0.0785.
The largest errors are at the ends of the domain (x0 = +-2), where the net's last linear piece
reaches only 3.93 instead of 4. Away from the kinks the net follows x0^2 to within about 0.03.

## Heuristics

- For a 1-input ReLU net, first list each unit's kink (-b/w). Drop units whose active side never
  meets the input box: they are constant 0 there.
- Evaluate the net at the kinks and the box ends, and compute the slope of each piece. Kinks that
  are symmetric about 0 plus slopes that change sign suggest an even function such as |x| or x^2.
  Slopes that grow roughly linearly (equal slope steps at roughly equal kink spacing) mean a
  quadratic.
- Compare the net at simple points (0, +-0.5, +-1, +-1.5, +-2) with candidate formulas before
  fitting anything. Integer and half-integer inputs make x^2 easy to spot.
- A piecewise-linear net that approximates a smooth curve is least accurate at the box ends, where
  its last segment stops bending. Check the program's margin at the corners first. The snap
  tolerance is 2% of the output range.
- Prefer the clean formula (x0 * x0) over a fitted scale or offset (such as 0.98*x^2 + 0.02) when the
  clean formula passes. The fit would follow the net's interpolation error, not the function the
  net learned.
