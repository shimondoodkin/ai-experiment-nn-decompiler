# Notes

## What I changed and why

- The dump is a 1-input, 8-unit tanh layer feeding a linear output, on x0 in [0.5, 4].
- I tabulated the net at 36 points next to sqrt(x), log(x) and 1/x. The output falls from about
  1.975 at x0=0.5 to 0.245 at x0=4, convex and decreasing. It matches 1/x0 almost exactly:
  0.99345 at 1, 0.49709 at 2, 0.33296 at 3, 0.24515 at 4.
- So the whole network became `return 1 / x0`, and none of the 8 hidden units is needed.
- Check: PASS (snap), empirical over 204203 inputs. Max |program - net| = 0.0251, against a snap
  tolerance of 0.0346. The largest errors are at the left edge (x0 near 0.5), where the
  curve is steepest and the net undershoots 1/x (1.975 against 2.0). In the middle of the domain the
  error is below about 0.007.
- PASS (exact) would mean reproducing the net's own approximation error, so I did not attempt it.

## Heuristics

- For 1-input regression nets, tabulate the output on a grid next to a few candidate
  functions (x, x^2, sqrt, log, exp, 1/x) before simplifying unit by unit. A smooth tanh/sigmoid sum
  is often just a smooth approximation of one elementary function, and spotting that function beats
  merging units.
- Read off values at integer inputs (f(1), f(2), f(3), f(4)): ratios such as f(1)/f(2) = 2 and
  f(1)/f(3) = 3 identify 1/x right away; constant differences point to log, constant ratios to exp.
- Expect the largest error at the domain edges, and most of all where the target is steepest.
  Small nets fit the steep end poorly, so a clean formula that is off only at an edge is still the
  right answer.
- Try the clean formula with no fitted constants first. Add a scale or offset only if the check
  fails; here it passed with room to spare.
