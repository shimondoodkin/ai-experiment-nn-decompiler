# Notes

## What I changed and why

The dump is a 2 -> 16 tanh -> 1 regression net on x0, x1 in [-2, 2]. Most hidden units have weights
of about equal size on both inputs, pointing along the diagonal (1, 1) or the anti-diagonal (1, -1).
That points to a function of x0 + x1 and x0 - x1. Evaluating the net on a 9x9 grid showed:

- zero (about 0.01) along both axes (x0 = 0 or x1 = 0);
- linear growth in each input when the other is held fixed, with slope equal to the other input;
- values close to +-4 at the corners.

That is the product: x0 * x1 = ((x0 + x1)^2 - (x0 - x1)^2) / 4, which explains why the tanh units sit
on the two diagonals (each set builds one of the two squares out of tanh pieces).

`agent.py` returns `x0 * x1`. Check: PASS (snap), empirical, 260545 inputs, max |program - net| =
0.123 against a snap tolerance of 0.157. The residual is small everywhere inside the box
(max 0.027 for |x0|, |x1| <= 1.5, RMS 0.018, mean +0.017). The largest errors are at the
anti-diagonal corners (-2, 2) and (2, -2), where the net reaches only about -3.88 instead of -4:
the tanh pieces saturate near the edge of the training box. I did not add the +0.017 offset or a
corner correction: they would only fit the net's approximation error, not the function.

An exact match is not possible with a short program: the net is a smooth sum of 16 tanh units, and
the only way to reproduce it exactly is the dump itself.

## Heuristics

- Look at the hidden units' weight directions first. Units that come in groups with |w0| ~ |w1| along
  (1, 1) and (1, -1) suggest a function of x0 + x1 and x0 - x1; for a regression output, the product
  x0 * x1 = ((x0 + x1)^2 - (x0 - x1)^2) / 4 is the first candidate.
- Print the net on a coarse grid (for example 9x9 over the box) before snapping any weight. Zero rows
  and columns on the axes plus slopes equal to the other input identify a product immediately.
  Recognising the function from the grid is much faster than simplifying 16 smooth units one by one.
- With smooth (tanh, sigmoid) nets, an exact pass is usually impossible for a clean formula; aim for a
  snap pass and report the residual's size and where it is largest (usually the box corners, where
  the units saturate).
- Do not fit the net's error (small constant offsets, corner bends): check that the clean formula
  already passes, and keep it if it does.
- Compare the max error with the snap tolerance. If the margin is small, look at where the error
  lies; if it is only at the box edges, it is the net's approximation error, not a missing term.
