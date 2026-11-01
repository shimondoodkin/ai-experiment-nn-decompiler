# Notes

## What the net computes

`dump.py` is a 1 -> 8 tanh -> 1 regression net on x0 in [0.1, 5]. Sampling it on a grid and
comparing with candidate functions (log, sqrt, 1/x, exp(-x)) showed that it follows `log(x0)`
closely: net(1) = 0.003, net(2) = 0.694, net(3) = 1.098, net(4) = 1.387, net(5) = 1.607.

## What I changed

- I replaced all 8 tanh units with `log(x0)`. With plain `log`, the check fails only at the low
  edge, x0 < ~0.11. There `log` keeps falling, but the net bends off: net(0.1) = -2.207, while
  log(0.1) = -2.303, which is 0.096 off against a snap tolerance of 0.076.
- To fix the edge I clamped `log` from below at a round constant: `max(log(x0), -2.2)`.
  - Max error: 0.061, at x0 ~ 0.111, where the clamp starts.
  - RMS error: 0.0049, the lowest of the candidates I tried.
- I also tried shifting the argument, `log(x0 + c)`:
  - c = 0.005 has a smaller max error (0.047) but a higher RMS (0.0076);
  - c = 0.01 has a max error of 0.074 and an RMS of 0.013.

  I kept the clamp, because the net matches plain `log` everywhere except the edge and the clamp
  changes nothing above x0 ~ 0.11.
- Result: `PASS (snap)`, max |program - net| = 0.061. `PASS (exact)` (atol 1e-5) would need
  essentially the whole tanh sum, so I did not try for it.

## Heuristics

- For a 1-input regression net, sample the output on a grid first and put it next to standard
  functions: log, sqrt, 1/x, exp(+-x), x^2. Clues that can identify the function quickly:
  - net(1) ~ 0 and net(2) ~ 0.693 point to log;
  - net(4) ~ 2 points to sqrt.
- Steep functions (log near 0, 1/x near 0) are where a small net bends away from the true
  formula, at the steep end of the domain. If the clean formula fails only there, patch only the
  edge instead of dropping the formula:
  - a clamp `max(f, c)` / `min(f, c)` with a round c near the net's edge value;
  - or a small argument shift `f(x + c)`.

  Pick the patch by max error first (it must be within the snap tolerance), then by RMS error,
  and prefer the patch that leaves the rest of the domain untouched.
- In a 1D tanh net, units with large |slope| (here -4.1, -3.1, -2.0) and centres near the low
  edge are what bend the curve there. Units with small slopes (|w| < 0.35) together make up the
  slow log-like trend.
