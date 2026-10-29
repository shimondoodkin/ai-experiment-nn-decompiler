# Notes

## What the network does
1 float input in [-2, 2], binary output, 4 tanh hidden units.

- `h1_0` and `h1_3` have near-zero input weights (|w| < 0.002) and large biases, so they are
  constants over the domain (tanh(-2.54) ~ -0.988, tanh(2.80) ~ 0.993). Folded into the bias.
- `h1_1 = tanh(8.13*(x0 - 0.49995))` is a soft step up at x0 ~ +0.5.
- `h1_2 = tanh(-8.31*(x0 + 0.49844))` is a soft step down at x0 ~ -0.5.
- Output = -8.1*h1_1 - 8.1*h1_2 + const(~ -7.94). The two steps with equal negative weights form
  a bump: score ~ +8.24 in the middle, ~ -7.94 on both sides.

Numerically scanning the score on a 4M-point grid gives zero crossings at x0 = -0.500792 and
x0 = 0.502291. So: class 1 iff x0 is inside a window of half-width 1/2 around 0.

## Final program
`return abs(x0) < 0.5` - PASS (snap); all disagreements are in the net's unsure zone right at the
edges. A version with the raw thresholds (`-0.500792 < x0 < 0.502291`) passes exact, but the
snapped symmetric window is the meaningful one, so `agent.py` keeps it.

## Heuristics
- tanh/sigmoid units whose input weight is tiny relative to the input range (|w| * range << 1)
  are constants: replace with tanh(bias) and fold into the output bias.
- A tanh unit `tanh(w*x + b)` with large |w| is a step at x = -b/w; read the threshold off as
  -b/w before snapping.
- Two steps at a and b combined with same-sign output weights (one rising, one falling) form a
  window/bump; for a binary output this is just `a < x < b`, and a symmetric pair becomes `abs(x) < c`.
- For 1-input binary nets, scan the score on a dense grid to find the exact zero crossings; then
  snap them to nearby simple values (here +-0.5) and check that disagreements stay in the unsure zone.
- If snapped thresholds only pass "snap", keep them anyway when they are clearly the intended
  values; the raw crossings give the "exact" variant if needed.
