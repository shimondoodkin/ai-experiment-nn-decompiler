# Notes

## What I did

- The dump is 2 -> 16 sigmoid -> 1 regression on x0, x1 in [-2, 2].
- The hidden units' weight vectors all lie close to two directions, (1, 1) and (1, -1), with
  slopes near +-0.8 to +-1.6 and biases of various sizes. That pattern (sigmoids along the sum
  and difference directions, output weights of mixed sign) suggests something like
  a*s(x0+x1) - a*s(x0-x1), i.e. the polarisation identity x0*x1 = ((x0+x1)^2 - (x0-x1)^2)/4.
- Printing the net on a 9x9 grid confirmed it: the table is x0*x1 to about 0.01 inside the box
  (0 on both axes, 1 at (1,1), -1 at (1,-1), 4 at (2,2) with net 3.92).
- `agent.py` is `return x0 * x1`. Check: PASS (snap), max |program - net| = 0.097 against a
  snap tolerance of 0.157 (2% of the output range). The largest errors are at the corners
  (|x0| = |x1| = 2), where the sigmoids saturate and the net under-shoots |x0*x1| = 4 by ~0.08-0.1.
- PASS (exact) is not reachable with a closed formula: the net is a sum of 16 sigmoids and
  differs from x0*x1 by up to 0.1; only something as large as the dump would match to 1e-5.

## Heuristics

- Before simplifying units one by one, print the net on a coarse grid (e.g. 9x9 over the
  domain). A clean table often reveals the formula directly (here a multiplication table).
- Group hidden units by weight direction. Units along (1,1) and (1,-1) with opposite-signed
  output weights point to a function of x0+x1 and x0-x1; the classic case is the product via
  ((x0+x1)^2 - (x0-x1)^2)/4. Units along a single axis point to a function of that input alone.
- Zero along both axes plus a sign pattern of + in quadrants I/III and - in II/IV is the
  signature of x0*x1 (or an odd-odd function of it).
- Saturating activations (sigmoid/tanh) under-shoot at the domain corners; expect the largest
  error of the clean formula there, and compare it to the snap tolerance (2% of output range)
  rather than trying to model the under-shoot.
- Try the cleanest formula first with the checker; only add correction terms if it fails.
