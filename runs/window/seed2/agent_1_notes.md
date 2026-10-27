# Notes

## What I changed and why

- Neuron h1_3 = relu(-0.051*x0 - 0.808): on x0 in [-2, 2] its pre-activation is at most
  0.102 - 0.808 < 0, so it is always 0. Removed it.
- h1_0 and h1_2 have nearly the same kink (x0 = 0.4164 and 0.4207) and both feed the output with
  negative weight, so they act as one ramp "x0 > ~0.42". h1_1 is the mirror ramp "x0 < ~-0.40".
- Solved for the zero crossings of the output exactly: right edge x0 = 0.50046, left edge
  x0 = -0.50043. Between them all ramps are 0 and the output is the positive bias 5.68.
  So the net says class 1 iff -0.5 < x0 < 0.5, i.e. |x0| < 1/2.
- Final program: `return 0.5 - abs(x0)`. check.py: PASS (snap); the only disagreements are
  in a ~0.0005-wide sliver beyond +/-0.5 inside the net's unsure zone. Getting PASS (exact)
  would need the unsnapped edges 0.50046 / -0.50043, which carry no meaning.

## Heuristics

- For each ReLU, compute its pre-activation range over the input bounds; drop neurons that are
  never active (or never inactive -> they are linear and fold into the bias/other terms).
- Group neurons whose kinks (x = -b/w) are close and whose output weights share a sign; they
  behave as a single ramp.
- For binary outputs only the decision boundary matters: solve output = 0 piecewise per linear
  region rather than simplifying the score itself, then snap the boundaries.
- A positive bias plus negative-weighted ramps on both sides of 0 is a "band" detector:
  rewrite as `c - abs(x - m)` with snapped centre m and half-width c.
- Mirror-symmetric boundaries (+0.5004 / -0.5004) strongly suggest abs().
- Accept PASS (snap) rather than encoding the noisy exact boundaries.
