# Notes: m97e748 (1 input -> 8 ReLU -> 1 regression output, x0 in [0, 200])

## What I changed and why

1. **Dead neurons removed.** Neurons 0, 2, 3, 4, 6 have a negative slope *and* a negative bias.
   On x0 >= 0 their pre-activation is always < 0, so they are identically 0 on the whole domain
   and contribute nothing. Only neurons 1, 5, 7 survive.
2. **Each live unit rewritten as `slope * relu(x0 - kink)`.** For a 1-D ReLU unit
   `v * relu(w*x + b)` with w > 0 this equals `(v*w) * relu(x - (-b/w))`. This gives
   - kink 3.3917, added slope 0.15025 (neuron 7)
   - kink 22.855, added slope 0.03975 (neuron 1)
   - kink 63.398, added slope 0.19230 (neuron 5)
   - output bias -0.1548 (the flat value below the first kink).
   An exact-level rewrite with these numbers passes `PASS (exact)` (max err 1.8e-5; kept in
   `scratch/agent_exact.py`).
3. **Snapping.** Snap tolerance is 2% of the output range (~1.257). I snapped
   kinks to half-steps (3.5, 23, 63.5), slope increments to 0.15 / 0.04 / 0.19 (so marginal
   rates 0.15 -> 0.19 -> 0.38, the last one exactly doubling), and dropped the -0.155 bias as
   "zero". Max error 0.205 (at x0 = 200, where the slope error accumulates), i.e. 16% of the
   tolerance. Coarser snaps (0.15/0.2/0.4 slopes, kink 64) blew past 1.0 at x0 = 200, so I kept
   two-decimal rates. `python check.py agent.py` -> `PASS (snap)`.
4. **Interpretation.** The function is a convex, monotone, piecewise-linear "bracket" curve
   (zero below ~3.5, then marginal rates 15% / 19% / 38%), like a progressive tax/tariff
   schedule. It is also a reasonable 3-hinge fit of a smooth convex curve
   (roughly x^1.5/45), so the original target may have been smooth; the language cannot express
   that, so the hinge form is the honest description.

## Heuristics

- For a single-input layer, a unit is dead on [lo, hi] iff `w*x + b <= 0` at both endpoints;
  check endpoints instead of sampling. Same sign of slope and bias with x >= 0 is a quick tell.
- For 1-D input, rewrite each live first-layer unit as `(v*w) * relu(x - kink)` (or
  `relu(kink - x)` for w < 0); the network becomes a sum of hinges, i.e. a piecewise-linear
  function you can read as "starting value + slope changes at kinks".
- List kinks in sorted order and accumulate slopes: cumulative (marginal) slopes are usually
  more interpretable than the per-unit increments (here 0.15, 0.19, 0.38).
- Compute the snap budget first (here 2% of output range) and spend it where errors accumulate:
  slope errors grow linearly with distance from the kink, so the far end of the domain
  (x0 = 200) dominates; kink shifts cost `slope * shift`. Evaluate candidates on a dense grid
  before running the checker.
- A small output offset (here -0.155 vs. range 63) next to a flat region is usually bias
  leakage for "0"; drop it if the budget allows.
- Keep an exact (unsnapped but simplified) version as a reference: it confirms the structural
  rewrite is right before any snapping, separating "wrong structure" from "too much snapping".
