# Notes

## What I changed and why
- The network is one ReLU neuron followed by an identity output (weight 1, bias 0), so the output
  layer was dropped.
- All 8 input weights are identical (0.02 = 1/50, verified exactly in weights.npz), so the weighted
  sum factors into `sum(x0..x7) / 50`.
- The bias -20 is a threshold of 20 * 50 = 1000 on the raw total. So
  `relu(0.02*total - 20) == relu(total - 1000) / 50`. The output is "how far the sum of the 8 bytes
  goes above 1000, scaled by 1/50" (the maximum total is 2040, so the output ranges 0..20.8).
- Result: `PASS (exact)` (max error ~5e-6, float32 rounding).

## Heuristics
- Read the raw weights file, not only the dumped code, to confirm whether coefficients are exactly
  "nice" (e.g. 0.02 = 1/50) or just printed rounded.
- Identity layers (weight 1, bias 0) and single-neuron chains collapse; fold them away.
- When all weights of a neuron are equal, factor them out: `w * sum(inputs) + b`.
- Rewrite `relu(w*s + b)` with w > 0 as `relu(s + b/w) * w` so the threshold `-b/w` is expressed
  in input units (here 1000) and becomes interpretable; snap that threshold to a round number or
  a half-step between integers.
- Prefer dividing by an integer constant (/50) over multiplying by a decimal (0.02) for exactness
  and readability.
- Compute the reachable input range (here total in [0, 2040]) to check whether a ReLU is ever
  inactive/always active; if one side is unreachable, the ReLU can be removed.
