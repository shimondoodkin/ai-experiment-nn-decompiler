# Notes

## What I changed

- The dump is a 1 -> 4 -> 1 tanh net with a binary output (class 1 if score > 0).
- All first-layer slopes are tiny (|w| <= 0.017), so over x0 in [-2, 2] each tanh unit
  barely moves: the pre-activations change by at most 0.07. Unit 0 is effectively zero
  (weight 1e-6 in, 2e-7 out).
- I scanned the score on a dense grid (400k points over [-2, 2]): it stays in
  [-1.088, -1.074]. It never comes close to the threshold 0, so the net predicts class 0
  everywhere.
- `agent.py` returns the constant -1 (a negative score = class 0). The checker gives
  PASS (exact) on 205k inputs with no disagreements.

## Heuristics

- Before refactoring, bound the output range over the input box (dense scan for 1-D, or
  interval arithmetic: |w*x| <= |w|*max|x| per unit). If the whole range sits on one side
  of the threshold, the binary net is a constant and the program is `return <constant>`.
- Units whose input weights are tiny relative to the input range (|w| * range << 1) are
  constants: replace tanh(w*x + b) by tanh(b) and fold it into the bias.
- Units with tiny output weights (~1e-7) can be dropped outright.
- A degenerate (collapsed) net, with all weights near zero except biases, usually means
  training failed or the data was one class; check this first, since it ends the work.
