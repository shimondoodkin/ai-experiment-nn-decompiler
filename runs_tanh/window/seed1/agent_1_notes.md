# Notes

## What the network does
One input x0 in [-2, 2], four tanh hidden units, binary output (class 1 if score > 0).

All four hidden units have tiny input weights (|w| <= 0.019) and moderate biases, so over
[-2, 2] each tanh moves by at most ~0.03. The score is almost a constant:
score(x0) ~= -1.124 - 0.0118 * x0, ranging from -1.101 (x0 = -2) to -1.147 (x0 = 2).
It never comes near the threshold 0, so the classifier is constant: always class 0.

## What I changed
- Replaced the whole network with `return -1` (a negative constant score = always class 0).
  I used -1 instead of 0 so the score is not sitting exactly on the threshold.
- check.py: PASS (exact), 0 disagreements over 200203 inputs.

## Heuristics
- Before simplifying units one by one, evaluate the output over the input range (endpoints and a
  coarse grid). If its sign never changes, the binary net is a constant classifier.
- Bound each unit's pre-activation over the input box: |w| * (hi - lo) / 2 tells how much a unit can
  move. When all weights are tiny relative to the domain, each unit is ~a constant
  (tanh(b) or sigmoid(b)), and the output reduces to a constant plus a tiny linear term.
- For a constant binary output, return a constant clearly on one side of the threshold (e.g. -1),
  not the threshold itself, so the decision is robust.
- A net that learned a constant usually means the training signal was dominated by one class in this
  domain (or the true boundary lies outside it); don't invent a boundary the net doesn't have.
