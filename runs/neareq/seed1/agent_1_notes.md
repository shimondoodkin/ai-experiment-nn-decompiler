# Notes

## What the network does
Class 1 iff |x0 - x1| < 0.2, i.e. the two inputs are almost equal. The output is `0.2 - abs(x0 - x1)`.

## How I got there
- Every live hidden neuron has weights of about (+a, -a) or (-a, +a), so each one depends only on d = x0 - x1.
- Neurons 0, 4, 6 are about a*relu(d - 0.1), with a around 3.3 to 3.6. Neurons 1, 2, 3 are about a*relu(-d - 0.1).
  All six use the same hinge at |d| = 0.1, so they act as one hinge per side.
- Neuron 5 (0.55*d + 2.68) is always positive on the input box (|d| <= 4 gives at least 0.48), so it is linear.
  It adds to the bias: about 8.56 + 1.44*d.
- Neuron 7 is always 0 on the box (its maximum is about -0.07), so it is dead and I removed it.
- Collapsed output: 8.56 + 1.44d - 90.3*relu(d - 0.1) - 81.4*relu(-d - 0.1).
  Its zeros are at d = +0.198 and d = -0.202, so I snapped both to |d| = 0.2. The slight
  asymmetry is training noise, and `0.2 - |d|` has the same sign everywhere.
- Only the sign matters for a binary output, so I could replace the steep piecewise-linear
  score with the simple `0.2 - |d|`.
- Result: `PASS (snap)`. The only disagreements are points exactly on |d| = 0.2. There the net is
  unsure, and its own boundary there is not symmetric in x0 and x1, so no simple formula could
  match it exactly.

## Heuristics
- Look for neurons whose weights are opposite in sign and nearly equal in size. They read a difference (x0 - x1), so rewrite them with a new variable `d`.
- Divide each neuron by its weight size to get relu(d - t). Neurons with almost the same t and same direction can be merged into one hinge, with their output weights times their sizes added together.
- Use the input bounds to check each neuron's preactivation range. If it is always positive, the neuron is linear: fold it into the bias and the linear terms. If it is always negative, it is dead: remove it.
- For a binary output, solve for where the collapsed score is zero, then snap those crossing points (here +-0.198 and -0.202 become +-0.2). Then write the simplest score with the same sign, for example `t - abs(d)` for a band.
- Two hinge groups that mirror each other around 0, with a positive bias, usually mean a band `|d| < t` (an "almost equal" test).
- Disagreements exactly on the snapped boundary are expected. Accept them when the checker says they are in the unsure zone.
