# Notes

## What the network computes
A 2-input regression net (16 ReLU units) on [-3, 3]^2. Sampling it shows one bump at the origin:
about 0.95 at (0, 0), about 0.37 at distance 1 along either axis, about 0.16 at (1, 1), and near 0
(ripples of about +-0.1) beyond distance 2. It looks like exp(-(x0^2 + x1^2)), but that formula is
0.13 away from the net at worst (mean 0.035), and the snap tolerance is 0.022, so a closed formula
cannot pass. The net's own ripples in the tails are bigger than the tolerance.

## What I changed
1. Removed unit 5: its pre-activation is below 0 on the whole box (0.017*3 + 0.031*3 - 0.346 < 0),
   so it is always 0. Removing only this unit gives PASS (exact).
2. Removed unit 15 (output weight -0.04, contribution at most 0.06): after refitting the output layer
   the error is 0.0025.
3. Normalised each remaining unit so its largest input coefficient is +-1 (the scale goes into the
   output weight).
4. Snapped the coefficients and biases one at a time, greedily, to the nicest nearby value
   (0, then multiples of 1, 0.5, 0.25, 0.1, 0.05). After each try I refit the output weights and bias
   by a minimax LP on a grid plus random points, and kept the snap if the max error stayed under a
   budget (0.012 first, then 0.016). Then I rounded the output weights to 3 decimals.
5. Grouped the 14 ramps by where their crease lines are: 8 lines pass through or next to the origin
   (they shape the peak, negative weights) and 6 lines at about distance 1.25 (they bend the slopes
   back to flat tails, positive weights).

Result: 14 ReLU units, PASS (snap), max |program - net| = 0.0173 (tolerance 0.0220).

What did not work:
- Removing more units with only an output refit: 13 units give 0.024 at best (LP), 12 give 0.04.
- Retraining smaller nets on the teacher (all parameters free, L_p loss with p rising to 16-32):
  8 units 0.18, 10 units 0.065, 12 units 0.041, 13 units 0.020 (too close to the limit, and with
  unround weights). So 14 units is about the minimum for this tolerance.
- Setting the small 0.02 x0-tilts of the two x1 rim lines to 0 (which would allow
  relu(abs(x1) - 1.25)) pushes the error to 0.025-0.026.

## Heuristics
- Before anything else, find dead units by interval arithmetic over the input box: a unit with
  max(w.x + b) <= 0 on the box is always 0 and can be removed exactly.
- Sample the net on a grid and print it. It shows the shape (here a bump) and whether a closed
  formula is plausible. Compare the best formula with the net under the checker's tolerance before
  spending effort on it: if the net's own ripples are bigger than the tolerance, keep a piecewise
  linear program.
- Prune by greedy backward elimination with an output-layer refit. Use a minimax (LP) refit, not
  least squares, because the checker judges the maximum error. The LP gives a lower max error for
  the same set of units.
- Normalise each ReLU unit so its largest coefficient is +-1 (relu(s*z) = s*relu(z) for s > 0).
  This makes the coefficients comparable and easy to snap.
- Snap greedily, one parameter at a time, nicest candidate first, refit the outputs after each
  snap, and accept under an error budget below the tolerance (about 70% of it), so the boundary
  points the checker adds do not push the error over.
- When a structural search gets stuck, test how far a full retrain of k units on the teacher gets;
  it gives a lower bound on the number of units needed and tells you when to stop pruning.
- On Windows with many jobs running, set torch.set_num_threads(1); with the default the small
  training loops were 100 times slower.
- Group ReLU units by where their crease lines lie (through the centre vs at a rim), and name them
  after that; it makes a sum of ramps readable.
