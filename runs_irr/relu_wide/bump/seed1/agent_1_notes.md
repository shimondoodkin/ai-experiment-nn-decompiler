# Notes

## What the network computes

A 2 -> 64 -> 1 ReLU regression net on x0, x1 in [-3, 3]. Tabulating it on a grid shows a single
round bump at the origin: about 1.03 at (0, 0), about 0.39 at distance 1, about 0.02 elsewhere.
The closest simple formula is `exp(-(x0*x0 + x1*x1)) + 0.01`. Its max error against the net is about
0.035–0.047, and the snap tolerance is 0.0208 (2% of the output range of about 1.04), so the formula
alone FAILs. The leftover error has no structure: the net is up to 0.05 high along the box edges and
corners, and it has ReLU-kink ripples of ±0.02 near the peak. A Gaussian feature plus a few of the
net's own units did not help either. Greedy selection with exp(-r²) as a feature stalled at about
0.024 minimax.

## What I changed

1. Removed 10 dead units. Seven had weights around 1e-6 or smaller. Units 18 and 41 are off
   everywhere on the box (checked at the 4 corners, since a unit is linear before the ReLU). Unit 37
   contributes less than 1e-9. On its own, this plus 3-decimal rounding passes (54 units, max error
   0.0045).
2. Ran backward elimination over the 54 live units. At each step it drops the unit whose removal,
   after a least-squares refit of the output layer (with free constant, x0 and x1 terms), gives the
   smallest max error. This kept 35 units.
3. Rounded the hidden weights to 3 decimals, then refit the output weights and the linear term with
   a minimax LP on a 101x101 grid plus random points. I rounded those to 4 decimals, because
   rounding output weights to 3 decimals added about 0.005 of error across 35 terms.
4. Result: `agent.py` PASS (snap), max |program - net| = 0.0171 against a tolerance of 0.0208,
   35 units instead of 64.

I also tried fine-tuning all weights of smaller subsets (Adam with an L^p loss, p rising to 32). This
gave no real gain: 32 units reached 0.0175, 28 units 0.0207 (too close to the tolerance) and
24 units 0.030. 35 units with the LP refit kept the best margin.

## Heuristics

- Tabulate the net on a coarse grid first. A radially symmetric peak means a Gaussian or bump target;
  try `exp(-r²)` and `exp(-r²) + c` and measure the max error against the snap tolerance.
- When a smooth target is approximated by many ReLU ridges, the net's own approximation error
  (here about 2x the tolerance) decides whether a closed formula can pass. If it cannot, simplify the
  ReLU program instead of forcing the formula, and record the near formula in a comment.
- Find dead units by checking the pre-activation at the corners of the input box: it is affine, so
  its max and min over the box are at corners.
- Backward elimination with an output-layer refit (least squares to choose, minimax LP for the final
  fit), plus free constant and linear terms, removes many more units than magnitude pruning does.
- Round hidden weights first, then refit the output layer on the rounded features. Output weights
  multiply features of size about 3, so they need one more decimal than the hidden weights.
- Python puts the script's folder first on sys.path. A scratch file named after a stdlib module
  (here `select.py`) is silently imported by numpy, scipy or torch and runs instead. Avoid stdlib names
  for scratch scripts.
