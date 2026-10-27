# Notes

## What I did

- `spec.json`: 2 float inputs (x0 in [1, 6.9], x1 in [0.1, 2.5]), 3 outputs, argmax. The ranges match
  Iris petal length / petal width, so classes 0/1/2 read as setosa / versicolor / virginica.
- A nearest-neighbour class map of `samples.csv` showed three diagonal bands: class 0 at small
  length+width, class 2 at large, class 1 in between. Single linear fits of the outputs were poor, so
  the outputs are piecewise linear (a ReLU net).
- Exact recovery: local least-squares fits on 7-nearest-neighbour patches gave about 10 distinct
  affine pieces. The difference between two adjacent pieces is rank-1 (one hidden unit flipping),
  and that rank-1 factor gives the unit's line and its output weights. This found 6 hidden lines.
  A joint least-squares refit (lines + output weights) reproduced all samples to about 3e-6. That
  version (`scratch/exact_net.py`) gets `PASS (exact)`.
- Readable version: since only the argmax counts, I traced the two class boundaries with the exact
  net (bisection along x0 for each x1):
  - 1|2 boundary: x0 ~ 6.72 - 1.05*x1 (slightly flatter below x1 = 0.4). Snapped to
    `x0 + x1 > 6.65` (one "size" sum). `x0 + x1 > 6.6` and `> 6.7` both fail; 6.65 is the
    half-step between them.
  - 0|1 boundary: two pieces, slope -1.32 for x1 < 1.2 and -0.89 for x1 > 1.5. One line
    `x0 + 1.25*x1 < 3.45` passes. A two-piece `x0 < max(3.5 - 4/3*x1, 2.85 - 0.9*x1)` follows the
    curve more closely (1535 vs 2371 disagreements, all in the unsure zone). I kept the single line
    because it is easier to read.
- Final `agent.py`: `PASS (snap)`, 0 disagreements where the function is confident.

## Heuristics

- For argmax nets with 2 inputs, draw a coarse class map first (1-NN on the samples). It shows the
  topology right away: bands, wedges or islands.
- Piecewise-linear check: fit local affine pieces on k-NN patches and cluster the gradients. The
  number of distinct pieces bounds the number of active ReLU units.
- Adjacent affine pieces differ by a rank-1 matrix (input-line x output-weights). An SVD of the
  difference gives each hidden unit's hyperplane directly. Refit all units together with
  least squares to reach exact precision. Hyperplanes that show up in several region pairs are the
  same unit.
- For a readable program, do not snap the net's weights one by one, because argmax errors add up.
  Find the class boundaries of the exact model (bisection along one input) and snap those lines.
- Normalize each boundary line so one input has coefficient 1 (`x0 + a*x1 > c`). Then try a = 1
  and simple fractions (1.25, 4/3) first, and move c in half-steps (6.6 -> 6.65) until the check
  passes.
- A slightly curved boundary often passes with one line whose slope sits between the two
  piece slopes. Use `max(line1, line2)` only if the single line fails.
- Watch the "confident" disagreement count, not the total. Unsure-zone disagreements are allowed,
  so pick the simplest form with 0 confident errors.
- Recognize known datasets from the input ranges (here Iris petal length/width). It suggests
  meaningful variable names and the expected class order.
