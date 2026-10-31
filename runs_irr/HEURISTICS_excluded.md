# Excluded agent heuristics

Agent runs left out of distillation and reproduction because blindness broke. Kept here for the record; tool development does not read this file.

- relu_wide/bump/seed1 (macb4a7): a process listing it ran (PowerShell, while debugging its own scripts) showed the command lines of the irrational rerun batch, which carry every run path with its task name.

## bump/seed1 — agent_1 (macb4a7, 2026-10-31)

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

