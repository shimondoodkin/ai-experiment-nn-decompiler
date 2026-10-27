# Changes

Replaced the 16+14-neuron ReLU network with the rule it was meant to compute:

    return max(min(x0, x1), min(max(x0, x1), x2))   # median(x0, x1, x2)

## Why

- In `repair.csv` the `label` column equals median(x0, x1, x2) exactly on all 2000 rows.
- The network computes roughly median + 1.3 inside one axis-aligned box
  (x0 in (-0.95, 1.9), x1 in (-2.15, 0.65), x2 in (-2.95, -0.05)). 218 of its 402 failures
  lie in that box (none of its correct answers do), and the offset is spurious there.
- The other 184 failures come from the network fitting the median only to about +/-0.15
  (worse near the domain corners and on the box's ramp edges). That is more than the 0.119
  scoring tolerance.
- Writing the median directly removes the planted offset and the approximation error together,
  with no special cases.

## Result (`python score.py agent.py`)

2000/2000 right; 402/402 network failures fixed; 0 of the 1598 correct answers broken.
