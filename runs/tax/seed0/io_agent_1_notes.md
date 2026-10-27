# Notes

## What the function is
One input x0 in [0, 200], regression output. Sorting the samples by x0 and looking at finite-difference
slopes shows a convex piecewise-linear curve with four pieces (least-squares fits, residuals < 1e-5):

| range of x0     | fitted piece             |
|-----------------|--------------------------|
| < ~2.86         | 0.06627 (constant)       |
| ~2.86 .. ~21.55 | 0.12201 x - 0.28250      |
| ~21.55 .. ~79.9 | 0.19892 x - 1.94001      |
| > ~79.9         | 0.39994 x - 17.99960     |

Snapping: 0.19892x - 1.94 ~ 0.2(x - 10); 0.39994x - 18.0 ~ 0.4x - 18 = 0.2(x - 10) + 0.2(x - 80).
Both pieces meet exactly at x = 80 (value 14). The two low pieces (floor 0.066, slope 0.122) are the
net's smooth-ish approximation of the corner at x = 10 (zero tax below the allowance). So it reads as a
progressive tax: 0% up to 10, 20% from 10 to 80, 40% above 80.

## Program (agent.py)
    tax = 0.2 * relu(income - 10) + 0.2 * relu(income - 80)

`python check.py agent.py` -> PASS (snap), max error 0.94 at x0 = 10 (the rounded corner), tolerance 1.24.

For reference, `agent_exact.py` is the unsnapped max-of-four-lines form
(max(0.0662678, 0.122014x - 0.2825, 0.198915x - 1.94, 0.399938x - 17.9996)), which gives PASS (exact),
max error 1.6e-5. I kept the snapped, readable version as agent.py.

## Heuristics
- For 1-D inputs: sort by x, compute consecutive slopes, and print where the slope changes; plateaus of
  constant slope reveal linear pieces and their breakpoints. Ignore isolated one-sample slope spikes
  (two samples straddling a kink).
- Fit each piece by least squares away from kinks, then intersect neighbouring pieces to get breakpoints.
- If slopes increase monotonically the function is convex: it is max(lines) or c + sum(k_i * relu(x - t_i)).
  Express it as relu increments: slope deltas become rates, the zero-crossings of the lines become
  thresholds (here 0.2x-2 crosses 0 at x=10 -> relu(x-10)).
- Write each piece as slope*(x - root) and snap the root and slope to round numbers; check that adjacent
  snapped pieces still meet at a round breakpoint (here both give 14 at x = 80).
- Short pieces with odd slopes near a corner, and small nonzero floors, are usually the network rounding
  a sharp kink (e.g. a 0 -> 20% step at 10); replace them with a single relu at the round threshold and
  check the error stays within snap tolerance.
- Piecewise-linear with increasing marginal rates on a money-like input suggests tax brackets / tiered
  pricing; name variables accordingly (income, tax).
- Keep an exact max-of-lines version as a fallback to confirm the piece fits are right (PASS exact).
