# Notes

## What the network computes

The inputs are two booleans, so there are only 4 possible inputs. Evaluating the dump on each of them
(approximate output values):

| x0 | x1 | output | class |
|----|----|--------|-------|
| 0  | 0  | about -11.4 | 0 |
| 1  | 0  | about +12.0 | 1 |
| 0  | 1  | about +12.3 | 1 |
| 1  | 1  | about +23.2 | 1 |

That is logical **OR**. The final program is `return x0 + x1 - 0.5` (positive iff x0 or x1 is set),
and the check gives `PASS (exact)` on all 4 inputs.

## What I changed and why

- **h1_0 and h1_2 are dead.** Their pre-activations are negative at every corner of the boolean
  domain: h1_0 = relu(-0.51 x0 - 0.17 x1 - 0.61) is always below 0, and h1_2 peaks at
  -0.024 when x0=0, x1=1. Their output weights are tiny too (-0.03, -0.21). Removed.
- **h1_1 = relu(3.25 (1 - x0 - x1))** is a "NOR" detector: 3.25 at (0,0), and 0 elsewhere (at the
  other corners it is 0 or a rounding residue of about 1e-5). The output weight is -3.88, so it pushes
  (0,0) down by about 12.6.
- **h1_3 = relu(3.66 x0 + 3.75 x1 + 0.15)** is always active on the domain, so it acts linearly as
  "count of set inputs", about 3.7 (x0 + x1). The output weight is +2.97.
- Together: out ~= 11 (x0 + x1) - 12.6 [x0 = x1 = 0] + 1.2. Only the sign matters for a binary
  output, so I dropped the scale and the redundant NOR gate. With integer inputs, the half-step
  threshold 0.5 between "0 set" and "1 set" reproduces the decision exactly.

## Heuristics

- With small discrete (boolean or integer) domains, enumerate the full truth table first; the
  decision function is usually a recognizable logic or counting rule, and that beats snapping
  coefficients one by one.
- A ReLU unit whose pre-activation is negative at every point of the (bounded) domain is dead;
  remove it. Check the maximum over the box corners for linear pre-activations.
- A ReLU unit whose pre-activation is positive on the whole domain is linear; drop the relu and fold
  it into the affine output.
- Near-equal weights on several inputs (3.66 and 3.75, or 3.2532 / 3.2532 / 3.2532 bias) mean the unit
  depends only on the sum of those inputs; a bias equal to minus the weight means "fires only when all
  are 0" (NOR-like), and a bias near 0 with positive weights means "count".
- Rounding residues (such as 7e-6 from 3.253171 - 3.253164) are noise; treat them as exact zeros.
- For binary or argmax outputs only the sign or ranking matters: rescale freely, and put a
  threshold on integer-valued sums at a half step (k - 0.5) to get the most robust margin.
- Redundant units (one unit's job already done by another) can be dropped when the truth table is
  unchanged; for example the NOR gate here only widens the margin of a decision the count unit already
  makes.
