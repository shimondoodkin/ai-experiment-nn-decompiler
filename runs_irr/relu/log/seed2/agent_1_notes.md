# Notes

## What the network is

1 input (x0 in [0.1, 5]), 8 ReLU units, regression output.

- **Dead units:** h1_1 and h1_3 have negative slope and negative bias, so they are 0 for every x0 > 0. Removed.
- **Always-on units:** h1_0, h1_2 and h1_5 have kinks at x0 = 0.0980, 0.0965 and 0.1001, at or below the
  lower bound 0.1. On the domain they are plain linear terms and fold into one line.
- **Real kinks:** three units bend inside the domain: h1_7 at x0 = 0.373, h1_6 at 1.056 and h1_4 at 2.448.
  The slopes go 4.55, 1.48, 0.59, 0.28, always decreasing. So the function is concave and piecewise linear,
  and it equals the **min of four lines**. No relu is needed.
- **Shape:** the values follow log(x0) closely (-0.57 vs -0.53 at 0.59, 1.11 vs 1.11 at 3.04, 1.66 vs 1.61 at 5).
  At the low end the net's value is -2.14 and log gives -2.30. That gap is above the snap tolerance (0.076), so
  plain `log(x0)` does not pass. I kept the piecewise form.

## What I changed

- Folded the three always-on units and the bias into each segment line. I computed each segment's exact
  line from the net's values at its ends.
- Wrote the net as `min(steep, mid, gentle, flat)`.
- Snapped the coefficients to one decimal place, or two for the flat piece: 4.5x-2.6, 1.5x-1.45, 0.6x-0.5,
  0.28x+0.27. The max error is 0.029, against a tolerance of 0.076, so it gives **PASS (snap)**.
- Coarser snaps fail or come close to failing. For example, 0.25x+0.4 for the last piece gives an error of
  0.086. Three lines give an error of at least about 0.22, so all four pieces are needed.
- There is also an exact version, the same min with unrounded coefficients, in `scratch/exact.py`. It gives
  **PASS (exact)** (max error 1.6e-5). I submitted the snapped one because it reads better.

## Heuristics

- **Classify each first-layer ReLU unit by where its kink (-b/w) lies relative to the input box.**
  - If the unit is zero everywhere on the box, it is dead: drop it.
  - If it is positive everywhere on the box, it is linear: fold it into the affine part.
  - Only units whose kink falls inside the box shape the function.
  - Kinks within about 1% of a bound behave like linear units.
- **For a 1-input ReLU net, list the segments (kinks sorted, slope per segment) and look at the slope signs.**
  - If the slopes only decrease, the function is concave: `min(line_1, ..., line_k)`.
  - If they only increase, it is convex: `max(...)`.
  - Either form is shorter and clearer than a sum of relus.
- **Compare the curve with familiar functions** (log, sqrt, 1/x, x², abs). Accept one only if its max error
  stays under the snap tolerance across the whole domain. Smooth targets often fail near the domain edge,
  where the net has few kinks: here log failed at x = 0.1. When that happens, keep the piecewise form but
  name the shape in a comment.
- **Snap each line's slope and intercept separately, then measure the max error on a dense grid** before
  running the checker. Spend the error budget on readability (one decimal place) rather than on removing
  segments. Try k-1 lines first to see whether a segment can go.
- **Fit the segment lines from net values just inside each kink**, not by adding up weight products. It is
  less error-prone, and the result is the exact line.
