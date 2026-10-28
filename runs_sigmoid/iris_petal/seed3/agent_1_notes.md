# Notes

## What I did

- The dump is a 2 -> 8 sigmoid -> 3 argmax net. Reading the weights alone is not helpful, so I
  evaluated the net on a grid over the input box and printed the class map as text.
- The map has three classes in three bands: class 0 in the low-x0 / low-x1 corner, class 2 in the
  high-x0 / high-x1 corner, class 1 between them. The two boundaries are almost straight lines and
  never meet inside the box, so the program is two cuts in a chain (no 0/2 boundary).
- I located each boundary by scanning x0 finely for each x1 (25 rows) and fit a line x0 = c - a*x1:
  - 0|1: x0 = 5.47 - 4.21*x1 (slightly curved near x1 = 1)
  - 1|2: x0 = 6.74 - 1.10*x1
- I snapped them to x0 + 4.25*x1 < 5.5 -> 0 and x0 + 1.1*x1 > 6.75 -> 2. I compared a few snapped
  variants on 400k random points by mismatch rate per boundary. For 0|1, (4.25, 5.5) gave 0.09%
  vs 0.23% for (4.2, 5.5) and 0.45% for (4, 5.4). For 1|2, (1.1, 6.75) gave 0.20% vs 1.2% for (1, 6.6).
- Result: `PASS (snap)`, 0 disagreements where the net is confident; all the disagreements are in the
  unsure zone near the slightly curved boundaries. An exact pass would need the sigmoids back,
  because the true boundaries curve a little.

## Heuristics

- For low-dimensional argmax nets (especially sigmoid ones), print the class map as ASCII on a grid
  first; it shows the region structure (bands, corners, which classes touch) at once.
- If the class regions are ordered bands, write the program as a chain of cuts (`if s1 < t1: return 0;
  if s2 > t2: return 2; return 1`). You only need one cut per adjacent pair, not a full argmax.
- Locate each boundary with a fine 1D scan per row, fit a line, and normalise the coefficient of one
  input to 1 (x0 + a*x1 vs t). Small coefficients and thresholds are then easy to snap.
- Snap one boundary at a time and measure its mismatch rate separately on random points. Prefer the
  simplest value whose mismatch is close to the best fit. Quarter steps (4.25) can beat a nearby
  integer by several times.
- Sigmoid nets give smoothly curved boundaries, so straight cuts usually reach only a snap pass. The
  curvature is largest where several hidden units switch at once, which is where the remaining
  mismatches are.
