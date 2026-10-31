# Notes

## What the network computes

Regression, inputs x0, x1 in [-3, 3]. A grid scan shows a smoothed cone: about
sqrt(x0^2 + x1^2), with 0.33 at the origin, about 3 on the axes at the box edge and 4.15-4.34 in the
corners. So the net is a smoothed Euclidean distance from the origin (|x|).

## What I tried

1. **Closed formula.** The best minimax fit of `s*sqrt(a*x0^2 + b*x1^2 + e*x0*x1 + c) + d + linear`
   has max error 0.080 against the net, equal to the snap tolerance (0.080 = 2% of the range 4.01).
   `sqrt(x0^2 + x1^2 + 0.08)` has max error 0.11. The net's own ripple (the gap between a sum of 16
   sigmoids and a true cone) is about +-0.1, so no clean formula passes. Adding a few of the net's
   units as corrections to the sqrt base lowered the error only to about 0.06, not worth it.
2. **Pruned and snapped net (final).**
   - Units 5, 6, 10 and 15 have small input weights (nearly linear over the box) and small output
     weights; a greedy minimax (LP) selection left them out first. Unit 13 could also go: with 11 units
     (after snapping) the refit error was 0.036, against 0.019 with 12.
   - Input weights and biases are snapped to one decimal, the output weights are refit by minimax
     (linear program over a 61x61 grid) and rounded to 2 decimals. The linear x0/x1 terms the refit
     gave were tiny (0.03, 0.02); refitting without them was slightly better (0.0345), so they are gone.
   - The two units with negative output weights were flipped with `-a*sigmoid(z) = a*sigmoid(-z) - a`,
     so every term is a positive-weight, outward-facing step and the constants fold into one -0.73.
   - The units fall into two rings by offset b/|w|: an inner ring (about 0.65-0.9, 7 units) and an
     outer ring (about 2.3-2.7, 4 units), named by compass direction.

Result: `PASS (snap)`, max |program - net| = 0.0345, tolerance 0.0802 (290k inputs). 11 sigmoids
instead of 16, all coefficients with 1-2 decimals. Not exact: the net's ripple is part of its output,
and dropping units changes it.

## Heuristics

- Scan the output on a coarse grid first; a radial or other symmetric shape names the function
  (here a distance |x|) even when the program has to stay a sum of units.
- Compare the best closed-formula error with the snap tolerance before investing in it. A smooth
  sigmoid net approximating a cone has ripple of a few percent of the range; if a minimax fit of the
  formula family sits at the tolerance, the formula will not pass robustly.
- Sigmoid units with small input weights over the whole box are almost linear; they are the first
  to prune (their effect folds into a constant/linear term, which may then turn out unnecessary).
- Select units greedily by minimax error (LP refit of the output layer), not least squares: the
  checker judges max error, and least-squares greedy picks poorly.
- After pruning, snap input weights and biases first (one decimal kept error well under tolerance
  here; 0.25 or 0.5 steps did not), then refit the output weights by minimax and round them last.
- `-a*sigmoid(z) = a*sigmoid(-z) - a` turns negative-weight units into positive outward steps; with
  directions atan2(w1, w0) and offsets b/|w|, the units of a radial function group into rings.
- Keep a margin: aim for about half the snap tolerance on a dense grid, since the checker also probes
  boundary and corner points.
