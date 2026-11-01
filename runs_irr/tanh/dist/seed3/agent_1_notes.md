# Notes

## What the network does

A 2 -> 16 tanh -> 1 regression net on x0, x1 in [-3, 3]. On a grid its output looks like the
Euclidean distance from the origin, `sqrt(x0*x0 + x1*x1)`: about 4.2 at the corners, about 3 at
(±3, 0) and (0, ±3), and lowest near (0, 0). It smooths the cone tip, so it gives about 0.46 at the
origin, and it wiggles by up to about 0.23 at the corners. A closed form cannot pass: the best
`a*sqrt(x0^2 + x1^2 + c) + d` (and variants with a shifted centre or an elliptic quadratic form)
is about 0.25 from the net at its worst, while the snap tolerance is 0.075 (2% of the output range
0.45..4.22). So `agent.py` stays a small tanh network, with the distance reading in a comment.

## What I changed

1. **Nine near-flat units folded.** Units 0, 3, 4, 7, 11, 12, 13, 14 and 15 have input weights of
   about 0.02-0.035 and biases of about ±1, so over the box their pre-activation moves by less
   than 0.2 and tanh is almost linear there. Their sum is a constant plus a slope
   (5.79 + 0.059*x0 + 0.094*x1, max error 0.05), or exact to 0.002 with a quadratic. They all
   point the same way, (0.6, 1)-ish, which is why they merge.
2. **Seven real units kept.** Dropping any of the remaining seven (1, 2, 5, 6, 8, 9, 10) and
   refitting everything else costs at least 0.055 even with free hidden weights, so all seven stay.
   With their original weights plus a constant and a linear term, the max error was 0.014.
3. **Refit, then greedy snapping.** I refitted the hidden weights and biases of the seven units on
   a grid (output layer solved by least squares each time). That brought the max error down to 0.003.
   Then I snapped one parameter at a time to the 0.1 grid (then to 0.05), most-round first,
   re-optimising the free ones after each snap, and kept a snap only if the max error stayed
   below 0.035. Every hidden parameter snapped except one weight (0.44). Two units then had an
   x1 weight of exactly 0: `tanh(0.5*x0 + 0.7)` and `tanh(0.5*x0 - 0.9)`, a pair of x0 steps
   that together make the |x0|-like valley.
4. **Output layer snapped.** The x0 slope could be dropped (error 0.030). Dropping the x1 slope
   could not (0.06). Then I snapped the output coefficients to 0.1/0.05 the same greedy way:
   1, 1.5, 2.3, 0.45, -0.6, 1.65. The last one, -1.51, I left at two decimals, because -1.5 pushed
   the error to 0.045.
5. **Names.** The units are named by the side of the box where they raise the output
   (`down`, `down_far`, `up`, `diag_pp`, `diag_mp`, `left`/`right`).

Result: 16 units -> 7. `python check.py agent.py` gives PASS (snap), max |program - net| = 0.038
(tolerance 0.075), 265k inputs.

## Heuristics

- **Look at the output on a coarse grid first** and compare with simple shapes (distance, product,
  sum of abs). For regression nets the snap tolerance (2% of the range) is often smaller than the
  net's own error against the obvious formula. Measure that gap before spending time on a closed
  form; if it is bigger than the tolerance, keep a reduced network and put the formula in a comment.
- **Near-flat units:** if |w|·(input half-range) is small (pre-activation moves < ~0.2), the unit is
  linear over the box. Fold all such units into a constant plus a linear term (add a quadratic if
  the linear error is too big). Many of them often share one direction, which shows the net used them
  as a cheap slope or bias.
- **Prune with a refit, not by deletion:** test each unit's removal by refitting the output layer
  (and the hidden weights, if needed) and measuring the max error. A unit is necessary only if the
  refit cannot cover it.
- **Refit before snapping, then snap greedily:** a full refit of the kept units against the net
  gives headroom (here 0.014 -> 0.003). Then snap parameters one at a time, nearest-to-round first,
  re-optimising the rest after each snap, and accept a snap only while the max error stays under
  about half the tolerance. Snapping all at once on a 0.1 grid without re-optimising used most of
  the tolerance (0.067 of 0.075).
- **Solve the output layer by least squares** inside the hidden-weight optimisation (variable
  projection). It is fast and stable for these small nets.
- **Keep about half the tolerance as headroom**, because the checker also probes boundary and
  bisected points that a grid can miss.
- **Paired tanh steps on one input** (same weight, opposite-sign output coefficients, biases on
  either side of 0) are a smooth `|x|` or a bump. Name them as such.
