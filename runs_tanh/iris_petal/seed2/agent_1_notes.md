# Notes

## What the network does

The net is 2 inputs -> 8 tanh units -> 3 argmax scores. Its decision map over the input box
(x0 in [1, 6.9], x1 in [0.1, 2.5]) has three regions in diagonal bands:

- class 0 in the low corner (small x0 and small x1);
- class 1 in the middle band;
- class 2 in the high corner (large x0 + x1).

Classes 0 and 2 never touch inside the box, so the program is two tests in a row.

## What I changed

1. **Looked at the decision map, not the units.** I printed the argmax class on a 60x25 grid. That
   showed two monotone boundaries, which makes the 8 tanh units unnecessary.
2. **Measured the boundaries.** For each x1 I root-solved where the pairwise score difference
   crosses 0. I also measured the width of the checker's unsure zone around each crossing (top-2
   margin < 5% of p95, about 0.485 here), which is how far a snapped boundary may move.
3. **1|2 boundary.** It is almost a line, x0 ~ 6.66 - 1.04*x1, with a slight bend at low x1. The
   unsure band is about +/-0.07 wide. `x0 + x1 > 6.6` stays inside the band at every x1, and so does
   the slope-1 diagonal. 6.55 and 6.5 fail, so 6.6 is the snapped constant.
4. **0|1 boundary.** It is clearly curved: the slope dx0/dx1 goes from about -1.8 at low x1 to
   about -1.05 at high x1. A single line does fail; I tried `x0 + 1.5*x1 < 3.7`, which gave 1631
   confident disagreements. A quadratic in x1 with simple coefficients,
   `x0 < 3.9 - 2.2*x1 + x1^2/3`, follows the measured crossings to within about 0.03, well inside
   the band.
5. Result: `PASS (snap)`, 0 confident disagreements. All differences are in the unsure zone, and
   most of them are along the 1|2 diagonal.

## Heuristics

- For low-dimensional argmax nets, start from the decision map (class on a grid) rather than the
  hidden units. Many units are only there to bend one boundary slightly, and the readable program
  is a short if-chain over the boundaries.
- Find the order in which the regions are arranged (bands, corners) so that the if-chain can test
  one boundary at a time. Check that the non-adjacent classes really never meet inside the box.
- Measure each boundary by root-solving pairwise score differences (score_a - score_b = 0, with the
  third score below both) along one input. Also measure the checker's unsure-zone band around each
  crossing: that band is the snap budget for the boundary's position.
- Try a nice-coefficient line first (slope +/-1, simple fractions). If the boundary's slope changes
  by more than the band allows, express it as x0 = quadratic(x1) with simple coefficients rather
  than stacking line pieces. One product term is shorter than a min/max of 2-3 lines.
- Sweep the snapped constant a little (for example 6.5 / 6.55 / 6.6) and keep the simplest value
  that passes with 0 confident disagreements.
- Nets trained on clustered data extrapolate their boundaries smoothly into empty corners of the
  input box. The program has to follow the net there too, not the data's likely structure.
