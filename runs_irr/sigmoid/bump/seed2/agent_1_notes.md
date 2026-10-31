# Notes

## What the network computes

On the grid the output is a round bump centred on the origin: 1.00 at (0, 0), 0.78 at radius 0.5,
0.37 at radius 1, 0.10 at radius 1.5, about 0 further out. That is exp(-(x0*x0 + x1*x1)).

The formula alone FAILS the check. Max |formula - net| = 0.040, and the snap tolerance is 0.0206.
The difference is the network's own ripple. It is spread over the whole square, with many
local extrema of about ±0.03, and the largest values are near the edges (x1 = 3 or x0 = -3).
Fitting a·exp(-r²) + c by minimax still leaves 0.0345. Adding up to 8 of the original sigmoid
units to the Gaussian as correction terms only got that down to 0.029. The ripple is not in the
span of a few units plus a Gaussian, so a "formula + small correction" program does not pass.

## What I did

1. Pruning. Backward elimination with a minimax (LP) refit of the output weights only: units 3
   and 14 could go (error 0.0075). Unit 3 nearly duplicates unit 5 (weights (-1.02, -0.61, -1.39)
   and (-0.87, -0.77, -1.62)).
2. A full nonlinear refit of all weights. I minimised a smooth max (log-sum-exp of |error|,
   temperature raised in steps 100 → 3000) on a 73×73 grid with L-BFGS, starting from the
   original weights. This gave 14 units at 0.0014, 13 at 0.0070 and 12 at 0.0104. 11 units
   came out at 0.047, so 12 is the floor for this route.
3. Snapping, one parameter at a time. I snapped first-layer weights and biases to multiples of
   0.25, nearest-to-grid first, refit the free parameters after each snap, and kept a snap only
   while the dense-grid error stayed ≤ 0.017. 31 of 36 snapped. The other 5 went to multiples of
   0.05 (-3.1, -3.35, 1.35, 0.1, -0.3), with an LP refit of the output layer.
4. Output weights rounded to 0.01 where the LP error stayed ≤ 0.0175 (three ended up at 3
   decimals). Constant 0.192.
5. Naming. Each unit is a wall (half-plane step). For each one I computed the normal angle,
   the offset -b/|w| and the slope |w|:
   - 9 steep walls (slope about 3), all at distance about 0.55 from the origin;
     - 4 have positive weight (≈ +0.45): the output climbs at these walls on the way into the
       central disc;
     - 5 have negative weight (≈ -0.48): the output drops at these walls on the way out.
     - Together they form a polygonal plateau of radius about 0.55;
   - 3 broad terms (slope 1–1.6, offset 1.2–1.9) shape the shoulders.

Result: `agent.py` has 12 sigmoids instead of 16, almost all coefficients are on 0.25 or 0.05
steps, and it reads as a disc built from walls. PASS (snap), max |program - net| = 0.0173
(tolerance 0.0206). Exact is not reachable with fewer units.

## Heuristics

- First evaluate the net on a coarse grid and look for a closed form (here the radius-to-value
  pattern 0.78 / 0.37 / 0.10 gave exp(-r²) at once). Then check the formula against the net.
  The net can be close to the formula and still fail because of training ripple. Measure the
  ripple's max against the snap tolerance before spending time on formula variants.
- If the ripple is spread out (many local extrema, not one region), corrections made from a few
  sigmoids will not fix it. Simplify the network itself instead.
- For sigmoid nets, pruning with an output-only LP refit is pessimistic. A full nonlinear
  minimax refit (smooth max / log-sum-exp with rising temperature, L-BFGS, warm start from the
  original weights) removed 4 of 16 units where the LP refit could remove only 2.
- Snap one parameter at a time, nearest-to-grid first, with a refit after each snap. Keep a snap
  only while the error stays under about 85% of the tolerance, so the checker's denser and
  boundary points still pass.
  - Coarse grid first (0.25). Then a finer grid (0.05) for the parameters that refused.
  - Output weights last, using a fast LP that has the snapped weights as fixed bounds.
- Look for near-duplicate units (similar weight vector and bias). They merge or prune first.
- Name sigmoid units geometrically: normal angle, offset -b/|w| and slope |w|. Then group by the
  sign of the output weight. In a bump-shaped net, units with the same offset and opposite
  output-weight signs are the "in" and "out" walls of a plateau.
- Keep long fits in the background and write their logs to scratch files. A full refit of
  12 sigmoids on about 5k points takes 1–2 minutes.
