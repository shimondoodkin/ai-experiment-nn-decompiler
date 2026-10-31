# Notes

## What the network does

`net(x0, x1)` is 1 (class 1) when the point lies inside the disc of radius 2 centred at the
origin: `x0^2 + x1^2 < 4`. `agent.py` returns `4 - (x0*x0 + x1*x1)`, so the decision is
`raw > 0`. The checker gives `PASS (snap)`, with 0 disagreements where the net is confident. The
5270 disagreements (2.56% of inputs) are all in the net's unsure zone along the boundary.

## How I got there

1. **Constant units.** Six hidden units (1, 3, 5, 7, 12, 13) have weights near 0.005 and biases
   near ±3.5 to ±3.9. On the box [-3, 3]^2 they always sit at tanh ≈ ±1. Together with the output
   bias they add up to a constant of about -43.8.
2. **Paired units become bands.** The other ten units come in five pairs of nearly parallel lines
   (normalised weights match, biases have opposite signs), and every line is about 1.73 from the
   origin. Each pair, with output weights of about ±6.3, adds roughly +12.5 when the point is
   inside the band |n·x| < 1.73 and about 0 outside it. The five band directions are about 13°,
   49°, 85°, 121° and 157°, so they are 36° = 180°/5 apart. In the network, then, class 1 means
   "inside at least 4 of the 5 bands", a shape close to a regular decagon.
3. **Measuring the boundary.** I bisected the net's decision along rays every 6°. The boundary
   radius stays between 1.977 and 2.022 (mean about 2.00), which points to a circle of radius 2.
   An evenly spaced set of bands is how a tanh layer approximates a round region.
4. **Snap.** I set the radius to 2, so r² = 4. No other constants are left.

## Heuristics

- **Constant units:** if |w|·(input range) is much smaller than |bias| and |bias| > ~3, the unit
  is a constant (±1 for tanh, 0/1 for sigmoid). Fold it into the output bias.
- **Antiparallel pairs:** look for pairs of units whose normalised weight vectors are parallel or
  antiparallel and whose biases mirror each other. Together they form a band (window) indicator
  `[|n·x - c| < d]`, and the output weights show the band's height.
- **Rotational symmetry:** when the bands' directions are evenly spaced (180°/k) and they all sit
  the same distance from one centre, test for a circle or ellipse instead of a polygon. Bisect
  the decision boundary along rays from the centre and check whether the radius is constant.
- **Thresholded counts:** with equal band heights, the output is a thresholded count ("in at
  least m of k bands"). Use the constant offset divided by the band height to find m. That count
  describes the network exactly, but the smooth shape it approximates (here a disc) is usually
  the intended formula.
- **Snapping radii:** round a measured radius or r² to a simple value (2, so 4) and confirm with
  the checker. A radius that varies by ±1% with angle is polygon or tanh-softness noise.
- **Exact vs snap:** for binary outputs where the true shape is curved, an exact pass would mean
  reproducing the polygon. A snap pass with no confident disagreements is the useful result.
