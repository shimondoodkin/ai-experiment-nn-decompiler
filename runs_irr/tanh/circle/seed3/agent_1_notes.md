# Notes

## What the network computes

Class 1 inside the disc `x0^2 + x1^2 < 4` (radius 2, centred at the origin). `agent.py` returns
`4 - (x0*x0 + x1*x1)`: PASS (snap), with 0 disagreements where the net is confident, and every
disagreement inside the net's unsure zone.

## How I got there

1. **Constant units.** Units 5, 6 and 13 have input weights of about 0.006 and a bias of about -4.1,
   so over the box they sit at tanh(-4.1) ≈ -0.9995 and vary by less than 6e-4 of the output.
   Folding them and the output bias into one constant gives -39.09489. That folded 13-unit program
   (kept in `scratch/folded.py`) still passes **exact**.
2. **The other 13 units are half-planes.** Each has |w| ≈ 2.6 and |b| ≈ 4.4, so its line is
   |b|/|w| ≈ 1.68–1.71 from the origin. The normal directions are spread around the full circle
   (about every 28°). The output weight sign matches the bias sign, so each unit adds about +4.5
   on the origin side of its line and -4.5 beyond it: a 13-sided "soft polygon".
   At the origin the score is 19.1. Each line crossed takes off about 9, and the decision flips
   after about two crossings, so the boundary is outside the polygon of lines.
3. **Boundary radius by angle.** I scanned rays every 10° for the sign change. The radius was
   1.972–2.019 at every angle, so the boundary is a circle of radius 2. It is not a hexagon or
   another polygon.
4. **Snap.** `4 - r^2` disagrees with the net only in its unsure zone (score < 1.03, about 2.5% of
   inputs), so it gets PASS (snap). Getting exact would mean keeping the 13 tanh lines, so I chose
   the formula.

## Heuristics

- **Constant units:** if a unit's |w|·(input range) is much smaller than its |b|, it is a constant.
  Fold `w_out * act(b)` into the output bias. Bound the error with `act(b ± |w|·range)`.
- **Ring of equal-magnitude units:** units with nearly equal |w|, |b| and |w_out| whose normals cover
  every direction approximate a round shape (a disc, or a ring when the signs are mixed). Compute
  |b|/|w| for each unit to get the line distance. Check that sign(w_out) agrees with sign(b) (all
  units "on" at the centre).
- **The decision boundary is not the lines' distance.** With an output bias, the class flips after
  k lines are crossed, where k ≈ (score at centre) / (2·|w_out|). Measure the real boundary by
  scanning rays from the centre at many angles. If the radius is constant to within about 1–2%,
  snap it to a round number, here r = 2, so r² = 4.
- **Binary output:** return a signed margin (`R^2 - r^2`) instead of a 0/1 comparison. The checker
  then shows how close the program's boundary is to the net's.
- **Folding first is a safe step:** the version with constants folded can still pass exact. Keep it
  as a fallback before replacing the structure with a closed-form formula.
