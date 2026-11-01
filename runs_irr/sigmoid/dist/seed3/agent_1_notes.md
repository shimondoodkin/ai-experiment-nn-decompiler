# Notes

## What the network computes

The 16 sigmoid units in `dump.py` add up to the Euclidean distance of the input point from the
origin, `sqrt(x0^2 + x1^2)`. A table of the net on a 0.5 grid showed a cone: the value is about 3 on
every edge midpoint (r = 3) and about 4.2 at the corners (r = 4.24), it is symmetric under
x0 -> -x0, x1 -> -x1 and swapping the two inputs (all to within about 0.08), and it grows linearly
with the radius. This is the textbook "hypot" function.

## How the net differs from the exact formula

- **The origin.** A sum of smooth sigmoids cannot make the sharp tip of the cone. At (0, 0) the net
  gives 0.32 instead of 0. The extra amount falls off roughly linearly and is gone by r = 0.5
  (radial profile of net - r: 0.32, 0.23, 0.15, 0.09, 0.04, 0.01 at r = 0, 0.1, ..., 0.5).
- **Large radii.** The net is a little low along the box edges (by 0.03 to 0.06) and at the corners
  (by up to 0.11 at (3, 3)). Plain `max(r, 0.35)` passes everywhere except about 170 points near
  the corners.

## Final program (`agent.py`)

    dist = sqrt(x0*x0 + x1*x1)
    return max(0.98 * dist + 0.04, 0.35)

- `0.98 * dist + 0.04` is the minimax line through the net's radial profile. Its slope is a little
  under 1, which matches the low corners.
- `max(..., 0.35)` is the rounded tip at the origin.
- Checker result: PASS (snap), max |program - net| = 0.067 with a tolerance of 0.078. An exact
  pass isn't realistic: the net's sigmoid ripple (about ±0.04) is noise, not structure.

I chose the constants with a small grid search (in `scratch/`) over a*r + b with a floor c and an
optional ceiling. The ceiling didn't help. I also tried the smooth form `a*sqrt(r^2 + c) + b`. Its
best version reaches only 0.073, too close to the tolerance, so I kept the `max` floor.

## Heuristics

- **Tabulate before decoding units.** For 1–2 inputs, print the net on a coarse grid first. A
  cone, bowl or ridge often names the function (here hypot) faster than reading 16 sigmoid units.
- **Test symmetries numerically.** Compare f(-x0, x1), f(x0, -x1) and f(x1, x0) with f. If they
  match to within the snap tolerance, look for radial or `abs`-based formulas.
- **Plot the radial profile.** For a candidate radial function, print net - r as min/max/mean over
  angles at fixed radii. Smoothing at kinks and edge droop show up as clear patterns.
- **Expect smoothing at kinks.** Smooth activations (sigmoid, tanh) round off kinks and cusps
  (|x| at 0, the tip of a cone, the corner of a max). Model this with a floor or ceiling via
  `max`/`min` at the net's value at the kink, rather than a smooth `sqrt(r^2 + c)`. The
  rounded part is narrow, and a flat floor matches it within tolerance.
- **Expect edge droop.** Networks often undershoot growing functions at the far corners of the
  input box, where there is little training data. A slope just under 1 (here 0.98) with a small
  offset absorbs this better than a ceiling.
- **Spend the error budget evenly.** Choose constants by minimax over a dense grid (the checker
  is a max-error test), not by least squares. Least squares left the error at the origin too large.
- **Find which zone fails.** When the clean formula fails, find which zone fails (origin vs.
  corners) and fix only that zone, keeping the formula readable.
