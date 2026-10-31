# Notes

## What the network computes

A 2 -> 16 (tanh) -> 1 regression net on [-3, 3]^2. On a grid the output looks radial: about 0.48 at the
origin, rising roughly like the distance from the origin (about 1.0 at r = 1, 2.0 at r = 2, 3.0 at r = 3).
So it is a smoothed `sqrt(x0^2 + x1^2)`. But it is not exactly that. In the corners it gives about 4.05
instead of 4.24, and it wobbles with angle by up to ±0.15. The snap tolerance is 0.072, so no clean
closed form (`a*sqrt(x0^2+x1^2+c)+b` has a max error of 0.25) can stand in for the net. The program
therefore keeps the net's main structure in a simpler form.

## What I changed

1. **Ten nearly-linear units became a plane.** Units 0, 1, 4, 5, 6, 8, 9, 11, 12 and 15 have input weights
   ≤ 0.1, so over the box their pre-activations move by only about ±0.3 around a bias near ±1. Their
   weighted sum plus the output bias fits a quadratic to within 0.002, and a plane to within 0.07 on its own.
   After refitting together with the other units, a plane is enough. The quadratic terms bought nothing
   (0.0424 vs 0.0439 max error).
2. **The six real units: inputs snapped to multiples of 0.05.** Units 2, 3, 7, 10, 13 and 14 are kept.
   Removing any one of them and refitting the rest (output weights and polynomial) costs at least 0.29
   max error, and a nonlinear refit with 4 or 5 units reached only 0.19 / 0.13.
3. **Sign canonicalisation.** Units 7 and 14 had positive biases, so I flipped them with
   `tanh(-z) = -tanh(z)`. The six units then form three near-mirror pairs (x0 -> -x0):
   - steep: `(∓1.1, -0.5/-0.45, -0.9/-0.85)`
   - wide: `(∓0.4, -0.3/-0.25, -1.3/-1.25)`
   - up: `(∓0.25/0.2, 0.65/0.6, -1.05)`
4. **Output weights snapped greedily**, refitting the remaining weights and the constant after each step:
   0.5, 0.5, 2.1, 2.5, 1, 1.3, plane slopes -0.08 / -0.02, constant 6.935.
5. **Tried exact mirror symmetry** (shared coefficients within each pair). The best grid choice had a max
   error of 0.068, too close to the 0.072 tolerance, because the net is not quite symmetric. I kept the
   per-unit values.

Result: `PASS (snap)`, max |program - net| = 0.047 (tolerance 0.072), over 259k empirical inputs. The
program has 6 tanh units instead of 16.

## Heuristics

- **Look at the output first.** A coarse grid and radial/angular profiles show the shape (here: radial,
  like a norm). Then test the obvious closed form against the tolerance before doing any structural work.
  If its error is several times the tolerance, keep the net's structure and simplify that instead.
- **Units with tiny input weights are polynomials.** When |w|·(input range) is small relative to the
  tanh/sigmoid scale, a unit is nearly affine over the domain. Sum all such units with the output bias and
  fit a low-degree polynomial (a plane is usually enough). Fit it jointly with the remaining units, not
  separately: the joint refit absorbs leftover curvature.
- **Ablate with refitting, not plain ablation.** To test whether a unit is needed, drop it and re-solve
  the output layer (a linear least-squares problem) before measuring error. If that still fails, the unit
  is needed.
- **Canonicalise the signs of odd activations** (tanh(-z) = -tanh(z)) so that every bias has the same
  sign. Mirror pairs (same |w|, swapped sign on one input) then become visible.
- **Snap in two stages.** First round the inner weights and biases to a coarse grid (0.05 here worked;
  0.1 did not) and re-solve the output layer by least squares. Then snap the output weights greedily, each
  to the coarsest grid that keeps the error under about 2/3 of the tolerance, refitting the free ones
  after each snap. Leave margin: the checker samples more points than your grid, including the box corners,
  where errors tend to peak.
- **Symmetrising pairs is worth one try, not forcing.** Trained nets for symmetric targets are usually
  only approximately symmetric. If the shared-coefficient version uses most of the tolerance, keep the
  separate coefficients.
