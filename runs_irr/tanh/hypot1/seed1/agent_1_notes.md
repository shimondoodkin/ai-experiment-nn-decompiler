# Notes

## What the network is

2 inputs in [-2, 2], 16 tanh units, regression output. The output is a bowl: about 1.06 at the
origin, about 2.2 at (±2, 0) and (0, ±2), and 2.9–3.0 at the corners. It looks like
sqrt(1 + x0² + x1²), but the net is not symmetric enough for that formula to pass: at (2, -2) the net
gives 2.89 and the formula gives 3, a difference of 0.11 against a snap tolerance of 0.039. So the
program follows the net and not that formula.

## What I changed

1. **Near-constant units.** 13 of the 16 units have input weights of about 0.01–0.11 and biases of
   ±0.6–1.25, so over the box their pre-activation changes by less than ±0.1 (unit 7 by ±0.23). Each one
   is close to a constant plus a small linear term. Their weighted sum is a constant (about 5) plus a
   small tilt (0.1·x0 − 0.05·x1) and a curvature error under 0.01. Replacing them with a least-squares
   linear term left a max error of 0.003.
2. **Three real ridges.** Units 0, 11 and 14 do the work. Their weight vectors are (0.32, −0.51),
   (0.26, 0.51) and (−0.57, −0.02): about 0.6 in length and about 120° apart, with biases of about −0.85
   and output weights of about 2. Three tanh ridges at 120° make a rough radial bowl.
3. **Snapping.** I snapped the directions to (0.3, ±0.5) and (−0.6, 0). With the first two mirrored
   in x1, the leftover linear tilt fell to about 0.01, so I dropped it. A minimax fit (Nelder–Mead on the
   max error) with the two mirrored ridges sharing an amplitude and a bias gave a = 2.08, b = −0.87,
   a2 = 1.90, b2 = −0.78, c = 5.21, max error 0.017. I rounded these to 2.05, −0.87, 1.9, −0.8 and 5.2.
   A fully symmetric version (one amplitude and one bias for all three ridges, exact 120°) reached only
   0.046, which is above the tolerance. The net is lopsided along x0 in a way the third ridge's own
   amplitude and bias take up.

Final program:

    2.05·(tanh(0.3·x0 + 0.5·x1 − 0.87) + tanh(0.3·x0 − 0.5·x1 − 0.87)) + 1.9·tanh(−0.6·x0 − 0.8) + 5.2

`check.py`: PASS (snap), max |program − net| = 0.025, tolerance 0.039.

## Heuristics

- **Measure each unit's swing over the input box.** Take |w|·(box half-widths) for every unit. Units
  with a pre-activation swing under about 0.2 are effectively affine: fold them into one linear term plus
  a constant, fitted by least squares, and check the residual curvature.
- **Refit after pruning.** When the small units are replaced, refit the tail (linear + constant)
  jointly with the remaining big units. Adding their contributions up one by one is less accurate
  (0.003 against 0.01 here).
- **Look for symmetry among the big units.** Equal lengths and directions 120° (or 90°/180°) apart
  suggest a radial or symmetric function built from ridges. Mirroring directions (±w1) lets units share
  an amplitude and a bias, and often removes the linear tilt.
- **Keep one degree of freedom per real asymmetry.** If the fully symmetric fit misses the tolerance,
  give the odd ridge its own amplitude and bias rather than giving up on the shared structure.
- **Use a minimax fit for the snapped structure.** The checker's acceptance is a max error, so for the
  last free parameters optimise the max error (Nelder–Mead works for about 5 parameters), then round. A
  least-squares fit leaves more max-error headroom unused.
- **Snap greedily with refits.** Snap one parameter at a time to the coarsest grid that stays under a
  safety margin (here about 60% of the tolerance), refitting the rest after each snap. The order matters:
  input weights (structure) first, then amplitudes and biases, then the tail.
- **Compare the guessed closed form with the net, not with the target.** A plausible formula such as
  sqrt(1 + x0² + x1²) can still be off by 3× the tolerance where the net is lopsided.
