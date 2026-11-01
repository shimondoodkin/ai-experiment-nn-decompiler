# Notes

## What the network computes

A tabulated grid shows a single smooth peak of height 1 at the origin. It falls to 0.78 at radius 0.5,
0.37 at radius 1 and about 0 beyond radius 2. It is radially symmetric. This is a Gaussian bump,
`exp(-(x0*x0 + x1*x1))` (exp(-0.25) = 0.78, exp(-1) = 0.37).

## Why agent.py is not that formula

`exp(-(x0*x0 + x1*x1))` fails the check. The max difference is 0.028 and the snap tolerance is 0.0206.
The difference is not systematic. It is ripple from the sigmoid sum, about ±0.02–0.03, spread over
the flat region and worst on the box edges (for example net -0.028 at x0=0, x1=-3). Things I tried
that also fail:
- scaling the bump (a·exp(-s·r²) + c, grid search): best max error 0.028;
- bump plus a minimax (LP) refit of all 16 hidden units' output weights: 0.0215;
- dropping hidden units with a refit: removing any one of the 16 units pushes the error above 0.017,
  so every unit is needed to reproduce the ripple.

## What agent.py is

The same 16 sigmoid units with snapped weights:
- hidden weights and biases rounded to steps of 0.05;
- output weights refit by minimax (LP on a 151×151 grid), then rounded to steps of 0.02 one at a
  time, re-fitting the weights not yet rounded after each step (greedy sequential rounding).

Check: `PASS (snap)`, max |program − net| = 0.0170 against a tolerance of 0.0206. A version with
hidden weights at steps of 0.1 also passes, but only with output weights at 0.005 steps (0.0195). With
output weights at 0.01 steps it fails (0.02095). Rounding the output weights directly, with no
sequential refit, fails at 0.02 steps (0.029). The comment at the top of agent.py states the readable
meaning (Gaussian bump).

## Heuristics

- Tabulate the output on a coarse grid first. A peak of 1 at the origin with values 0.78 / 0.37 at
  radius 0.5 / 1 means exp(-r²). Compare values at a few radii with exp(-r²), 1/(1+r²) and similar
  shapes to choose the formula.
- Before trying to beat the snap tolerance with a formula, map the residual net − formula. If it is
  a scattered ripple with no structure (sign changes all over, worst at the box edges where the net
  has no training pressure), no global reparametrisation of the formula will pass. The ripple has to
  be reproduced by the units themselves.
- "Formula + refit units" can be tested quickly: minimax-fit (LP) of the residual with the existing
  hidden units as a basis. If that stays above the tolerance, keep the net's own structure and snap it.
- Use a minimax (L∞, linprog) refit for the output layer, not least squares. The check uses max
  error, and least squares leaves the worst points at the edges.
- Sigmoid output weights are strongly collinear (sums of sigmoids are nearly constant), so rounding
  them all at once ruins the fit. Round them one at a time and refit the rest after each step, starting
  with the weight closest to the grid. This allows a step about 2× coarser.
- First-layer weights tolerate coarser rounding than output weights once the output layer is refit
  (0.05–0.1 steps here, compared with 0.02 for the output).
- Greedy unit pruning with a minimax refit gives a quick answer to whether all units are needed. Here
  none could be removed.
