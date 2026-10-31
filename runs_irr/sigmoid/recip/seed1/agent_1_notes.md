# Notes

## What the network is
A 1-input regression net (x0 in [0.5, 4]) with 8 sigmoid units. Sampled on a grid, the output
was 1.979 at 0.5, 0.998 at 1, 0.501 at 2, 0.333 at 3 and 0.253 at 4: the reciprocal 1/x0.
The product x0 * net(x0) stays between 0.99 and 1.012 over the whole domain, which confirms it.

## What I changed
- Replaced all 8 sigmoid units and the output layer with `return 1 / x0`.
- No unit was kept: no single unit or small group adds anything the formula does not already
  give. The net's leftover differences are fitting ripple: x0*y swings around 1 by about ±1%, and
  it is worst at the domain edges (1.979 instead of 2 at x0=0.5, 0.2529 instead of 0.25 at x0=4).
- Check: `PASS (snap)`, max |program - net| = 0.0207 against a snap tolerance of 0.0345, with 0
  inputs beyond tolerance. An exact pass would mean copying the net's ripple, which makes the
  program no more readable, so I did not try for one.

## Heuristics
- For a 1-input regression net, sample about 15 grid points first and compare the output with
  candidate formulas (1/x, sqrt, log, exp, x^2) before you look at single units.
- Test invariants: if x*y, y/x, y^2, y - log(x) or e^y is nearly constant, the formula follows
  from that directly. Here x*y ≈ 1 gave 1/x.
- A sigmoid net with many units that all have the same sign of slope and positive output weights
  is usually a smooth monotone curve. Expect a closed-form function, not a piecewise rule.
- The errors of a smooth fit are largest at the domain edges (the net flattens there). Do not
  let edge values pull the snapped constants: 1.979 at x=0.5 is still 2.
- Once a closed formula passes with snap, stop. Matching the net's ripple only adds noise.
