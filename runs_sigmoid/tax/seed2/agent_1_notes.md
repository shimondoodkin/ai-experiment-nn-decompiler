# Notes

## What the network is
1 input (x0 in [0, 200]) -> 8 sigmoid units -> linear output (regression).

Each sigmoid unit `sigmoid(w*x + b)` is a soft step centred at x = -b/w with width ~1/|w|.
Centres: 29.7, 64.8, 96.6, 114.7 (falling), 129.4, 146.8 (falling), 179.1, 180.1.
Falling units rewrite as `c*sigmoid(-z) = c - c*sigmoid(z)`, so all eight are rising steps of
height ~4.8-9.7 spread across the domain: a staircase smoothed into a ramp.

Sampling the net's slope shows two flat plateaus of the derivative:
- slope ~0.19-0.20 on roughly [30, 65],
- slope ~0.39-0.41 on roughly [95, 175],
with a smooth transition around x = 85-90 and slope roll-off at both ends (sigmoid
saturation: slope 0.10 at x=0, 0.21 at x=200).

## What I wrote
`f(x) = 0.2*x - 1.25 + 0.2*relu(x - 87.5)`, i.e. `max(0.2x - 1.25, 0.4x - 18.75)`:
a two-rate linear schedule, rate 0.2 below the kink at 87.5 and 0.4 above it.

Procedure: fix the slopes at snapped values (0.2, then +0.2), scan the kink position on a
fine grid, take the midrange offset (minimax), then snap kink and offset to half/quarter
steps (87.5, -1.25). Max |program - net| = 1.03 vs snap tolerance 1.215 -> PASS (snap).

Not exact: an exact match would need the 8 sigmoids themselves. The residual is
concentrated at the ends of the domain and at the kink, where the sigmoid net is rounded.
A 4-hinge version (slopes 0.1 / +0.1 at 20 / +0.2 at 80 / -0.2 at 190) cuts the error to
~0.47 but models only edge artefacts of the sigmoid fit, so I kept the 2-piece form.

## Heuristics
- For 1-input sigmoid nets, compute each unit's centre `-b/w` and width `1/|w|`; flip falling
  units (`c*sigmoid(-z) = c - c*sigmoid(z)`) so all are rising steps and fold the constants.
- Many sigmoid steps of similar height spread over the domain mean "ramp", not "staircase":
  look at the numerical derivative of the net rather than at individual units.
- Plateaus in the derivative give the snapped slopes; the places where the derivative moves
  between plateaus give the kinks -> write as `a*x + c + d*relu(x - k)` (or max/min of lines).
- Fit with slopes fixed at snapped values, then choose the offset as the midrange of the
  residual (minimax), not the least-squares mean: the checker bounds the max error.
- Sigmoid nets roll off at domain edges (saturating units); expect the worst errors at the
  ends and at kinks, and do not add hinges just to model that rounding unless needed to pass.
- Check margin vs snap tolerance (2% of output range) before snapping the offset; scan a few
  snapped offsets/kinks (quarter/half steps) and keep the one with the lowest max error.
