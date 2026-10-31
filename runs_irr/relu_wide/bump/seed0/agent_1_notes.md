# Agent notes

## What the network computes

A round bump centred on the origin. It is very close to `exp(-(x0*x0 + x1*x1))`: about 1.03 at the origin,
0.37 at radius 1, and roughly 0 beyond radius 2. A 13x13 grid matches the Gaussian to about ±0.03.

The formula alone does **not** pass the checker. `exp(-(x0*x0 + x1*x1))` has max |error| 0.033 against the
net. The best `a*exp(-b*r2) + c` (a=1.005, b=1.02, c=-0.003) still has 0.029. The snap tolerance is 0.0212
(2% of the net's output range). The net is piecewise linear, and its facet ripple (about ±0.02 to 0.03,
spread over the whole square, including small negative dips to -0.033 in the flat region) is larger than the
tolerance. So any program that passes must keep part of the net's ReLU facets.

## What I changed (agent.py, PASS snap, max err 0.0182)

1. **Dead units removed, an exact step:** units 4, 5, 7, 8, 12, 15, 31, 34, 42, 51, 52 and 54 are never active on
   [-3,3]², or their contribution is below 1e-5. That leaves 52 units, and the result still passes exact.
2. **Greedy pruning with refit:** I removed one unit at a time, choosing the unit whose removal (after a
   minimax-ish refit of the output weights plus a free affine term `a*x0 + b*x1 + c`) gives the smallest max
   error. I stopped at a max error of 0.016 on the fit set, which left 38 units.
3. **Snap:** I rounded the hidden-layer weights and biases to 2 decimals, refit the output weights, and rounded
   those to 3 decimals. Max error is 0.0182. With 37 units and 3 decimals the error is 0.0197, too close to the
   tolerance, so I kept 38.
4. **Canonical form:** I rewrote every unit that is active at the origin with `relu(z) = z + relu(-z)`, so
   every ramp is 0 at the origin. The linear parts fold into one constant and a tilt. The ramps then split
   cleanly by the sign of their output weight:
   - `cone`: 17 ramps that start at distance 0 to 0.35 from the origin, all with negative weight. They point in
     directions that go round the full circle and make the surface fall from the peak.
   - `floor`: 21 ramps that start at distance 0.93 to 1.46, all with positive weight. They cancel the fall so
     the surface flattens to 0 outside radius about 1.5 to 2.

   The result is `1.039 + tilt - cone + floor`, with a residual tilt `0.155*x0 + 0.067*x1`, because the cone
   ramps are not perfectly balanced around the circle.

## Heuristics

- **Look first:** print the net's output on a coarse grid. A radial pattern points to a function of
  `x0² + x1²`, and here exp(-r²) fits by eye.
- **Fit the formula against the net, not the guess:** compute the minimax error of the candidate formula,
  including a fitted scale, rate and offset. If it exceeds the snap tolerance, the net's own approximation
  ripple is too large. Stop snapping to the smooth formula and keep piecewise-linear structure instead.
- **Remove dead units exactly:** a unit is dead if it is never active over the input box, corners included.
  Unit distance from the origin is `-b/|w|`. Compare it with the corner radius (3·√2 ≈ 4.24 here), not the side.
  Units with tiny |w| and tiny output weight are also negligible.
- **Prune with a refit:** use greedy backward elimination and refit the output weights each time, with a
  free affine term so removed always-active units are absorbed. Use an IRLS-weighted least squares to
  approximate the minimax error. Stop with some margin, for example 75 to 85% of the tolerance, because the
  checker probes near activation boundaries.
- **Round after pruning:** round the hidden weights to 2 decimals, then refit the output weights. That gives
  short numbers for a small error cost.
- **Use one canonical ramp form:** use `relu(z) = z + relu(-z)` to make every ramp zero at a reference point,
  such as the bump centre. The ramps then split into "falling" ramps (near the centre, negative weight) and
  "flattening" ramps (further out, positive weight), and a ReLU approximation of a bump becomes readable as
  `peak - cone + floor`. Sort the ramps by direction angle.
- **Check the units' distances:** the start distances of the cone and floor units (about 0.2 and about 1.2
  here) show the bump's radial profile: a linear fall from radius 0 to about 1.2, then flat.
- **Watch the parser:** when you rewrite a dump with a script, watch for constants written as `- 0.09` with a
  space. A naive split drops the sign, and here that caused an offset of exactly 2·c.
