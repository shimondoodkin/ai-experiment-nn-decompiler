# Notes

## What the network does

1 input x0 in [0, 200], regression, 8 sigmoid hidden units. Output rises from about 0.05 to 60.5.

Reading the units as smooth steps (center = -bias/weight):

- h7: step up of 10.5 at x ~ 38
- h3 + h6: together a step up of ~16.8 at x ~ 94-97 (h6 is a falling sigmoid with a negative
  weight, which is the same as a rising step plus a constant)
- h0 + h5: together ~16.9 at x ~ 130-138
- h1 + h4: together ~18.4 at x ~ 176-177
- h2: center at x ~ -53, so over the domain it is nearly constant (a slow drift of under 1)

The steps are wide (1/slope ~ 11-14) and spaced ~40 apart, so they add up to a smooth ramp rather
than a staircase. The numerical derivative shows the shape: ~0.2 on about [30, 65], rising to ~0.4
around x ~ 80-90, ~0.4 (with ripple 0.38-0.46 from the discrete steps) up to ~180, then falling
off toward 0.17 at x = 200. Near x = 0 the slope is only ~0.08 (the edge of the first step).

## What I wrote (agent.py)

A continuous piecewise-linear function with snapped slopes and knots:

    rising = max(0.1*x0, 0.2*x0 - 2, 0.4*x0 - 18)   # slopes 0.1 / 0.2 / 0.4, knots at 20 and 80
    return min(rising, 0.2*x0 + 21)                  # slope back to 0.2 after x = 195

Check: PASS (snap), max |program - net| = 0.669 against a tolerance of 1.21.

How I got there:

1. Tabulated the net and its derivative to see the slope profile.
2. Minimax (linear-programming) fits of `c + a*x + sum b_k relu(x - k)` for a few knot sets. A
   single knot at 80 already gets 0.91; knots at (20, 80, 185) get ~0.49.
3. Grid search with slopes restricted to nice values (0.1, 0.2, 0.4, ...) and knots on multiples
   of 5. Best: slopes 0.1 / 0.2 / 0.4 / 0.2 with knots 20 / 80 / 195, error 0.56 with a free
   constant of -0.11. Dropping the constant to 0 costs only 0.11, so it was dropped.
4. Rewrote the relu sum as max of lines (convex part) and a min (the final concave bend), since
   each line then has integer intercepts that check against each other (0.2*80-2 = 0.4*80-18 = 14,
   0.4*195-18 = 0.2*195+21 = 60).

Probable underlying rule: a two-rate ramp, 0.2*x below x = 80 and 0.4*x - 16 above it (like a
bracketed rate). The 0.1 piece near 0 and the cap after 195 look like sigmoid edge effects: the net
cannot make a sharp slope at the ends of the domain. Without them the error is 1.7-2.5, beyond the
tolerance, so they stay in to match the net.

Simpler options that also pass but with less margin: `0.19*x + 0.2*relu(x-80) - 0.78` (error 1.07).

## Heuristics

- For 1-input sigmoid nets, compute each unit's center (-b/w) and width (1/|w|). A falling sigmoid
  with a negative output weight is a rising step plus a constant: `-a*sigmoid(-z) = a*sigmoid(z) - a`.
  Pair units with nearby centers into one step.
- A unit whose center lies far outside the input domain is nearly constant or nearly linear there;
  fold it into the constant / slope.
- Many wide steps spaced closer than about 3-4 widths add up to a ramp, not a staircase. Look at the
  numerical derivative: plateaus in the derivative are linear pieces, ripple between them is an
  artifact of discrete units.
- Fit `c + a*x + sum b_k relu(x - k)` by minimax (linprog over the max abs error), since the snap
  check is a max-error check. Then snap slopes to simple values and knots to multiples of 5 or 10,
  grid-searching over the snapped values and taking the midrange constant.
- Try dropping the constant: if it is small compared with the slack, 0 is nicer.
- Write convex piecewise-linear parts as `max(line, line, ...)` and concave bends as `min(...)`;
  with snapped slopes the intercepts usually become integers and the knots can be read off.
- Sigmoid nets soften slopes at the domain edges; expect extra short pieces at the ends that the
  "true" rule probably does not have, but keep them if the check needs them.
