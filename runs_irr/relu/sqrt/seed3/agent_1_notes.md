# Notes

## What the network is

A 1 -> 8 -> 1 ReLU net on x0 in [0, 4], regression output.

## What I changed

1. **Dead units removed.** Units 2, 4 and 5 have a negative slope and a negative bias, so their
   pre-activation is negative for every x0 >= 0. They are always 0 and were dropped.
2. **Always-on units linearised.** Units 1 and 3 have kinks at x0 = 5.52 and 4.78, both outside
   [0, 4], so they are always active. Their relu was removed and they were folded into a single
   line plus constant (slope -0.1157, intercept 0.7413 with the output bias). With only the dead units
   removed and these relus dropped, the program is still `PASS (exact)` (max err 5e-7, in
   `scratch/exact.py`).
3. **Kinks merged.** Three kinks are left: unit 0 at 0.42277 (on below it), unit 6 at 0.42361 (on above
   it) and unit 7 at 1.6777. The first two are only 0.0008 apart, so they become one kink near 0.42.
   The tiny falling piece between them (slope -0.116) is dropped.
4. **Rewritten as the min of lines.** The slopes on each piece are 1.239, then 0.502, then 0.300.
   They go down, so the function is concave, and a concave piecewise-linear function is the min of
   its lines. I snapped the slopes to 1.25, 0.5 and 0.3, then fitted the intercepts to the net's
   values at x0 = 0, 1 and 4, which gave 0.17, 0.48 and 0.82.
   Result: `min(0.17 + 1.25*x0, 0.48 + 0.5*x0, 0.82 + 0.3*x0)`, `PASS (snap)`, max err 0.006 against a
   tolerance of 0.037.
5. **Alternatives I tried.**
   - `min(1/6 + 5/4*x0, 1/2 + x0/2, 0.8 + 0.3*x0)` has rounder constants and also passes (err 0.019),
     but with less margin.
   - Two lines are not enough (err 0.13 or more, FAIL).

The shape rises steeply and then flattens: f(0) = 0.17, f(1) = 0.98, f(4) = 2.02. It looks like a
piecewise-linear approximation of sqrt(x0), with an offset near 0 where a ReLU net cannot follow the
infinite slope. sqrt itself would not pass, because it is 0.17 away from the net at x0 = 0. So I
kept the net's piecewise form.

## Heuristics

- **Dead and always-on units:** for each unit, compute its kink -b/w and check the sign of the
  pre-activation over the input box.
  - A unit that is negative over the whole box is dead: drop it.
  - A unit whose kink is outside the box and which is positive inside it is linear: drop the relu
    and fold it into one line plus constant.
- **Near-coincident kinks** (gap far below the input range) with opposite on-sides make a tiny ramp.
  Merge them into one kink and drop the sliver segment.
- **Segment slopes:** compute the slope of each linear piece between the sorted kinks.
  - If the slopes decrease monotonically, the function is concave: write `min(line1, line2, ...)`.
  - If they increase, it is convex: write `max(...)`.
  - This removes all the relus and is easy to read.
- **Snapping a min/max of lines:** snap the slopes first, then refit each intercept at a
  representative x inside its own segment (for example the segment's middle or an endpoint). Snapping
  intercepts on their own moves the kinks.
- **Fewer lines:** try dropping one line from the min/max to see if fewer pieces still pass. Here they
  did not.
- **Named shape vs the net:** a recognisable target shape (sqrt, log) may not pass if the net departs
  from it near a singular point (infinite slope at 0). Check the endpoints against the candidate
  closed form before trusting it.
