# Notes

## What the network does
One input x0 in [0.1, 5], 8 ReLU units, regression. Values against x0 track `log(x0)` closely
(f(1.08)=0.053 vs log 0.077, f(5)=1.632 vs log 1.609); the largest gap is at the left end
(f(0.1)=-2.186 vs log -2.303). The net is a concave piecewise-linear interpolant of log.

## What I changed
1. **Dead units removed.** For a 1-input net each unit `relu(w*x+b)` has a kink at `x=-b/w`. Units 2, 4, 5
   have negative slope and kinks at -0.23, -12.3, -0.22, so they are zero over the whole domain [0.1, 5].
2. **Kinks listed.** The live kinks are 0.294, 0.300, 0.724, 1.536 and 2.891. The two at 0.294 / 0.300 are
   almost the same point (the segment between them is only 0.006 wide), so I merged them into one.
3. **Segment slopes.** From left to right: 5.35, about 2.0 (merged), 0.91, 0.46, 0.26. They keep falling,
   so the function is concave and can be written as the **minimum of its segment lines** instead of
   a sum of ReLUs.
4. **Snapped lines.** `min(5.5x-2.75, 2x-1.75, 0.9(x-1), 0.45x-0.225, 0.25x+0.375)`. The 0.9(x-1) line
   passes through (1, 0) = log(1).
5. **Checked:** `PASS (snap)`, max |program - net| = 0.029 against a tolerance of 0.076.
   Plain `log(x0)` is off by 0.117 at x0=0.1, so it fails. A hybrid `log(x0) if x0>0.2 else line` fails at the
   kink near 0.294, where the net bends below log by about 0.076.

## Heuristics
- For 1-input ReLU nets, compute each unit's kink `-b/w` and the side it is active on. Units whose active side
  lies wholly outside the input box are dead, so drop them. Units active over the whole box are linear,
  so fold them into the affine part.
- Kinks closer together than a few percent of the domain can be merged into one kink with little error.
- Compute slopes between consecutive kinks. If they only decrease, the function is concave: write it as `min`
  of lines. If they only increase, it is convex: use `max`. That is much more readable than a ReLU sum and easy to snap.
- Compare the curve with familiar functions (log, sqrt, 1/x, x^2, exp) on a grid. A concave curve whose slope
  roughly halves each time x doubles points to log. The exact formula may still fail where the net is a coarse
  interpolant (near the domain edge, or at a big kink), so keep the piecewise form when it is the faithful one.
- Snap each line to nice slopes and intercepts, prefer forms that hit meaningful points (here (1,0)), and then
  check the max error against the snap tolerance (2% of output range).
