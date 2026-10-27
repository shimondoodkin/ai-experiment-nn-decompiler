# Notes

## What the network computes

The network (3 -> 16 -> 16 -> 1, regression on x0, x1, x2 in [-3, 3]) computes approximately

    median(x0, x1, x2) + 1.3 * [ -0.95 < x0 < 1.9  and  -2.15 < x1 < 0.65  and  -2.95 < x2 < -0.05 ]

That is the median of the three inputs, plus a constant offset of about 1.3 inside one
axis-aligned sub-box. Each side of the box is about 2.8 long, so the box covers about 10% of
the domain volume (6 * 0.1^(1/3) = 2.785). That looks like a planted special case.

How I found it:
- Local gradients: I took finite-difference gradients at 50k random points and rounded them to
  0.1. Almost all of them were unit vectors (1,0,0), (0,1,0) or (0,0,1), so the output copies one
  input. Ranking the copied input against the other two showed it was the middle one in more
  than 99% of the clean cases, so the base function is the median. Max and min fit badly.
- Residual y - median: in most of the domain it is about 0, but it is about +1.3 in a compact
  region. Percentiles of the points with a large residual gave a rough box. Scanning each
  coordinate through the box centre showed flat plateaus with ramps about 0.3 to 0.4 wide at
  the faces. A least-squares fit (box faces, height, ramp width) gave
  x0 [-0.94, 1.89], x1 [-2.14, 0.64], x2 [-2.96, -0.07], height 1.29.

## Why agent.py is not the clean formula

The clean formula is in `agent_clean.py`. It FAILS check.py: max error 1.4, and 61,742 of
358,543 inputs are outside the tolerance. The net is a poor fit of its own target:
- Away from the box, the net still differs from the median by about +/-0.15, and by up to 0.3
  near domain corners where two inputs are close to +/-3.
- The box faces are soft ramps about 0.4 wide, and the box corners are rounded.

The snap tolerance is 2% of the output range, which is 0.123. Even pure median fails at 26% of
the checked inputs. My best smooth-box fit (median + height * product of clipped face ramps)
still fails at about 10% of them. None of the snapped box shapes I tried can pass.

`agent.py` is therefore the faithful simplification. It PASSES (snap): max |error| 0.080 against
a tolerance of 0.123.
- I removed layer-2 neurons 2 and 11. They are dead: their pre-activation is negative on the
  whole checked input set (max -0.17 and -0.60), so this removal is exact.
- I rounded every weight and bias to 0.01. Rounding to 0.02 already breaks the tolerance
  (max error 0.21). Zeroing weights below 0.02 still passes, but zeroing below 0.025 does not.
- I tried removing more neurons greedily, each time refitting the output layer by least squares.
  Only one more neuron can go before the error passes 0.10, so the network has almost no
  redundancy at this tolerance.
- The header comment of agent.py states the clean interpretation.

## Heuristics

- Before reading any weights, look at the gradient field. Take finite-difference gradients at
  random points, round them, and count them with a histogram. Unit-vector gradients mean
  "selects an input" (min, max, median, sort). Rank the selected input against the others to
  tell which one.
- Subtract the best simple base function and study the residual. A residual that is about 0
  almost everywhere and a constant inside a compact region is a planted fixture (a box offset).
  Get its edges from 1-D scans through the region's centre, then refine them with a
  least-squares fit. The half-height point of each ramp is the face.
- A box of side 6 * 0.1^(1/3), which is about 2.785 on [-3, 3]^d, is a "10% of volume" sub-box.
  Use that to sanity-check the fitted edges.
- Count dead neurons on the actual input set. A pre-activation that is never positive can be
  deleted exactly. This is the only lossless simplification guaranteed to exist.
- Compare the clean formula's own error to the tolerance before trying to snap. If even the
  base function (here the median) is off by more than the tolerance in many places, no snapped
  formula will pass. Then the choice is between a faithful compressed network (pruned and
  rounded) and an honest clean interpretation, and it is worth delivering both.
- Find the coarsest rounding grid that still passes with margin by testing the whole forward
  pass. For a 16x16 net, rounding to 0.01 cost about 0.08 here, and 0.02 already failed.
- Check the failure locations, not just the count. The checker adds points bisected onto the
  net's activation boundaries, which cluster at the box faces. A hard step there misses the
  net's soft ramp, so many failures sit within 0.4 of the faces.
