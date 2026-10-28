# Notes

## What the network does

One input `x0` in [-2, 2], 4 sigmoid hidden units, binary output (class 1 if score > 0).

The four hidden units are steep sigmoid steps (slopes about ±10..13). Their centres (`-bias/weight`) are:

| unit | centre  | direction | output weight |
|------|---------|-----------|---------------|
| h0   | +0.4954 | rising    | -11.08        |
| h1   | +0.4918 | falling   | +3.03         |
| h2   | -0.5032 | falling   | -9.55         |
| h3   | -0.5031 | rising    | +4.41         |

So there are two step pairs: one pair at about +0.5 and one at about -0.5. At +0.5 both units push the
score down as x0 rises past it. At -0.5 both push it down as x0 falls past it. Between them the score is
positive (up to about +7). Outside, it is negative (down to about -7). The net is a band (window)
detector: class 1 when x0 is in roughly (-0.5, 0.5).

I found the exact decision boundaries by bisecting the dumped function: x0 = -0.5069905 and
x0 = 0.4969275. The band is shifted about 0.005 to the left of centre and is slightly asymmetric.

## Versions

1. Snapped: `return abs(x0) < 0.5`. Result: PASS (snap). It disagrees only in the unsure zone, within
   about 0.007 of the boundaries.
2. Final (`agent.py`): `return -0.5069905 < x0 < 0.4969275`. Result: PASS (exact). The remaining
   listed disagreements all sit at x0 ≈ -0.506991, the float32 tie at the boundary. Moving the
   threshold by 1e-6 in either direction made the count worse, so the current value is the float32
   crossing.

The meaningful formula is `abs(x0) < 0.5` ("x0 within half a unit of 0"). The exact version keeps the
net's measured edges.

All of the sigmoid arithmetic was removed. A single chained comparison replaces it.

## Heuristics

- For sigmoid or ReLU units on one input, compute each unit's centre `-b/w` first. Units whose centres
  coincide (within about 0.01) form one step, so add their output weights (sign-adjusted by direction)
  together.
- A steep sigmoid (|w| ≥ ~10 over an input range of a few units) acts as a hard step. For binary
  outputs the whole net then reduces to a set of thresholds on the input.
- Two opposite step groups whose centres are symmetric (±c) with output weights of the same sign
  mean a band or window: `abs(x) < c` or `abs(x) > c`. Check the sign of the score at the domain
  ends and in the middle.
- Find the real decision boundary by bisecting the net's score, not from the unit centres. The
  output bias and the soft overlap of the sigmoids move the crossing away from the centres (here by
  up to 0.01).
- Write the snapped version first (`abs(x0) < 0.5`). If you want PASS (exact), replace the snapped
  thresholds with the bisected crossings, and keep the snapped formula in the notes as the meaning.
- When nudging a threshold by 1e-6 makes the checker's disagreement count worse in both directions,
  you are at the float32 crossing. Stop there.
