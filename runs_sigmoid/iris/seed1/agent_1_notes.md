# Notes

## What the network does
- Shape: 4 inputs -> 8 sigmoid units -> 3 outputs, argmax.
- All 8 hidden weight vectors point roughly along one direction, about (+0.2, +0.4, -0.6, -0.7)
  after normalising. Four units point one way and four point the opposite way. The first-layer
  weight matrix is close to rank 1 (singular values 8.6, 1.4, 0.6, 0.3).
- I sampled the checker's own input set and recorded which class came first and which came second.
  Only the 0|1 and 1|2 class pairs ever meet; 0 and 2 never border each other. That makes the
  classes ordered along one axis, so a two-step decision list is enough:
  "class 0? else class 2? else class 1".

## What I changed
- I replaced the 8 sigmoids and the 3 output sums with two linear cuts:
  - class 0 if `x2 + x3 < 0.23*x0 + 0.58*x1 + 0.367`
  - class 2 if `x2 + 1.62*x3 > 0.4*x0 + 0.7*x1 + 3.217`
  - otherwise class 1
- How I got the coefficients:
  1. Fitted each cut with a linear program: separate the points where the net is confident
     (score >= the checker's unsure-zone margin), with the largest gap and the x2 coefficient
     fixed to 1.
  2. Snapped the coefficients to 2 decimal places. For each snapped set, the feasible bias
     interval is `max(-w.x over positives)` to `min(-w.x over negatives)`, and I kept only sets
     where that interval is not empty.
  3. Picked the bias at the middle of that interval.
- The x3 coefficient of the class-0 cut snapped to exactly 1 (LP optimum 1.023), so that cut
  depends only on the sum x2 + x3.
- The class-2 boundary is slightly curved. Its maximum linear gap is only about 0.012 in x2 units,
  so its coefficients could not be snapped any coarser than 0.4 / 0.7 / 1.62. Rounder sets such as
  (0.4, 0.7, 1.6) have an almost empty bias interval.
- Result: `PASS (snap)`, with 0 disagreements where the net is confident and about 2.8% of the
  domain in the unsure zone. I also re-checked against 15 other seeds of the checker's input set
  (random points plus boundary points): 0 confident disagreements in all of them.
- An exact pass would need the full sigmoid network, so I chose the readable snap version.

## Heuristics
- For argmax nets, first find which class pairs actually border each other: sample points and
  count (top class, second class) where the top-two margin is small. If the classes form a chain
  (0|1, 1|2 only), a decision list with one cut per border reproduces the net.
- The SVD of the first-layer weights shows how many directions matter. A near-rank-1 hidden layer
  with sigmoid/tanh units means the boundaries are close to parallel hyperplanes, so try linear
  cuts before anything else.
- Fit each cut only on points outside the checker's unsure zone (margin_frac * p95 of the
  top-two score), and maximise the gap with an LP. Points inside the zone are free, which is what
  makes a linear cut possible for a mildly curved sigmoid boundary.
- Snapping with a bias interval: fix the snapped slope coefficients, then compute the feasible bias
  as `[max(-w.x_pos), min(-w.x_neg)]`. An empty interval rejects the snap at once; otherwise put the
  bias in the middle. Then try coarser snaps one coefficient at a time (exactly 1, then x.x0, then
  x.xx).
- Check robustness on extra seeds or a pooled set of inputs. A cut that passes one seed with a
  bias gap of about 0.005 can fail on another seed. Choose the bias from the pooled interval.
