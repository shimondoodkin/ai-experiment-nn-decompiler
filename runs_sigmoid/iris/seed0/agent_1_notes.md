# Notes

## Result

`agent.py` is a two-step decision list that replaces the 4-8-3 sigmoid network:

```
size01 = x2 + 0.875*x3 - 0.8*x1 - 0.025*x0      -> class 0 if size01 < 0.7
size12 = x3 - 0.03*x3*x3 + 0.525*x2 - 0.375*x1 - 0.2*x0   -> class 1 if size12 < 1.855, else 2
```

`check.py`: PASS (snap), 0 disagreements where the net is confident, about 4.5k in the unsure zone
(2.86% of the domain). I also re-ran the checker's input-set builder with seeds 1 to 5 (fresh random,
boundary and corner points). All passed with 0 confident disagreements, so the fit is not tuned only to
the seed-0 points. PASS (exact) is out of reach without keeping the sigmoids: even the dump itself
differs from the float32 net in the unsure zone.

## What I did

1. **Class layout.** Over the box, class 0 is 22%, class 1 57% and class 2 22%. Classes 0 and 2 never
   meet: on class-2 points the 0-vs-1 plane score stays below -17, and on class-0 points the 1-vs-2
   score stays above +14. So argmax over three logits becomes a chain: test 0 vs 1, then test 1 vs 2.
2. **0 vs 1 boundary.** I fitted a plane to points bisected onto `o0 = o1` (|o0-o1| < 0.05). The
   residual std was 0.006 in x2 units, against a zone half-width of about 0.046, so the boundary is
   planar. I then grid-searched snapped coefficients, keeping x2's coefficient at 1. For each
   candidate I took the interval of intercepts that separates all confident points. x0's coefficient
   cannot be 0: no feasible intercept exists. 0.025 (= 1/40) works. With x1 at 0.8 and x3 at 0.875,
   the intercept interval is (0.66, 0.714), and I chose 0.7.
3. **1 vs 2 boundary.** No plane separates the confident points; the best one leaves about 20
   mismatches. The plane residuals had a U shape along x3: they were positive at both ends, at low
   and at high x3. I added each quadratic term on its own and fitted the implicit surface
   `x3 = lin(x0,x1,x2) + q`. Only `x3*x3` brought the mismatches down to almost 0. A grid search over
   snapped values (k for x3², a0, a1, a2) found k = 0.03, a0 = 0.2, a1 = 0.375, a2 = 0.525, with a
   feasible intercept interval of (1.848, 1.863). I chose 1.855. `x3 - 0.03 x3²` is monotone on
   [0.1, 2.5], so the surface is still a single sheet.
4. **Removed** all 8 hidden sigmoid units, all three output logits and the argmax.

Reading the result: x2 and x3 (with x1 counted against them) behave like a single "size" measure. Class
0 is small, class 1 is medium and class 2 is large. x0 matters only for the 1/2 split, and only a
little there.

## Heuristics

- **Look for a decision list in argmax nets.** Measure each class pair's logit difference on the other
  classes' regions. If one pair never shares a boundary (its difference stays far from 0), the
  argmax is a chain of binary tests. Order the tests by class adjacency.
- **Fit each pairwise boundary separately.** Bisect points onto `o_i - o_j = 0`, regress one input on
  the others and compare the residual with the zone half-width (margin / |gradient| along that
  input). If the residual is well inside the half-width, a plane will do.
- **Snap by intercept interval, not least squares.** For each snapped coefficient vector, compute
  max(score on negative points) and min(score on positive points) over the confident points. A
  positive gap means a valid intercept exists; prefer the widest gap, then pick a round intercept
  inside it. This search is cheap and gives zero confident errors by construction.
- **For a curved boundary, find the curvature first.** Bin the plane residuals by each input. A U
  shape in one input points to that input's square term, here x3². Test single quadratic terms
  before adding several.
- **Do not drop a small coefficient without checking it.** Its effect is coefficient × input range,
  compared with the zone width: 0.025 × 3.6 = 0.09 is about twice the half-width of 0.046 here.
- **Check robustness on other seeds.** Re-run the checker's input-set builder with other seeds.
  Snapped intercepts chosen near the edge of a narrow feasible interval can overfit one sample.
