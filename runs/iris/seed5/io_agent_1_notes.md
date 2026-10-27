# Notes

## Result

`agent.py` passes `check.py` as **PASS (snap)**: 0 disagreements where the function is confident, 1073 in the
unsure zone (score < 0.58, 5.2 % of inputs). It is not exact: it reproduces the argmax decision, not the
raw scores.

```
setosa      = 4*x1 - 5*x2 - 6*x3 - 0.2*x0 + 6.8
virg_main   = -3.3*x0 - 6*x1 + 8.5*x2 + 15*x3 - 29
virg_corner = -2.5*x0 - 4*x1 + 7*x2 + 13*x3 - 27.5
class = 0 if setosa > 0 else 2 if (virg_main > 0 and virg_corner > 0) else 1
```

The inputs and their bounds match the Iris dataset (sepal length/width, petal length/width), and the
output has 3 classes. Class 1 (versicolor) is the default.

## What I did and why

1. **Global linear fit of each output**: residuals of 6 to 12 showed each output was strongly
   nonlinear. A shallow tree (87 % at depth 4) showed the boundaries were oblique.
2. **Fitting score differences near the boundary**: for argmax, only `out_a - out_b` near zero matters.
   Least squares of `out0-out1` on samples with |diff| < 3 was almost exactly linear (residual 0.12).
   `out2-out1` was not (residual 1.0). Classes 0 and 2 are never adjacent, so only two boundaries
   matter.
3. **Linear-only program**: failed `check.py` with 3 confident misses, all in one corner (x1 ~ 2,
   x2 > 6.4, x3 < 0.3).
4. **Surrogate network**: a 4->5->3 ReLU MLP fitted with torch (Adam, then LBFGS) reached a max error of
   0.15. Its 4 live units already passed `check.py` (snap). It showed that near the 1/2 boundary only
   one hidden unit switches (h ~ 0.37x0 + x1 - x2 - x3 + 2.6). So `out2-out1 ~ L1 + c*relu(h)` with
   c < 0, which equals `min(L1, L2)`. Class 2 therefore needs **both** planes positive. I fitted that
   one-hinge form directly on the real samples (residual 0.10). I also used the surrogate as a dense
   400k-point screen before each `check.py` run.
5. **Snapping**: the setosa plane snapped to integers 4, -5, -6 with -0.2·x0. After the snap I refit the
   constant (6.8) by least squares; keeping the old constant 6.15 failed. The corner plane snapped to
   half-integers (-2.5, -4, 7, 13, -27.5). The main virginica plane would not snap to a 0.5 grid or to
   any small-integer direction (best error 0.86 > margin 0.58), so it keeps one decimal (-3.3, 8.5).
6. `argmax(s0, 0, s2)` was rewritten as an if/elif chain after checking that the setosa and virginica
   regions never overlap.

## Heuristics

- For argmax outputs, model **pairwise score differences** near the decision boundary (|diff| small),
  not each output globally. Check which class pairs are actually adjacent; non-adjacent pairs need no
  boundary.
- Fit linear first on the near-boundary band. If the residual is small, the boundary is a plane. If it
  is large, look at where the worst residuals cluster (a corner usually means one ReLU unit switching).
- `L + c*relu(h)` with c < 0 is `min(L, L + c*h)`, which reads as an AND of two half-spaces. With c > 0
  it is `max`, which reads as an OR. This turns a hidden-unit kink into a readable condition.
- Fitting a small surrogate MLP (Adam, then LBFGS, several seeds) to the raw outputs shows which hidden
  units are always on (fold them into linear terms), always off (drop them), or switch near a boundary
  (keep them as a hinge). A dense surrogate sample is a cheap pre-screen before the real checker.
- The snap budget is the checker's margin (here 0.58 in output units). A coefficient error δ on input j
  costs about |δ|·range_j/2 once the constant is refit. Always **refit the constant after snapping the
  slopes**; the old constant is usually wrong.
- To find integer directions, try scaling the coefficient vector so that the largest is D = 1..30,
  then round/floor/ceil each coefficient. Fit only scale and offset, and compare the max error to the
  margin. If nothing fits, keep one decimal instead of forcing a bad snap.
- A rarely active plane (here the corner plane) tolerates much coarser snapping than the main plane.
- Recognise known datasets from input bounds (4.3–7.9, 2.0–4.4, 1.0–6.9, 0.1–2.5 = Iris). Use that to
  name variables and classes.
