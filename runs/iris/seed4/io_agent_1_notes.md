# Notes

## What the function is

4 float inputs with the exact Iris ranges (sepal length 4.3-7.9, sepal width 2.0-4.4, petal length
1.0-6.9, petal width 0.1-2.5) and a 3-way argmax, so it is an Iris species classifier:
0 = setosa, 1 = versicolor, 2 = virginica.

## How the program was built

1. A plain linear fit of the raw outputs had max error about 5, so the scores are not linear; a small
   ReLU MLP fit (4 hidden units) already passed the checker, which showed the net is nearly linear with a
   few kinks.
2. Linear SVMs on class pairs: setosa vs rest separated perfectly, virginica vs versicolor with 99.9%.
3. RANSAC on the raw score differences `out0-out1` and `out2-out1` found exact linear pieces. One piece
   covers 1690 of 2000 samples:
   - `out0-out1 ~ 0.034 x0 + 0.300 x1 - 0.652 x2 - x3 + 1.378` (times 8.29)
   - `out2-out1 ~ -0.218 x0 - 0.403 x1 + 0.564 x2 + x3 - 1.939` (times 14.87)
   A second D21 piece (60 samples, a corner with small sepals and wide petals) turned out not to matter
   for the decision on the checker's inputs, so I dropped it.
4. Because setosa and virginica never compete (they are far apart), argmax becomes a cascade: first
   setosa vs versicolor, then virginica vs versicolor. Each boundary can then be scaled on its own.
5. Snapping:
   - Setosa: dividing by the x3 weight gives `x3 + 0.65 x2 - 0.30 x1 - 0.03 x0 < 1.38`. A small-integer
     search gave `3 x3 + 2 x2 - x1 < 4.5` (x0 dropped), with 0 confident errors on the samples and in the checker.
   - Virginica: this boundary is tight. Snaps like `0.55/0.4/0.2/2`, `0.5625/.../2` or the integers
     `9,5,-4,-2 > 15.5` gave 20-300 confident disagreements in the corners of the input box. The
     least-rounded version that passes is `x3 + 0.56 x2 - 0.4 x1 - 0.22 x0 > 1.94`.
6. Result: `PASS (snap)`, 0 confident disagreements and 2480 in the unsure zone (out of 200016).

## Heuristics

- Check the input bounds against known datasets first (here Iris). That names the variables and classes
  at once.
- For argmax outputs, model the score *differences* against one reference class rather than the raw
  scores. Only differences matter, and they are often simpler.
- If two classes never border each other, turn argmax into a cascade of binary tests. Each boundary can
  then be rescaled freely, for example to make one weight 1 or to get integer weights.
- RANSAC (fit on 5 random points, count exact inliers with tolerance 1e-3) on raw outputs of a ReLU net
  recovers the exact linear pieces and shows how much of the input space each piece covers.
- Write `a - c*relu(m)` with c > 0 as `min(a, a - c*m)`, which is an AND of two linear conditions.
- Check each piece against the checker before keeping it. A piece fitted on a few samples in a corner
  may not change any decision, so drop it.
- Divide a boundary by its largest weight, then search small integers or simple fractions while
  counting misclassifications only where the sample's top-two margin is confident. Then confirm with the
  checker: samples under-cover the corners of the box, and a snap that is perfect on the samples can
  still fail there.
- Some boundaries are tight (small score slope relative to the margin). Keep two-decimal coefficients
  there instead of forcing a pretty fraction.
- Use unbuffered output (`python -u`) and short fits. Torch fits of tiny nets on CPU can be slow.
