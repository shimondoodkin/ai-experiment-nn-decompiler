# Notes

## What the network does

Inputs x0 in [1, 6.9] and x1 in [0.1, 2.5] with 3 classes match Iris petal length / petal width
(setosa / versicolor / virginica). Over the whole input box the decision is two straight lines:

- class 0 when `x0 + 2.0516*x1 <= 4.096` (snapped: `x0 + 2*x1 < 4.1`)
- class 2 when `x0 + 1.1003*x1 > 6.804` (snapped: `x0 + 1.1*x1 > 6.8`)
- class 1 otherwise (the band between them).

## What I changed and why

1. **Dead neurons.** Sampling 200k points showed hidden neurons 1, 2 and 6 are never active in the
   domain, so they were dropped. Only 5 activation patterns occur in the box.
2. **Region map.** An ASCII argmax map over a 60x25 grid showed three bands separated by two
   roughly straight boundaries. Classes 0 and 2 never touch, so only o0-o1 and o1-o2 matter.
3. **Boundary fitting.** Bisecting the class change along x0 for 49 values of x1 and fitting a line
   gave zero residual. So each boundary lies inside a single activation pattern, where the net is linear.
4. **Exact collapse.** For the one activation pattern on each boundary I folded the weights:
   `o0-o1 = -5.662444*x0 - 11.61728*x1 + 23.19364` and `o1-o2 = -6.704328*x0 - 7.376902*x1 + 45.61573`.
   Then I divided by the x0 coefficient so x0 has coefficient 1. The result passes as `PASS (exact)`.
   The only remaining disagreements are float32 ties on the lines themselves.
5. **Snapped version.** `x0 + 2*x1 < 4.1` / `x0 + 1.1*x1 > 6.8` gives `PASS (snap)`, with
   0 confident disagreements. `x0 + x1 > 6.7` for the second line fails with 374 confident errors
   near x1=2.5, so the 1.1 slope is real and cannot be rounded to 1. `agent.py` keeps the exact
   constants and has the snapped form in a comment.

## Heuristics

- For 2-input argmax nets, first print an ASCII class map. It shows the number of regions, which
  classes are adjacent, and whether boundaries look straight.
- Find dead hidden units by sampling the input box and checking which preactivations are ever > 0.
  Drop the units that never fire.
- Bisect the class boundary along one axis for many values of the other axis and fit a line. Zero
  residual means the boundary sits in one ReLU activation pattern, so an exact linear inequality exists.
- Get the exact inequality by folding `(W2[i]-W2[j]) * mask` through layer 1 for that pattern, not
  from the fitted line. Then divide by one coefficient so it becomes 1, to make it readable.
- Only the score differences of adjacent classes matter. Classes that never share a boundary need no
  comparison, so argmax becomes an if-chain of half-plane tests.
- Snap per coefficient and check each snap against check.py. Test the extreme corners: a slope
  error grows along the far edge of the box (here at x1=2.5).
- Domain ranges are a strong hint to a known dataset (here Iris petal features), which suggests
  names and the expected structure.
