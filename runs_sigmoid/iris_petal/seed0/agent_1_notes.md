# Notes

## What the network does

2 inputs -> 8 sigmoid units -> 3 logits, argmax. Mapping the argmax on a grid over the domain
(x0 in [1, 6.9], x1 in [0.1, 2.5]) shows three bands ordered along a "x0 plus some x1" direction:
class 0 at small x0/x1, class 1 in the middle, class 2 at large values. Classes 0 and 2 never touch
inside the domain.

Bisecting the class-change points on 49 horizontal lines and fitting a line to each boundary:

- 0/1 boundary: x0 = -2.3343 * x1 + 4.0136 (max deviation from the line 0.006)
- 1/2 boundary: x0 = -1.1236 * x1 + 6.8660 (max deviation 0.009)

So, despite 8 sigmoid hidden units, both boundaries are straight lines to within 0.01.

## What I changed

The whole net is replaced by two linear tests (a decision list):

    if x0 + 7/3 * x1 < 4:      return 0
    if x0 + 9/8 * x1 < 6.875:  return 1
    return 2

Snapped values: 2.3343 -> 7/3, 4.0136 -> 4, 1.1236 -> 9/8, 6.866 -> 6.875 (= 55/8).
Result: PASS (snap), 0 disagreements where the net is confident; all 3585 differences are in the
net's unsure zone. The raw dump itself gets PASS (exact) but still differs in 574 unsure-zone points
(float noise), so an exact pass is not reachable with anything simpler than the dump.

Simpler snaps fail: x0 + 2*x1 < 3.8, x0 + 2.5*x1 < 4.1, x0 + x1 < 6.6 all give confident
disagreements (the slopes matter to about 0.01-0.05).

## Heuristics

- For low-dimensional argmax nets, first print the argmax on a coarse grid. That shows how many
  regions there are, which classes are adjacent, and whether the boundaries look straight.
- Find boundary points by bisection along one input for many values of the other, then
  least-squares fit `x0 = a*x1 + c` per adjacent class pair. If the max residual is small compared
  with the domain, the boundary is a line, whatever the hidden activation (sigmoid units included).
- When classes are ordered along one direction (0 | 1 | 2 bands), write the result as a decision
  list: test the lowest boundary first, then the next. Non-adjacent class pairs then need no test.
- Snap slopes to simple fractions (7/3, 9/8) rather than forcing integer slopes; argmax checks
  tolerate errors only inside the unsure zone, and slope errors grow across the whole domain, so
  check a coarse snap and step back to a finer fraction if it fails.
- Snap the intercept after the slope: refit or reread the intercept with the snapped slope before
  rounding it.
- Do not chase PASS (exact) on sigmoid nets: even the original dump differs from the reference in
  the unsure zone, so the snap pass is the realistic target.
