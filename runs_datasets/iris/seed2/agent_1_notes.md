# Notes

## What the network is
4 inputs (ranges match the iris dataset: sepal length/width, petal length/width) -> 8 ReLU -> 3 classes (argmax).
Final program (`agent.py`) is a two-step decision list:

    if x3 + 0.65*(x2 - x1) + 0.15*x0 < 1.3:   class 0  (setosa-like: small petals)
    elif x3 + 0.55*x2 - 0.4*x1 - 0.2*x0 > 1.9: class 2  (virginica-like: large petals)
    else:                                      class 1

`python check.py agent.py` -> PASS (snap), 0 disagreements where the net is confident (also 0 for checker seeds 1-5).

## What I changed and why
1. Dead neurons: h1_2, h1_5, h1_7 are never positive anywhere in the input box (all-negative weights on
   positive inputs plus negative bias, or bias far below the reachable max). Removed.
2. Always-on neurons: h1_6 is active on ~100% of the box, h1_0 and h1_4 on ~98.5%; only h1_1 (68%) and h1_3 (40%)
   really switch. So the net is nearly linear per region; I computed the effective linear form
   W2 * diag(active) * W1 for each activation pattern.
3. Region structure: class 2 only occurs where h1_3 is on, class 0 only where h1_3 is off. Hence the argmax
   reduces to two pairwise boundaries: o1 - o0 (in the h1_3-off region) and o2 - o1 (in the h1_3-on region).
   Toggling h1_1 changes these boundaries only slightly.
4. Normalised each boundary so the x3 coefficient is 1 and grid-searched coefficients/thresholds on a 0.05
   lattice around the exact values, counting mismatches only on points where the net is confident
   (score >= checker margin). Both boundaries snapped to 0.05 multiples with 0 confident mismatches;
   x1 and x2 coefficients in the first boundary are equal and opposite, giving the `x2 - x1` term.
5. Threshold 1.35 for the first boundary left one confident mismatch in the corner (small x0, large x1/x2,
   tiny x3) where h1_1 tilts the boundary; 1.3 fixed it (both were valid on my own sample).

## Heuristics
- First find dead neurons (max of pre-activation over the input box < 0, via interval arithmetic) and
  always-on neurons (min > 0); the latter make the layer linear and can be folded into the next layer.
- Enumerate activation patterns on a big random sample, tabulate class counts per pattern; often one switching
  neuron separates which classes can co-occur, turning argmax into a decision list of pairwise boundaries.
- For argmax nets, compute pairwise differences o_i - o_j per pattern; those are the real decision surfaces.
- Normalise a boundary by its largest coefficient (or a natural one) before snapping; look for equal-magnitude
  pairs (x2 - x1) that suggest a combined feature.
- Snap with a small grid search scored only on confident points (the checker ignores the unsure zone), then
  prefer the coarsest lattice that still gives 0 errors. Coarse 0.1 lattices failed here; 0.05 worked.
- Steep boundaries (large scale of o_i - o_j) need more precise coefficients than shallow ones.
- When a single error remains, inspect where it is: usually a corner where a secondary neuron (here h1_1)
  tilts the boundary; shift the threshold toward the safe side rather than adding a branch.
- Validate with several checker seeds, not only the default one.
