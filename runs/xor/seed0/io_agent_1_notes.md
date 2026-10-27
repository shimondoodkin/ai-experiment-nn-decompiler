# Notes

The samples cover all 4 boolean input combinations. Output is strongly positive (~+10..+12)
exactly when x0 and x1 differ and strongly negative (~-11) when they are equal: this is XOR.
Wrote `net` as `x0 != x1` (returns the 0/1 decision). `check.py`: PASS (exact), exhaustive.

The raw scores are not symmetric (9.59 vs 11.91 for the two "true" cases), which is just training
noise in the magnitude; only the sign matters for a binary output, so I returned the decision rather
than trying to fit the score.

## Heuristics

- With few boolean inputs, enumerate the truth table from the samples first; match it to a named
  gate (AND/OR/XOR/NAND/implication) before fitting any arithmetic.
- For a binary output, ignore score magnitude asymmetries; return the decision when the score
  carries no meaningful structure.
- For bools, `a != b` is the cleanest XOR; `a == b` is XNOR.
