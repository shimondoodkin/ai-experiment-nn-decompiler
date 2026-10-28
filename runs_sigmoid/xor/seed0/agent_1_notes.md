# Notes

## What I did
- The net has 2 boolean inputs, one hidden layer of 4 sigmoid units, and a binary output (class 1 if score > 0).
- With only 4 possible inputs, I evaluated the dump on the full truth table:
  (0,0) -> -10.64, (0,1) -> +8.56, (1,0) -> +8.57, (1,1) -> -7.97.
  All margins are large (|score| > 7), so the decision is unambiguous: class 1 exactly when x0 != x1, i.e. XOR.
- Structure check: the hidden units are symmetric in x0/x1 (coefficient pairs nearly equal), so each unit
  depends only on the count x0 + x1. Unit 0 is an AND detector (fires at count 2, bias ~ -1.5 x weight),
  unit 3 is an OR detector (fires at count >= 1), units 1 and 2 are NOR detectors (fire at count 0).
  Output = OR - AND - NOR + const, which is the classic XOR construction.
- Final program: `return x0 != x1` (comparison gives 1/0; 1 > 0 is class 1, 0 is class 0).
  `check.py` reports PASS (exact), exhaustive over 4 inputs.

## Heuristics
- For boolean (or small discrete) input domains, enumerate the whole truth table first; the dump's score
  at each point often identifies the logic function directly, and the margins tell whether any point is borderline.
- For binary outputs, the exact score values do not matter, only the sign relative to the threshold;
  replace the whole network with the boolean expression and return a comparison (1/0) or +-1.
- Nearly equal input coefficients in a unit mean it depends on the symmetric sum (count of true inputs);
  then the unit is a threshold on that count: bias/weight between -1 and 0 -> OR, between -2 and -1 -> AND,
  negative weights with positive bias -> NOR/NAND.
- Redundant hidden units (here two NOR detectors with the same role) are common in small sigmoid nets;
  they merge into one term once the function is identified.
- An OR unit minus an AND unit with a strongly negative bias/NOR correction is the usual XOR pattern.
