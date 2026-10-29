# Notes

## What I changed

- The net has 2 bool inputs, so there are only 4 inputs. I evaluated the dump on all of them
  (`scratch/tt.py`):

  | x0 | x1 | score |
  |----|----|-------|
  | 0  | 0  | -8.19 |
  | 0  | 1  | +7.96 |
  | 1  | 0  | +10.66 |
  | 1  | 1  | -8.23 |

  Every score is far from the threshold 0 (|score| > 7.9), so there is no ambiguous case.
- The truth table is XOR. I replaced the four sigmoid units and the output layer with
  `return x0 != x1` (a comparison gives 1/0, and 1 > 0 is class 1).
- Check result: `PASS (exact)`, exhaustive over the 4 inputs.

Reading the units (not needed for the result, but it agrees): h1_0 and h1_1 are near-copies, both
approximately NAND(x0, x1) (on unless both inputs are 1); h1_2 is NOR (on only at 0,0) and has a
large negative output weight; h1_3 is "x0 and not x1". Together: NAND minus NOR, i.e. OR and not
AND, which is XOR.

## Heuristics

- With all-bool inputs and a small input count (2^n up to a few thousand), evaluate the dump on
  every input and read the truth table first. Name the Boolean function (AND, OR, XOR, majority,
  threshold count, ...) and write it directly; this beats simplifying the weights unit by unit.
- For binary outputs, check the margin: if every |score| is well above the threshold, the
  Boolean formula is the whole function and an exact pass is expected.
- Sigmoid units with large weights on 0/1 inputs act as Boolean gates. Read each one as a gate by
  checking the sign of its pre-activation at each corner. Near-duplicate units (similar weights and
  bias) are one gate whose output weights add up.
- XOR of two bools is `x0 != x1`; XNOR is `x0 == x1`. Comparisons return 1/0, which works directly
  as a binary output with threshold 0.
