# Notes

The samples cover all 4 boolean input combinations. Output is negative (about -13) for (0,0) and (1,1)
and positive (about +12..13) for (0,1) and (1,0): the decision is exclusive-or. I wrote
`net` as `x0 != x1` (returns the 0/1 decision). Checker: PASS (exact), exhaustive over 4 inputs.
The raw score magnitudes (about +-12..13, slightly asymmetric) are training artifacts and were not
reproduced, since for a binary output the decision is sufficient.

## Heuristics

- For all-boolean inputs with few inputs, the samples are often the full truth table: read the
  decision column directly and match it to a named boolean function (AND, OR, XOR, majority, ...).
- For binary outputs, return the 0/1 decision rather than fitting raw scores whose magnitudes are
  just large-margin noise (here roughly symmetric +-12.5).
- XOR of two booleans is most readably `a != b`; parity of more inputs can be `(a + b + c) == 1 or ... `
  or chained `!=`.
- Check the exact decision first with the simplest named formula before fitting coefficients.
