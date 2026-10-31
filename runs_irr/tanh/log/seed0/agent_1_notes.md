# Agent notes

## What I did

- The dump is a 1-input, 8-unit tanh net giving a regression output on x0 in [0.1, 5].
- I tabulated the net on a 50-point grid next to the usual 1-input candidates (log, sqrt, 1/x).
  The output matched log(x0) to about two decimals everywhere: -0.69 at 0.5, about 0 at 1, 0.69 at 2,
  1.10 at 3, 1.38 at 4 and 1.60 at 5.
- Final program: `return log(x0)`. No coefficients or offsets are left to snap. All eight tanh units together
  approximate a single log curve, so none of them has its own meaning.
- `check.py`: **PASS (snap)**, empirical, 208k inputs, max |program - net| = 0.066 against a snap tolerance
  of 0.077. The largest error is at the left edge (x0 = 0.1), where log is steepest and the net levels off
  (net -2.237, log -2.303). For x0 > 0.2 the error is at most 0.023, with a mean of -0.005.
- PASS (exact) is not realistic here. A sum of 8 tanh units cannot equal log to within 1e-5, and fitting
  a + b·log(x) or extra correction terms would only reproduce the net's approximation error, not the function.

## Heuristics

- For 1-input regression nets, first tabulate the net on a grid next to standard functions (log, exp,
  sqrt, 1/x, x², sin) over the given domain. Reading values at x = 0.5, 1, 2 and e quickly identifies log
  (0 at 1, symmetric values at 0.5 and 2).
- A positive domain that starts slightly above 0 (here 0.1) hints at log, 1/x or sqrt: functions whose
  domain or behaviour breaks at 0.
- Many smooth tanh/sigmoid units with mixed signs and no clear role usually build one smooth curve
  together. Look at the whole input-output curve instead of simplifying unit by unit.
- Expect the largest deviation where the target is steepest or most curved, usually at a domain edge. A
  large edge error with a small interior error means the net is under-fitting there, and the formula
  should still be the clean function.
- Once a clean closed form passes the snap check, stop. Do not add fitted correction terms to chase the
  net's approximation error.
