# Notes

## What the network computes

The dump is one tanh layer of 8 units on a single input x0 in [0, 4], output regression.
Tabulating it (scratch/look.py) shows output ~= sqrt(x0) almost everywhere: 0.99968 at 1,
1.41432 at 2, 1.73330 at 3, 1.99738 at 4 (errors of 1e-3 or less from x0 = 0.3 upward).
The only real deviation is near 0: the net gives 0.1428 at x0 = 0, where sqrt is 0, because
a sum of tanh units is smooth and cannot follow sqrt's infinite slope at 0.

## Final program (agent.py)

    root = sqrt(x0)
    near_zero = 0.125 + 2 * x0          # tangent of sqrt at x0 = 1/16
    return near_zero if x0 < 0.0625 else root

`1/8 + 2x` touches sqrt exactly at x = 1/16 (value 1/4, slope 2 on both), so the two pieces
join with matching value and slope. Check: PASS (snap), max |program - net| = 0.0178,
tolerance 0.0371 (2% of the output range). PASS (exact) is not reachable with a closed form;
it would mean keeping all 8 tanh units.

## Alternatives tried

- `sqrt(x0)` alone: FAIL, error 0.143 at x0 = 0.
- `sqrt(x0 + c)`: c = 1/80 passes (max error 0.0335, 90% of tolerance). It is shorter
  but has much less margin; c = 1/64 and 1/100 fail. I kept the tangent patch for margin.
- `max(sqrt(x0), line)`: wrong shape, since the line lies above the concave sqrt again for large x0.

## Heuristics

- For a 1-input regression net, tabulate the output on a grid and compare it against a short
  list of elementary functions (sqrt, log, log1p, exp, x^2, 1/x, |x|) before reading weights.
  Matching values at integer inputs (1, sqrt 2, sqrt 3, 2) identify the function fast.
- Smooth nets cannot follow singular slopes (sqrt at 0, log at 0, 1/x near 0). Expect the
  largest error at the domain edge where the target's derivative blows up, and patch only that
  edge instead of refitting the whole function.
- A good edge patch is the tangent line of the target at a simple point (here x = 1/16): it is
  continuous in value and slope, so the error stays small on both sides of the switch point.
  For sqrt, the tangent at x = a is sqrt(a)/2 + x/(2 sqrt(a)); a = 1/16 gives 1/8 + 2x.
- A shifted argument, f(x + c) with a simple c, is a one-term alternative for an edge offset;
  scan c over simple fractions and keep the one with the most tolerance margin.
- Do not take a max/min of a concave target with a line to patch an edge: the line can cross
  the curve again far from the edge. Use an explicit `if x < k` switch.
- Check margin, not only pass/fail: prefer the candidate whose max error is well below the
  snap tolerance, because the checker samples empirically.
