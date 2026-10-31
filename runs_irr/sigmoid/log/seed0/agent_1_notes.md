# Agent notes

## What the network computes

`net(x0) ~= log(x0)` (natural log) on [0.1, 5].

Final program:

```
x = max(x0, 0.11)
return log(x)
```

Result: `PASS (snap)`, max |program - net| = 0.049 against a snap tolerance of 0.077.

## What I did

1. Evaluated the dump on a grid over [0.1, 5]. The values matched log(x) to 2-3 decimals:
   net(1) = -0.005, net(2) = 0.695, net(3) = 1.098, net(5) = 1.605. It is monotone and concave,
   with slope ~1/x, so no other candidate (sqrt and so on) fits.
2. The plain `log(x0)` FAILS, only just: max error 0.0772 against a tolerance of 0.0766, and every
   failure is at x0 in [0.1, 0.1001]. There the net gives -2.225, while log(0.1) = -2.303. Eight sigmoids
   cannot follow log's slope of 10 at the left edge, so the net levels off a little.
3. The residual net - log(x) is under 0.005 for x >= 1, and under 0.028 for x > 0.12. The only real error is
   that the net levels off at the edge. I modelled it with an input clamp, `log(max(x0, 0.11))`, which
   gives a max error of 0.049. `max(log(x0), -2.2)` works about as well (0.047). I also tried a shift,
   `log(x0 + 0.004)`, which gives 0.043, but it changes the formula everywhere and its constant means
   nothing. I picked the clamp because it is a guard on the domain edge and leaves the formula as it is.
4. All 8 hidden units and their weights are gone. The sigmoid sum is just a smooth fit of log.

## Heuristics

- For a 1-input regression net, first sample the output on a grid and compare it with a short list of
  standard functions (log, sqrt, 1/x, exp, x^2, sin). Use landmark values: f(1) ~ 0 suggests log, and
  f(4)/f(1) = 2 suggests sqrt. That usually finds the formula before you need to read any weights.
- When a clean formula only just fails, look at where the counterexamples are. If they all sit at a
  domain edge where the true function is steep (log or 1/x near 0, sqrt near 0), the net is underfitting
  there. A clamp on the input (`f(max(x, c))`) or on the output (`max(f(x), c)`) models that, with c a
  round number just inside the edge. Prefer this to bending the whole formula with shifts or scales.
- Compute the residual net - formula over the whole domain. If it is tiny everywhere except one end,
  fix that end and leave the formula alone.
- A sum of sigmoids with mixed signs and wide-ranging slopes (0.5 to 7.6 here) is usually a smooth fit
  of one curve, not a sum of meaningful parts. Steep units clustered at one end of the input range mark
  where the target curves most (here, the left end of log).
