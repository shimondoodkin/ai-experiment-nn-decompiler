# Agent notes

## What the network computes

The dump is a 1-input, 32-unit ReLU net, so it is a piecewise-linear function of `x0` on [-2, 2].
I tabulated it (scratch/an.py) on a 0.1 grid and listed each unit's kink position `-b/w` and its
slope contribution `w*v`:

- f(0) = 1.0003, f(1) = 2.703, f(2) = 7.360, f(-1) = 0.3625, f(-2) = 0.1235.
- The function is convex and increasing, and its slope rises from about 0.17 at the left end to
  about 6.8 at the right end, roughly in proportion to the value itself.

Those values match e^x (1, 2.718, 7.389, 0.368, 0.135). The net is a piecewise-linear approximation
of `exp(x0)`, with about 25 active kinks spread over the domain.

## What I changed

`agent.py` is just `return exp(x0)`.

- Check result: `PASS (snap)`. The largest difference from the net is 0.029, at x0 = 2, where the
  net under-shoots e^2. The snap tolerance is 0.145.
- I did not try for `PASS (exact)`. An exact program has to reproduce every kink, including
  near-coincident ones (for example 0.1924, 0.1932 and 0.1964, or 1.7269, 1.7279 and 1.7286). That
  means keeping about 25 relu terms, which is the dump again with no added meaning.
- Dead or negligible units, which a piecewise version would drop:
  - unit 26 has weight about 1e-9;
  - units 4, 10, 13 and 15 have kinks outside [-2, 2] on their inactive side, or contribute
    less than 0.01 to the slope.

## Heuristics

- For a 1-input regression net, tabulate f on a coarse grid first and compare it with standard
  functions at landmark points (f(0), f(1), f(-1), the endpoints). Values like 1, 2.718 and 7.389
  identify exp at once. Look the same way for x^2, |x|, sqrt, log and sin.
- List the kinks `-b/w` with their slope changes `w*v`, sorted by position. If every slope change
  has the same sign and they are spread evenly over the domain, the net is a convex or concave
  smooth function approximated by a spline. Look for a closed-form smooth function, not a few
  linear pieces.
- A slope that grows in proportion to the value (f' ≈ f) means exp. f' ≈ 2x means a square, and
  f' ≈ 1/x means log.
- Check the closed form directly. The snap tolerance (2% of the output range) usually covers the
  net's own approximation error, and the largest error tends to sit at the domain ends, where the
  network was trained least densely or where curvature is largest.
- Units whose weights are about 1e-9, or whose kink lies outside the input box, contribute nothing
  and are the first to drop.
- Near-coincident kinks (within about 0.01) are the signature of a spline fit. Merging them keeps
  a snap pass, but rules out exact equality.
