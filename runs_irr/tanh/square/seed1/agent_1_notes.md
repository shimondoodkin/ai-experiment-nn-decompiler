# Notes

## What the network computes

One input x0 in [-2, 2], regression. The output is a symmetric bowl: about 0.02 at x0 = 0,
about 0.98 at x0 = +-1, about 3.92 at x0 = +-2. It is close to `x0 * x0`, but the net is a bit
flatter at the edges (3.91 instead of 4 at x0 = 2), so `x0 * x0` alone fails the check
(max error 0.086 > tolerance 0.078). The best plain quadratic fit also fails (0.083), and a
quartic is borderline (0.073).

## What I changed

- **Six near-linear units folded away.** Units 0, 1, 2, 5, 6, 7 have slopes of about 0.03, so
  over [-2, 2] their pre-activations move by only +-0.07 and tanh is practically a line there.
  Together they give about 6.82 + 0.098 * x0 (max deviation from that line 0.007). The slope is
  cancelled by the asymmetry of the two real units, so only a constant remains; it merges with
  the output bias.
- **Two real units, made symmetric.** Units 3 and 4 are tanh steps with slopes -0.824 and -0.873
  at x0 = -1.84 and x0 = +1.81, weights +4.25 and -3.76. Using tanh(-z) = -tanh(z) they are two
  rising steps with opposite signs, i.e. a "bump" turned upside down. I snapped them to one shared
  slope 5/6, positions +-1.8 (offset 5/6 * 1.8 = 1.5), one shared weight 4, and a constant 7.25
  (from a small grid search over nice values, choosing the smallest max error).
- Result: `4 * (tanh(5/6*x0 - 1.5) - tanh(5/6*x0 + 1.5)) + 7.25`.
  `PASS (snap)`, max |program - net| = 0.028 (36% of the snap tolerance). Not exact: an exact
  version would need all 8 units (folding the six small ones already costs 0.007 > atol).

Readable meaning: a smooth approximation of x0^2 (a "soft square" that saturates toward the edges
of the domain), built as the difference of two tanh steps at +-1.8.

## Heuristics

- **Units with tiny input slope are lines.** If |w| * (input range / 2) is small (< ~0.1), the
  unit is linear over the whole domain; replace all such units by one fitted line `a + b*x` and
  fold `a` into the output bias. Check whether the slope `b` cancels against the remaining units
  before keeping it.
- **Plot / tabulate the output first.** A few sample values (here 0, 1, 4 at x = 0, 1, 2) often
  reveal the target (x^2), but test the simple formula against the tolerance before accepting it:
  nets saturate at the domain edges, and the edge error is usually what breaks a clean formula.
- **When the clean formula misses at the edges, keep the net's own basis.** Two opposite tanh
  steps at +-s give a smooth bowl or bump `A*(tanh(k(x-s)) - tanh(k(x+s))) + C`. Fit A, k, s, C,
  then snap to nice values with a small grid search over (A, k, s, C) and pick the lowest max error.
- **Symmetrize pairs.** Two units with near-equal |slope| and near-opposite positions usually
  encode an even (or odd) function; give them one shared slope, one shared position magnitude and
  one shared weight. Use tanh(-z) = -tanh(z) to put both in the same orientation first.
- **Prefer snapped k*s products that are nice too** (here 5/6 * 1.8 = 1.5), so the code reads as
  `tanh(k*x +- c)` with simple constants.
- **Error budget:** compute the snap tolerance (2% of output range) up front and aim for well under
  half of it, since the checker probes the domain corners and boundaries.
