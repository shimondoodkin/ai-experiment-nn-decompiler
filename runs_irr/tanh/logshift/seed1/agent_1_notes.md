# Notes

## What I did

The dump is a 1 -> 8 tanh -> 1 regression net on x0 in [-1.5, 4]. I sampled it on a grid. The
output is monotone increasing and concave. It is close to 0 at x0 = -1, and its values at
x0 = 0, 1, 2 and 4 are 0.691, 1.099, 1.386 and 1.792, which are ln 2, ln 3, ln 4 and ln 6. So the net
is `log(x0 + 2)`.

`agent.py` is `return log(x0 + 2)`, and the checker gives **PASS (snap)**. The largest difference,
0.019, is at the left edge x0 = -1.5, where the true log curves down most steeply and the eight smooth
tanh units can't follow it. Everywhere else the difference is below about 0.002. The snap tolerance
is 0.049.

I did not try for PASS (exact). That would mean keeping almost all eight tanh units with their raw
weights, which is the dump again and not a readable program.

## Heuristics

- For a 1-input regression net, sample about 20 points first and compare the values with known
  functions before touching the weights. Check where the output is zero (here x0 = -1, so the
  shift is +1 from log's root) and where it equals ln 2, ln 3 and so on, or 1, 2, 3 for sqrt and
  squares.
- Concave, increasing and slowing down like a log means `log(x + c)`; find c from the zero crossing
  (`log(x + c) = 0` at x = 1 - c).
- Expect the biggest error at the domain edge nearest a singularity of the true function (log near
  its pole, sqrt near 0). Smooth tanh/sigmoid sums underfit there. That error usually stays inside
  the snap tolerance, so it doesn't mean the formula is wrong.
- A closed-form formula that passes snap beats a tanh sum that passes exact, for readability. Only
  chase exact when the net is piecewise linear or very small.
