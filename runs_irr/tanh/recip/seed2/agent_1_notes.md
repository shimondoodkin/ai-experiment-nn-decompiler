# Notes

## What I changed

The dump is a 1 -> 8 tanh -> 1 regression net on x0 in [0.5, 4]. I sampled it on a grid and
put the values next to a few candidate functions (1/x, sqrt, log):

| x0 | net | 1/x0 |
|----|-----|------|
| 0.5 | 1.9746 | 2.0 |
| 1 | 0.9935 | 1.0 |
| 2 | 0.4934 | 0.5 |
| 3 | 0.3296 | 0.3333 |
| 4 | 0.2402 | 0.25 |

The net is a fit of `1 / x0`. The final program is:

```python
def net(x0):
    return 1 / x0
```

`check.py`: PASS (snap), max |program - net| = 0.0254, against a snap tolerance of 0.0347
(2% of the output range of 1.734). The difference is the net's fitting error, not
something the formula leaves out:
- It is largest at the low edge, x0 = 0.5, where the net undershoots the steep part of 1/x
  (1.975 against 2.0).
- Elsewhere it is a small wiggle, mostly under 0.01, that changes sign (+0.005 at x0 = 0.6,
  -0.006 at 1 and 2, -0.010 at 4).

None of the eight tanh units means anything on its own. They are a smooth basis that
approximates the curve together, so I removed all of them. A PASS (exact) would mean
copying the wiggle, which is training noise, so I did not try for one.

## Heuristics

- For a 1-input regression net, first evaluate it on a grid and compare with simple
  functions (1/x, sqrt, log, exp, x^2, |x - c|) before you look at single units. With
  smooth (tanh/sigmoid) hidden layers the units are usually a basis, not features.
- A positive input range that starts away from 0 (here [0.5, 4]) and outputs that fall
  steeply and then flatten point to 1/x. Check f(1) = 1 and f(2x) = f(x)/2.
- The biggest fit error tends to be at the edge of the domain where the target is steepest.
  When the edge error stays under the snap tolerance, the clean formula is the right answer.
- Measure max |formula - net| against the snap tolerance (2% of the output range) yourself
  before you run the checker. That tells you at once whether a closed form is good enough or
  needs a correction term.
- Do not chase PASS (exact) when the leftover error is a small wiggle that changes sign. That
  is approximation noise from training, not structure.
