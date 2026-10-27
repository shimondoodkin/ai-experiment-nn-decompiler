# Notes

## What the network computes

The network takes one input `x0` in [-2, 2] and has 4 hidden ReLUs. It returns class 1 exactly when `|x0| < 1/2`. In other words, it is a band (interval) detector centred on 0.

Final program (`agent.py`):

```python
def net(x0):
    return 0.5 - abs(x0)
```

`python check.py agent.py` gives `PASS (snap)`: 0 disagreements where the net is confident. The only differences are near the upper edge, inside the unsure zone.

## How I got there

1. **Dead neuron.** `h1_0 = relu(-0.0075*x0 - 0.385)`. Its pre-activation is at most -0.37 on [-2, 2], so it is always 0. I removed it.
2. **Piecewise-linear regions.** I found where each remaining ReLU switches on or off:
   - `h1_1` is on for x0 > -0.233.
   - `h1_2` is on for x0 < -0.4406.
   - `h1_3` is on for x0 < -0.4391.

   `h1_2` and `h1_3` switch at almost the same point and both feed the output with negative weights. Together they act as one steep left-side ramp.
3. **Output on each region.**
   - Between -0.44 and -0.23 the output is the bias, +4.56, so the class is 1.
   - For x0 > -0.23 the output is `3.09 - 6.32*x0`. It crosses zero at x0 = 0.4891.
   - For x0 < -0.44 the output is about `37.6 + 75.2*x0`. It crosses zero at x0 = -0.5004.

   A numeric scan confirmed exactly two sign changes, at -0.50036 and +0.48912.
4. **Snapping.** The two edges are about -0.5 and +0.5, symmetric around 0. I snapped both to ±1/2, which turns the interval test into the single expression `0.5 - abs(x0)`. The upper edge moves by about 0.011, which is well inside the snap tolerance (0.02 × range 4 = 0.08).

The score's magnitude doesn't matter because the output is binary. That is why the slopes (6.3 on one side, 75 on the other) can all be replaced by a unit-slope `abs`.

## Heuristics

- **Remove dead and always-on units first.** Check each ReLU's pre-activation at the corners of the input box. If it is always ≤ 0, drop the unit. If it is always ≥ 0, fold it into the linear part.
- **For low-dimensional inputs, list the breakpoints.** With 1 or 2 inputs, sort the ReLU kinks. The function is linear between them, so solve for the zero crossings of the output on each piece. Cheaply confirm the crossings with a dense numeric scan.
- **Kinks at nearly the same point with same-sign output weights are one feature.** Merge them, for example a steep ramp built from 2 neurons.
- **For binary outputs, only the decision boundary matters.** Ignore slopes and magnitudes and work out where the sign changes. Then write the simplest expression with the same sign pattern.
- **An interval `a < x < b` is `min(x - a, b - x)`.** When the interval is symmetric around 0, it is `c - abs(x)`. Typical shapes are a band (two crossings, positive inside), a threshold (one crossing) or an outside-band (negative inside).
- **Snap boundaries to half-steps or simple fractions, and prefer symmetric forms** when the two edges are close to ± the same value. A short formula usually reflects the intended function better than the net's slightly off training-noise edges.
- **Check after each simplification.** A first attempt that misses a region shows up straight away as confident disagreements. In this case a first guess `0.5 - x0` missed the lower edge.
