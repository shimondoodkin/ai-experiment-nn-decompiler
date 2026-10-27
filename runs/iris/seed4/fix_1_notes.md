# What I changed and why

## The failures
The network gets two repair cases wrong, and both sit on the class-1/class-2 (versicolor/virginica) boundary:

| x0 | x1 | x2 | x3 | net | label | `large_petals` score |
|----|----|----|----|-----|-------|------------------------|
| 6.3 | 2.8 | 5.1 | 1.5 | 1 | 2 | -0.9 |
| 6.0 | 2.7 | 5.1 | 1.6 | 2 | 1 | +2.0 |

These two points are near twins with opposite labels. Both have long petals (5.1), and the one with the
*narrower* petal is the virginica. A linear programme over the repair data showed that no single linear
class-2 test separates classes 1 and 2 on all 100 cases. Removing either one of these two points makes the
data separable, but nothing else does. So re-weighting `large_petals` can fix only one of the two failures, and
fixing either one alone means a big tilt of the network's boundary (petal-length:width weight ratio from 0.57
to below 0.36, or above 0.58). That tilt would also change the answer far from these points.

## The fix
I kept both of the network's linear tests unchanged and added one rule, which applies only near the
network's own class-2 boundary:

```python
if x2 > 4.95 and x3 < 1.75 and abs(large_petals) < 3:   # long, not-wide petals, borderline score
    return 2 if x3 < 1.55 else 1                        # narrow -> virginica, medium -> versicolor
```

In words: when the petals are long (> 4.95) but not wide (< 1.75), and the linear score is close to zero,
petal width decides instead of the score. Width up to 1.55 means virginica, and 1.55 to 1.75 means
versicolor. This is the same split that a decision tree finds in the long-petal part of Iris-type data. In
the repair set it covers three cases: the two failures and the 6.0/2.2/5.0/1.5 virginica, which stays
correct. The `abs(large_petals) < 3` guard keeps the rule close to the boundary. Over the whole input box
the new programme differs from the network on about 2.7% of uniformly sampled points. Without that guard
it was about 11.6%.

## Result
`python score.py agent.py`: right on 100/100. Both network failures are fixed, and 0 of the 98 correct
answers are broken.

## Caveat
Because the two failing points are near twins with opposite labels, any rule that fixes both has to draw a
fine line between them: x3 = 1.5 against 1.6, at the same petal length. The thresholds 1.55 and 1.75 are
therefore supported by only a few cases. The rule is also non-monotone in petal width inside that small
region, which reflects the data rather than a clean geometric boundary.
