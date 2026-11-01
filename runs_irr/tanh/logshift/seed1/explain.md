# Explain

Inputs: x0 in [-1.5, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return log(x0 + 2)
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.024 * log(x0 + 2.032) - 0.03784
```

Clean:

```python
def net(x0):
    return log(x0 + 2)
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: +0.0 +0.3 +0.2 +0.1 +0.0 -0.1 -0.2 -0.2 -0.3 -0.4
  - largest error per slice: +0.4 +0.3 +0.3 +0.2 +0.1 +0.1 +0.2 +0.3 +0.4 +0.4
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = -1.5)
  - mean error in 10 equal slices of x0: +0.0 -0.3 -0.1 -0.1 -0.0 +0.1 +0.2 +0.3 +0.3 +0.4
  - largest error per slice: +0.4 +0.3 +0.2 +0.1 +0.1 +0.1 +0.2 +0.3 +0.3 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 2 | 2.032 | 2.52% |
| scale | 1 | 1.024 | 1.74% |
| offset | 0 | -0.03784 | 1.53% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0493, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -1.61 | 1.07 | 0.00 | 0.00 | 0.05 | 0.12 | 0.82 | 0.417 | + | none (best 0.0475) |
| 1 | 180 | 1.66 | 2.02 | 0.66 | 0.23 | 0.11 | 0.00 | 0.00 | 0.237 | - | none (best 0.0288) |
| 2 | 0 | -2.92 | 14.8 | 0.00 | 0.00 | 0.75 | 0.25 | 0.00 | 0.0759 | + | rounded hinge |
| 3 | 0 | -0.942 | 2.77 | 0.00 | 0.00 | 0.29 | 0.32 | 0.39 | 0.481 | + | none (best 0.04) |
| 4 | 180 | -1.1 | 7.85 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.396 | - | rounded hinge |
| 5 | 0 | 3.51 | 8.85 | 0.00 | 0.31 | 0.69 | 0.00 | 0.00 | 0.226 | + | none (best 0.0127) |
| 6 | 180 | 0.28 | 4.53 | 0.00 | 0.47 | 0.53 | 0.00 | 0.00 | 0.465 | - | none (best 0.0311) |
| 7 | 0 | 3.98 | 12.1 | 0.00 | 0.17 | 0.83 | 0.00 | 0.00 | 0.169 | + | rounded hinge |

## Groups

- same line: units 0, 1 (gate angle 0 degrees, offset -1.61)
- the rest: units 2, 3, 4, 5, 6, 7
