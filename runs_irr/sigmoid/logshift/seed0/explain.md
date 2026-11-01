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
    return 1.025 * log(x0 + 2.036) - 0.03911
```

Clean:

```python
def net(x0):
    return log(x0 + 2)
```

% of the output range:
- distortion (clean minus faithful): largest 0.6%, at the box edge (x0 = -1.5)
  - mean error in 10 equal slices of x0: -0.1 +0.2 +0.2 +0.1 -0.0 -0.1 -0.2 -0.3 -0.4 -0.4
  - largest error per slice: +0.6 +0.2 +0.2 +0.1 +0.1 +0.2 +0.2 +0.3 +0.4 +0.5
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: +0.1 -0.3 -0.1 -0.0 -0.0 +0.1 +0.2 +0.3 +0.4 +0.4
  - largest error per slice: +0.5 +0.4 +0.3 +0.0 +0.0 +0.2 +0.3 +0.4 +0.4 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 2 | 2.036 | 2.83% |
| scale | 1 | 1.025 | 1.82% |
| offset | 0 | -0.03911 | 1.59% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0492, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1.28 | 9.32 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.207 | + | line |
| 1 | 0 | -0.269 | 4.44 | 0.00 | 0.00 | 0.53 | 0.47 | 0.00 | 0.306 | + | none (best 0.0205) |
| 2 | 180 | 1.79 | 1.68 | 0.75 | 0.19 | 0.06 | 0.00 | 0.00 | 0.462 | - | none (best 0.0598) |
| 3 | 180 | 1.79 | 1.65 | 0.75 | 0.19 | 0.06 | 0.00 | 0.00 | 0.393 | - | none (best 0.0489) |
| 4 | 0 | 2.77 | 8.05 | 0.00 | 0.23 | 0.77 | 0.00 | 0.00 | 0.23 | + | rounded hinge |
| 5 | 0 | 2.38 | 8.51 | 0.00 | 0.13 | 0.87 | 0.00 | 0.00 | 0.368 | + | none (best 0.0159) |
| 6 | 180 | 1.18 | 12.8 | 0.00 | 0.07 | 0.93 | 0.00 | 0.00 | 0.0685 | - | rounded hinge |
| 7 | 0 | -0.23 | 2.75 | 0.00 | 0.04 | 0.37 | 0.31 | 0.27 | 0.425 | + | none (best 0.0287) |

## Groups

- lines: units 0 (each a constant or a line on the box; together one linear term)
- same line: units 1, 7 (gate angle 0 degrees, offset -0.269)
- same line: units 2, 3 (gate angle 180 degrees, offset 1.79)
- the rest: units 4, 5, 6
