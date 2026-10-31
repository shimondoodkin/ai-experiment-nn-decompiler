# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return sqrt(x0 + 0.01328)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.029 * sqrt(x0 + 0.03306) - 0.05506
```

Clean:

```python
def net(x0):
    return sqrt(x0 + 0.01328)
```

% of the output range:
- distortion (clean minus faithful): largest 1.1%, inside the box (x0 = 0.322)
  - mean error in 10 equal slices of x0: +0.9 +1.1 +0.9 +0.7 +0.5 +0.3 +0.1 -0.1 -0.2 -0.4
  - largest error per slice: +1.1 +1.1 +1.0 +0.8 +0.6 +0.4 +0.2 +0.1 +0.3 +0.4
- ripple (faithful minus net): largest 1.1%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: +0.3 -1.0 -0.6 -0.1 +0.0 +0.0 +0.1 +0.2 +0.4 +0.8
  - largest error per slice: +1.1 +1.0 +0.9 +0.3 +0.0 +0.0 +0.1 +0.3 +0.6 +1.1

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.01328 | 0.03306 | 3.62% |
| scale | 1 | 1.029 | 3.16% |
| offset | 0 | -0.05506 | 2.99% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0368, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.733 | 2.64 | 0.00 | 0.00 | 0.43 | 0.42 | 0.15 | 0.337 | + | none (best 0.0231) |
| 1 | 0 | 2.22 | 7.38 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.177 | + | rounded hinge |
| 2 | 0 | 0.16 | 8.71 | 0.00 | 0.00 | 0.86 | 0.14 | 0.00 | 0.19 | + | rounded hinge |
| 3 | 0 | 2.23 | 7.05 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.257 | + | rounded hinge |
| 4 | 180 | 0.198 | 1.03 | 0.79 | 0.16 | 0.05 | 0.00 | 0.00 | 0.464 | - | none (best 0.0575) |
| 5 | 0 | 1.96 | 7.27 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.102 | + | line |
| 6 | 0 | 2.91 | 6.88 | 0.00 | 0.08 | 0.92 | 0.00 | 0.00 | 0.184 | + | rounded hinge |
| 7 | 0 | 2.32 | 7.14 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.13 | + | rounded hinge |

## Groups

- lines: units 5 (each a constant or a line on the box; together one linear term)
- same line: units 1, 3, 7 (gate angle 0 degrees, offset 2.22)
- the rest: units 0, 2, 4, 6
