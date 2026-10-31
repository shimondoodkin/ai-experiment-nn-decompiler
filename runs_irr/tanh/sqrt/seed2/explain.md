# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return sqrt(x0 + 0.01)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.026 * sqrt(x0 + 0.02384) - 0.0462
```

Clean:

```python
def net(x0):
    return sqrt(x0 + 0.01)
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, inside the box (x0 = 0.252)
  - mean error in 10 equal slices of x0: +0.9 +0.9 +0.7 +0.5 +0.3 +0.1 -0.0 -0.2 -0.3 -0.4
  - largest error per slice: +1.0 +1.0 +0.8 +0.6 +0.4 +0.2 +0.1 +0.2 +0.4 +0.5
- ripple (faithful minus net): largest 0.8%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: -0.0 -0.7 -0.3 -0.1 +0.0 +0.1 +0.2 +0.3 +0.5 +0.7
  - largest error per slice: +0.8 +0.8 +0.5 +0.2 +0.1 +0.2 +0.3 +0.4 +0.6 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.01 | 0.02384 | 2.91% |
| scale | 1 | 1.026 | 2.79% |
| offset | 0 | -0.0462 | 2.47% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0374, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -0.155 | 3.73 | 0.00 | 0.00 | 0.31 | 0.59 | 0.11 | 0.373 | + | none (best 0.0334) |
| 1 | 180 | 0.502 | 1.71 | 0.70 | 0.27 | 0.03 | 0.00 | 0.00 | 0.203 | - | none (best 0.0264) |
| 2 | 0 | 1.87 | 10.7 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.15 | + | line |
| 3 | 0 | -0.291 | 9.95 | 0.00 | 0.00 | 0.86 | 0.14 | 0.00 | 0.413 | + | none (best 0.0178) |
| 4 | 0 | 2.64 | 12.4 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0476 | + | line |
| 5 | 0 | -0.122 | 0.583 | 0.00 | 0.00 | 0.02 | 0.09 | 0.88 | 0.226 | + | none (best 0.0372) |
| 6 | 180 | -2.7 | 11 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.322 | - | rounded hinge |
| 7 | 180 | -3.24 | 8.14 | 0.00 | 0.00 | 0.95 | 0.05 | 0.00 | 0.135 | - | rounded hinge |

## Groups

- lines: units 2, 4 (each a constant or a line on the box; together one linear term)
- same line: units 0, 3, 5 (gate angle 0 degrees, offset -0.155)
- the rest: units 1, 6, 7
