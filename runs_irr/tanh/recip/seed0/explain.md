# Explain

Inputs: x0 in [0.5, 4]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return 1 / x0
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 1.045 / (x0 + 0.0212) - 0.02068
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.9%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.2 -0.1 -0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.6
  - largest error per slice: +0.9 +0.2 +0.1 +0.2 +0.3 +0.4 +0.5 +0.5 +0.6 +0.6
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = 0.603)
  - mean error in 10 equal slices of x0: -0.1 +0.5 +0.1 -0.1 -0.1 -0.1 -0.3 -0.5 -0.5 -0.4
  - largest error per slice: +0.6 +0.5 +0.3 +0.1 +0.1 +0.2 +0.4 +0.5 +0.5 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.045 | 5.20% |
| constant inside reciprocal | 0 | 0.0212 | 4.70% |
| offset | 0 | -0.02068 | 1.20% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0346, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.206 | 0.923 | 0.82 | 0.17 | 0.01 | 0.00 | 0.00 | 0.547 | + | none (best 0.0746) |
| 1 | 0 | -1.37 | 2.48 | 0.00 | 0.00 | 0.00 | 0.18 | 0.82 | 0.052 | + | rounded hinge |
| 2 | 180 | 1.02 | 1.49 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0271 | - | rounded hinge |
| 3 | 180 | 1.11 | 1.75 | 0.96 | 0.04 | 0.00 | 0.00 | 0.00 | 0.0283 | - | rounded hinge |
| 4 | 0 | 0.984 | 2.17 | 0.00 | 0.00 | 0.37 | 0.39 | 0.24 | 0.661 | - | none (best 0.0494) |
| 5 | 0 | 3.67 | 8.86 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.215 | - | rounded hinge |
| 6 | 0 | 0.185 | 0.997 | 0.00 | 0.00 | 0.02 | 0.18 | 0.80 | 0.46 | - | none (best 0.0628) |
| 7 | 0 | -0.705 | 1.77 | 0.00 | 0.00 | 0.00 | 0.16 | 0.84 | 0.046 | + | rounded hinge |

## Groups

- same line: units 0, 6 (gate angle 180 degrees, offset -0.206)
- same line: units 2, 3 (gate angle 180 degrees, offset 1.02)
- the rest: units 1, 4, 5, 7
