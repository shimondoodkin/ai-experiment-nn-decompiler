# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return sqrt(x0)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.006 * sqrt(x0 + 0.001295) - 0.00798
```

Clean:

```python
def net(x0):
    return sqrt(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 1.4%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: +0.2 +0.1 +0.1 +0.0 -0.0 -0.1 -0.1 -0.1 -0.2 -0.2
  - largest error per slice: +1.4 +0.2 +0.1 +0.0 +0.0 +0.1 +0.1 +0.2 +0.2 +0.2
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = 0.01)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.1 -0.1 -0.0 +0.1 +0.1 +0.1 +0.1 +0.2
  - largest error per slice: +0.6 +0.3 +0.2 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.001295 | 1.83% |
| scale | 1 | 1.006 | 0.61% |
| offset | 0 | -0.00798 | 0.41% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0393, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 86.6 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 2.92 | 0 | 0.73 | 0.27 | 0.0287 | - | hinge |
| 2 | 180 | 0.849 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | 1.27 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | -3.21 | 0 | 0.00 | 1.00 | 0.404 | + | line |
| 6 | 180 | -1.79 | 0 | 0.55 | 0.45 | 0.0761 | - | hinge |
| 7 | 0 | 2.67 | 0 | 0.67 | 0.33 | 0.000126 | - | constant |
| 8 | 180 | -0.107 | 0 | 0.97 | 0.03 | 0.0845 | - | hinge |
| 9 | 0 | 2.52 | 0 | 0.63 | 0.37 | 0.0286 | - | hinge |
| 10 | 180 | -0.0274 | 0 | 0.99 | 0.01 | 0.0805 | - | hinge |
| 11 | 180 | 4.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 180 | 0.756 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | 180 | 0.778 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 14 | 180 | -2.33 | 0 | 0.42 | 0.58 | 0.00386 | - | constant |
| 15 | 180 | -2.17 | 0 | 0.46 | 0.54 | 0.026 | - | hinge |
| 16 | 0 | 3.43 | 0 | 0.86 | 0.14 | 0.0111 | - | constant |
| 17 | 0 | -0.141 | 0 | 0.00 | 1.00 | 0.634 | + | line |
| 18 | 180 | -0.483 | 0 | 0.88 | 0.12 | 0.141 | - | hinge |
| 19 | 180 | -1.4 | 0 | 0.65 | 0.35 | 0.0736 | - | hinge |
| 20 | 0 | -1.64 | 0 | 0.00 | 1.00 | 0.12 | + | line |
| 21 | 0 | 0.135 | 0 | 0.03 | 0.97 | 0.0203 | + | line |
| 22 | 180 | -2.12 | 0 | 0.47 | 0.53 | 0.0155 | - | constant |
| 23 | 0 | 1.12 | 0 | 0.28 | 0.72 | 0.167 | - | hinge |
| 24 | 180 | -0.791 | 0 | 0.80 | 0.20 | 0.0993 | - | hinge |
| 25 | 0 | 7.33 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 26 | 0 | 0.487 | 0 | 0.12 | 0.88 | 0.314 | + | hinge |
| 27 | 180 | 0.557 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 180 | -0.247 | 0 | 0.94 | 0.06 | 0.0915 | - | hinge |
| 29 | 180 | -2.09 | 0 | 0.48 | 0.52 | 0.0151 | - | constant |
| 30 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 31 | 0 | 4.66 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 0, 2, 3, 4, 7, 11, 12, 13, 14, 25, 27, 30, 31 (magnitude 0.00398 together; a quarter of the tolerance is 0.00982)
- lines: units 5, 16, 17, 20, 21, 22, 29 (each a constant or a line on the box; together one linear term)
- same line: units 8, 10, 28 (gate angle 180 degrees, offset -0.107)
- same line: units 18, 26 (gate angle 180 degrees, offset -0.483)
- the rest: units 1, 6, 9, 15, 19, 23, 24
