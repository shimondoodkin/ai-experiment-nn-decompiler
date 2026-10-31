# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (7 nodes):

```python
def net(x0):
    return x0 * x0
```

## Faithful fit

The square formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no square formula follows).

Faithful:

```python
def net(x0):
    return 0.9994 * (x0 - 0.001183) * (x0 - 0.001183) - 0.00546
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.3%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2 +0.3
  - largest error per slice: +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2 +0.3 +0.3
- ripple (faithful minus net): largest 0.3%, inside the box (x0 = -1.09)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.1 -0.1 -0.0 -0.1 -0.1 -0.2 -0.1 -0.1
  - largest error per slice: +0.3 +0.3 +0.3 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| offset | 0 | -0.00546 | 0.14% |
| constant inside square | 0 | -0.001183 | 0.12% |
| scale | 1 | 0.9994 | 0.06% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0799, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -2.16 | 0 | 0.00 | 1.00 | 0.401 | - | line |
| 1 | 180 | 2.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 4.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 0 | 0.467 | 0 | 0.62 | 0.38 | 0.775 | + | hinge |
| 4 | 180 | -0.778 | 0 | 0.30 | 0.70 | 2.38e-10 | + | constant |
| 5 | 0 | 1.48 | 0 | 0.87 | 0.13 | 0.267 | + | hinge |
| 6 | 180 | -1.07 | 0 | 0.23 | 0.77 | 0.0483 | + | line |
| 7 | 0 | 1.74 | 0 | 0.94 | 0.06 | 0.133 | + | hinge |
| 8 | 180 | -538 | 0 | 0.00 | 1.00 | 0.000518 | - | constant |
| 9 | 180 | 3.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 0 | -3.76 | 0 | 0.00 | 1.00 | 0.0751 | - | line |
| 11 | 0 | -0.314 | 0 | 0.42 | 0.58 | 1.06 | + | hinge |
| 12 | 0 | 0.954 | 0 | 0.74 | 0.26 | 0.538 | + | hinge |
| 13 | 180 | 0.0669 | 0 | 0.52 | 0.48 | 1.03 | + | hinge |
| 14 | 180 | 7.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 180 | 6.09 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 16 | 0 | 4.11 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | 27.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | 0 | 1.22 | 0 | 0.81 | 0.19 | 0.404 | + | hinge |
| 19 | 0 | 2.47 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 20 | 180 | -0.202 | 0 | 0.45 | 0.55 | 1.19 | + | hinge |
| 21 | 180 | 1.39 | 0 | 0.85 | 0.15 | 0.365 | + | hinge |
| 22 | 0 | -5.59 | 0 | 0.00 | 1.00 | 0.33 | - | line |
| 23 | 180 | -0.707 | 0 | 0.32 | 0.68 | 1.28 | + | hinge |
| 24 | 180 | 1.7 | 0 | 0.92 | 0.08 | 0.182 | + | hinge |
| 25 | 0 | 13.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 26 | 0 | -0.532 | 0 | 0.37 | 0.63 | 1.05 | + | hinge |
| 27 | 0 | -0.902 | 0 | 0.27 | 0.73 | 0.171 | + | hinge |
| 28 | 180 | 1.09 | 0 | 0.77 | 0.23 | 0.536 | + | hinge |
| 29 | 0 | -0.813 | 0 | 0.30 | 0.70 | 1.15 | + | hinge |
| 30 | 0 | -2 | 0 | 0.00 | 1.00 | 0.553 | - | line |
| 31 | 0 | -0.674 | 0 | 0.33 | 0.67 | 0.433 | + | hinge |

## Groups

- negligible: units 1, 2, 4, 8, 9, 14, 15, 16, 17, 19, 25 (magnitude 0.000518 together; a quarter of the tolerance is 0.02)
- lines: units 0, 6, 10, 22, 30 (each a constant or a line on the box; together one linear term)
- same line: units 26, 27, 28, 29, 31 (gate angle 0 degrees, offset -0.532)
- the rest: units 3, 5, 7, 11, 12, 13, 18, 20, 21, 23, 24
