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
    return 0.9996 * (x0 + 9.095e-05) * (x0 + 9.095e-05) - 0.002585
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.1%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1
  - largest error per slice: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1
- ripple (faithful minus net): largest 0.2%, inside the box (x0 = 1.15)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.1
  - largest error per slice: +0.2 +0.2 +0.2 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| offset | 0 | -0.002585 | 0.06% |
| scale | 1 | 0.9996 | 0.04% |
| constant inside square | 0 | 9.095e-05 | 0.01% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0798, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 86.6 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 1.76 | 0 | 0.94 | 0.06 | 0.117 | + | hinge |
| 2 | 180 | 1.09 | 0 | 0.77 | 0.23 | 0.412 | + | hinge |
| 3 | 180 | 1.78 | 0 | 0.94 | 0.06 | 0.0984 | + | hinge |
| 4 | 180 | 1.31 | 0 | 0.83 | 0.17 | 0.296 | + | hinge |
| 5 | 0 | -1.43 | 0 | 0.14 | 0.86 | 0.202 | + | hinge |
| 6 | 180 | -4.41 | 0 | 0.00 | 1.00 | 0.0544 | - | line |
| 7 | 0 | -0.191 | 0 | 0.45 | 0.55 | 0.492 | + | hinge |
| 8 | 180 | 0.449 | 0 | 0.61 | 0.39 | 0.45 | + | hinge |
| 9 | 0 | 1.52 | 0 | 0.88 | 0.12 | 0.232 | + | hinge |
| 10 | 180 | 0.0675 | 0 | 0.52 | 0.48 | 0.526 | + | hinge |
| 11 | 180 | 4.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 180 | 0.863 | 0 | 0.72 | 0.28 | 0.519 | + | hinge |
| 13 | 180 | 1.56 | 0 | 0.89 | 0.11 | 0.191 | + | hinge |
| 14 | 180 | -0.858 | 0 | 0.28 | 0.72 | 0.0893 | + | hinge |
| 15 | 180 | -9.01 | 0 | 0.00 | 1.00 | 0.0278 | - | constant |
| 16 | 0 | 2.67 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 0 | 0.299 | 0 | 0.57 | 0.43 | 0.695 | + | hinge |
| 18 | 180 | -0.834 | 0 | 0.29 | 0.71 | 0.642 | + | hinge |
| 19 | 180 | -2.57 | 0 | 0.00 | 1.00 | 0.56 | - | line |
| 20 | 0 | -0.313 | 0 | 0.42 | 0.58 | 0.623 | + | hinge |
| 21 | 0 | 0.651 | 0 | 0.66 | 0.34 | 0.599 | + | hinge |
| 22 | 0 | -10.6 | 0 | 0.00 | 1.00 | 0.00309 | - | constant |
| 23 | 0 | 1.27 | 0 | 0.82 | 0.18 | 0.359 | + | hinge |
| 24 | 180 | -0.0346 | 0 | 0.49 | 0.51 | 0.281 | + | hinge |
| 25 | 0 | 7.33 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 26 | 0 | 1.03 | 0 | 0.76 | 0.24 | 0.469 | + | hinge |
| 27 | 180 | 0.635 | 0 | 0.66 | 0.34 | 0.605 | + | hinge |
| 28 | 180 | -0.466 | 0 | 0.38 | 0.62 | 0.687 | + | hinge |
| 29 | 180 | -0.807 | 0 | 0.30 | 0.70 | 0.0999 | + | hinge |
| 30 | 180 | 2.82 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 31 | 0 | 0.126 | 0 | 0.53 | 0.47 | 0.472 | + | hinge |

## Groups

- negligible: units 0, 11, 16, 22, 25, 30 (magnitude 0.00309 together; a quarter of the tolerance is 0.0199)
- lines: units 6, 15, 19 (each a constant or a line on the box; together one linear term)
- same line: units 4, 5, 13 (gate angle 180 degrees, offset 1.31)
- same line: units 7, 8, 10, 14, 17, 18, 20, 21, 24, 26, 27, 28, 29, 31 (gate angle 0 degrees, offset -0.191)
- the rest: units 1, 2, 3, 9, 12, 23
