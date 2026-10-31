# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return exp(x0)
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0):
    return 0.9915 * exp(1.003 * x0) + 0.01244
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.1%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.0 +0.0 +0.1 +0.1
  - largest error per slice: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.0 +0.1 +0.1 +0.1
- ripple (faithful minus net): largest 0.2%, at the box edge (x0 = 1.92)
  - mean error in 10 equal slices of x0: +0.1 +0.1 +0.1 +0.1 +0.0 +0.0 -0.0 -0.0 -0.1 -0.1
  - largest error per slice: +0.2 +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.9915 | 0.87% |
| x0 inside exp | 1 | 1.003 | 0.61% |
| offset | 0 | 0.01244 | 0.17% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.145, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -0.404 | 0 | 0.40 | 0.60 | 0.671 | + | hinge |
| 1 | 0 | -2.64 | 0 | 0.00 | 1.00 | 0.0815 | + | line |
| 2 | 0 | 1.08 | 0 | 0.77 | 0.23 | 0.589 | + | hinge |
| 3 | 180 | 8.97 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 0 | 0.85 | 0 | 0.71 | 0.29 | 0.634 | + | hinge |
| 5 | 0 | 1.48 | 0 | 0.87 | 0.13 | 0.435 | + | hinge |
| 6 | 180 | -0.159 | 0 | 0.46 | 0.54 | 2.33e-07 | - | constant |
| 7 | 180 | 0.325 | 0 | 0.58 | 0.42 | 0.832 | - | hinge |
| 8 | 0 | 3.86 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 180 | 2 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 0 | -0.945 | 0 | 0.26 | 0.74 | 0.385 | + | hinge |
| 11 | 0 | -1.49 | 0 | 0.13 | 0.87 | 0.349 | + | hinge |
| 12 | 180 | 2.16 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | 0 | 0.0359 | 0 | 0.51 | 0.49 | 0.61 | + | hinge |
| 14 | 0 | -0.671 | 0 | 0.33 | 0.67 | 0.298 | + | hinge |
| 15 | 0 | 1.29 | 0 | 0.82 | 0.18 | 0.526 | + | hinge |
| 16 | 0 | 0.326 | 0 | 0.58 | 0.42 | 0.658 | + | hinge |
| 17 | 180 | -1.85 | 0 | 0.04 | 0.96 | 3.57 | + | hinge |
| 18 | 180 | -2 | 0 | 0.00 | 1.00 | 0.777 | - | line |
| 19 | 0 | -0.289 | 0 | 0.43 | 0.57 | 1.22 | + | hinge |
| 20 | 180 | -2 | 0 | 0.00 | 1.00 | 1.64 | - | line |
| 21 | 180 | 2.08 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 22 | 180 | 3.08 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 23 | 0 | 0.601 | 0 | 0.65 | 0.35 | 0.675 | + | hinge |
| 24 | 180 | 2.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 0 | -1.2 | 0 | 0.20 | 0.80 | 0.155 | + | hinge |
| 26 | 180 | -2 | 0 | 0.00 | 1.00 | 1.05 | - | line |
| 27 | 180 | -2 | 0 | 0.00 | 1.00 | 1.03 | - | line |
| 28 | 180 | -2 | 0 | 0.00 | 1.00 | 1.75 | - | line |
| 29 | 180 | 2.23 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 30 | 180 | -1.68 | 0 | 0.08 | 0.92 | 3.99 | + | hinge |
| 31 | 180 | -2.07 | 0 | 0.00 | 1.00 | 0.426 | - | line |

## Groups

- negligible: units 3, 6, 8, 9, 12, 21, 22, 24, 29 (magnitude 2.33e-07 together; a quarter of the tolerance is 0.0362)
- lines: units 1, 18, 20, 26, 27, 28, 31 (each a constant or a line on the box; together one linear term)
- same line: units 0, 7, 19 (gate angle 0 degrees, offset -0.404)
- same line: units 5, 15, 17, 30 (gate angle 0 degrees, offset 1.48)
- the rest: units 2, 4, 10, 11, 13, 14, 16, 23, 25
