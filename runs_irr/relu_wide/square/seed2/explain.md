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
    return 0.9998 * (x0 - 0.0002887) * (x0 - 0.0002887) - 0.004201
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.2%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1
  - largest error per slice: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2
- ripple (faithful minus net): largest 0.2%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: -0.1 -0.0 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1
  - largest error per slice: +0.2 +0.2 +0.2 +0.2 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| offset | 0 | -0.004201 | 0.11% |
| constant inside square | 0 | -0.0002887 | 0.03% |
| scale | 1 | 0.9998 | 0.02% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0798, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -0.154 | 0 | 0.46 | 0.54 | 0.664 | + | hinge |
| 1 | 0 | -7.59 | 0 | 0.00 | 1.00 | 0.0252 | - | constant |
| 2 | 0 | 1.27 | 0 | 0.82 | 0.18 | 0.354 | + | hinge |
| 3 | 180 | 8.97 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 0 | 0.0939 | 0 | 0.52 | 0.48 | 0.414 | + | hinge |
| 5 | 0 | 1.76 | 0 | 0.94 | 0.06 | 0.118 | + | hinge |
| 6 | 180 | 5.13 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 180 | 0.969 | 0 | 0.74 | 0.26 | 0.536 | + | hinge |
| 8 | 0 | 3.82 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 180 | 1.48 | 0 | 0.87 | 0.13 | 0.268 | + | hinge |
| 10 | 0 | -0.498 | 0 | 0.37 | 0.63 | 0.3 | + | hinge |
| 11 | 0 | -0.52 | 0 | 0.37 | 0.63 | 0.325 | + | hinge |
| 12 | 180 | 0.0195 | 0 | 0.50 | 0.50 | 0.491 | + | hinge |
| 13 | 0 | -0.475 | 0 | 0.38 | 0.62 | 0.0967 | + | hinge |
| 14 | 0 | -0.215 | 0 | 0.45 | 0.55 | 0.0354 | + | constant |
| 15 | 0 | 1.51 | 0 | 0.88 | 0.12 | 0.237 | + | hinge |
| 16 | 0 | 0.469 | 0 | 0.62 | 0.38 | 0.69 | + | hinge |
| 17 | 180 | 0.708 | 0 | 0.68 | 0.32 | 0.658 | + | hinge |
| 18 | 180 | -4.54 | 0 | 0.00 | 1.00 | 0.252 | - | line |
| 19 | 0 | -0.343 | 0 | 0.41 | 0.59 | 0.92 | + | hinge |
| 20 | 180 | -0.875 | 0 | 0.28 | 0.72 | 0.792 | + | hinge |
| 21 | 180 | -5.09 | 0 | 0.00 | 1.00 | 0.0517 | - | line |
| 22 | 180 | 3.08 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 23 | 0 | 0.695 | 0 | 0.67 | 0.33 | 0.572 | + | hinge |
| 24 | 180 | 2.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 0 | -2.62 | 0 | 0.00 | 1.00 | 0.0209 | - | constant |
| 26 | 180 | 1.23 | 0 | 0.81 | 0.19 | 0.397 | + | hinge |
| 27 | 180 | -2.93 | 0 | 0.00 | 1.00 | 0.483 | - | line |
| 28 | 180 | -0.987 | 0 | 0.25 | 0.75 | 0.567 | + | hinge |
| 29 | 180 | 1.74 | 0 | 0.94 | 0.06 | 0.131 | + | hinge |
| 30 | 180 | -0.249 | 0 | 0.44 | 0.56 | 0.921 | + | hinge |
| 31 | 180 | -1.08 | 0 | 0.23 | 0.77 | 0.707 | + | hinge |

## Groups

- negligible: units 3, 6, 8, 22, 24 (magnitude 0 together; a quarter of the tolerance is 0.02)
- lines: units 1, 14, 18, 21, 25, 27 (each a constant or a line on the box; together one linear term)
- same line: units 0, 4, 10, 11, 12, 13, 17, 19, 30 (gate angle 0 degrees, offset -0.154)
- same line: units 2, 20, 23, 28, 31 (gate angle 0 degrees, offset 1.27)
- the rest: units 5, 7, 9, 15, 16, 26, 29
