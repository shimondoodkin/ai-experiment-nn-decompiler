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
    return 1.001 * (x0 + 0.0007942) * (x0 + 0.0007942) - 0.003412
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.1%, inside the box (x0 = -0.794)
  - mean error in 10 equal slices of x0: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.0 -0.0 -0.1
  - largest error per slice: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.0 +0.0 +0.1
- ripple (faithful minus net): largest 0.2%, inside the box (x0 = 1.52)
  - mean error in 10 equal slices of x0: +0.0 -0.0 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.1
  - largest error per slice: +0.2 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.001 | 0.10% |
| offset | 0 | -0.003412 | 0.09% |
| constant inside square | 0 | 0.0007942 | 0.08% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.08, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 1.37 | 0 | 0.84 | 0.16 | 0.213 | + | hinge |
| 1 | 180 | 1.68 | 0 | 0.92 | 0.08 | 0.101 | + | hinge |
| 2 | 180 | 2.02 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | 1.2 | 0 | 0.80 | 0.20 | 0.272 | + | hinge |
| 4 | 180 | 4.63 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | 1.02 | 0 | 0.75 | 0.25 | 0.368 | + | hinge |
| 6 | 0 | 1.52 | 0 | 0.88 | 0.12 | 0.23 | + | hinge |
| 7 | 0 | 0.225 | 0 | 0.56 | 0.44 | 0.821 | + | hinge |
| 8 | 0 | 3.53 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 9 | 0 | 0.461 | 0 | 0.62 | 0.38 | 0.708 | + | hinge |
| 10 | 180 | 0.425 | 0 | 0.61 | 0.39 | 0.439 | + | hinge |
| 11 | 0 | 1.75 | 0 | 0.94 | 0.06 | 0.117 | + | hinge |
| 12 | 180 | -0.623 | 0 | 0.34 | 0.66 | 0.553 | + | hinge |
| 13 | 180 | 0.578 | 0 | 0.65 | 0.35 | 0.507 | + | hinge |
| 14 | 180 | 0.768 | 0 | 0.69 | 0.31 | 0.475 | + | hinge |
| 15 | 180 | 2.97 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 16 | 180 | 0.893 | 0 | 0.72 | 0.28 | 0.139 | + | hinge |
| 17 | 0 | -1.03 | 0 | 0.24 | 0.76 | 0.00128 | - | constant |
| 18 | 0 | 2.16 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 19 | 0 | 1.04 | 0 | 0.76 | 0.24 | 0.464 | + | hinge |
| 20 | 0 | 0.791 | 0 | 0.70 | 0.30 | 0.561 | + | hinge |
| 21 | 180 | -2.06 | 0 | 0.00 | 1.00 | 0.143 | - | line |
| 22 | 0 | -0.248 | 0 | 0.44 | 0.56 | 0.942 | + | hinge |
| 23 | 180 | -0.0765 | 0 | 0.48 | 0.52 | 0.196 | + | hinge |
| 24 | 180 | 1.53 | 0 | 0.88 | 0.12 | 0.142 | + | hinge |
| 25 | 180 | -0.914 | 0 | 0.27 | 0.73 | 0.0596 | + | hinge |
| 26 | 180 | 0.134 | 0 | 0.53 | 0.47 | 0.0527 | + | hinge |
| 27 | 0 | 2.23 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 0 | 2.38 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 29 | 180 | 0.0304 | 0 | 0.51 | 0.49 | 0.757 | + | hinge |
| 30 | 180 | 1.84 | 0 | 0.96 | 0.04 | 0.0486 | + | hinge |
| 31 | 0 | 1.28 | 0 | 0.82 | 0.18 | 0.345 | + | hinge |

## Groups

- negligible: units 2, 4, 8, 15, 17, 18, 27, 28 (magnitude 0.00128 together; a quarter of the tolerance is 0.02)
- lines: units 21 (each a constant or a line on the box; together one linear term)
- same line: units 0, 1, 3, 5, 7, 10, 13, 14, 16, 22, 23, 24, 26, 29, 30 (gate angle 180 degrees, offset 1.37)
- same line: units 9, 12, 19, 20, 25 (gate angle 0 degrees, offset 0.461)
- the rest: units 6, 11, 31
