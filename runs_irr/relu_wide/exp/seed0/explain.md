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
    return 1.006 * exp(0.9976 * x0) - 0.007548
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.1%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.1 +0.1 +0.1 +0.0 +0.0 +0.0 -0.0 -0.0 -0.0 -0.0
  - largest error per slice: +0.1 +0.1 +0.1 +0.1 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 0.2%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.1 -0.0 -0.0 -0.1
  - largest error per slice: +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.006 | 0.61% |
| x0 inside exp | 1 | 0.9976 | 0.49% |
| offset | 0 | -0.007548 | 0.10% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.145, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 86.6 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 1.66 | 0 | 0.91 | 0.09 | 0.329 | + | hinge |
| 2 | 180 | 0.617 | 0 | 0.66 | 0.34 | 0.213 | + | hinge |
| 3 | 180 | 2 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | 180 | 2.89 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 5 | 0 | -1.98 | 0 | 0.00 | 1.00 | 1.11 | + | line |
| 6 | 180 | -1.76 | 0 | 0.06 | 0.94 | 0.101 | + | line |
| 7 | 0 | 0.287 | 0 | 0.57 | 0.43 | 0.647 | + | hinge |
| 8 | 0 | -0.00335 | 0 | 0.50 | 0.50 | 0.595 | + | hinge |
| 9 | 0 | 1.47 | 0 | 0.87 | 0.13 | 0.454 | + | hinge |
| 10 | 180 | -1.25 | 0 | 0.18 | 0.82 | 2.95e-05 | - | constant |
| 11 | 180 | 4.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 180 | 0.9 | 0 | 0.73 | 0.27 | 0.128 | + | hinge |
| 13 | 180 | 0.32 | 0 | 0.58 | 0.42 | 0.187 | - | hinge |
| 14 | 180 | -2.59 | 0 | 0.00 | 1.00 | 0.0105 | + | constant |
| 15 | 0 | 4.21 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 16 | 0 | 2.75 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 0 | 0.558 | 0 | 0.64 | 0.36 | 0.653 | + | hinge |
| 18 | 180 | -2 | 0 | 0.00 | 1.00 | 0.236 | - | line |
| 19 | 180 | -2 | 0 | 0.00 | 1.00 | 0.185 | - | line |
| 20 | 0 | -0.314 | 0 | 0.42 | 0.58 | 0.798 | + | hinge |
| 21 | 0 | 1.04 | 0 | 0.76 | 0.24 | 0.619 | + | hinge |
| 22 | 180 | -5.74 | 0 | 0.00 | 1.00 | 0.00537 | + | constant |
| 23 | 0 | 1.84 | 0 | 0.96 | 0.04 | 0.174 | + | hinge |
| 24 | 0 | -1.61 | 0 | 0.10 | 0.90 | 0.22 | + | line |
| 25 | 0 | 7.33 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 26 | 0 | 1.26 | 0 | 0.82 | 0.18 | 0.558 | + | hinge |
| 27 | 180 | 1.19 | 0 | 0.80 | 0.20 | 0.0691 | + | constant |
| 28 | 180 | -2 | 0 | 0.00 | 1.00 | 0.16 | - | line |
| 29 | 0 | -2.13 | 0 | 0.00 | 1.00 | 0.224 | + | line |
| 30 | 180 | 1.4 | 0 | 0.85 | 0.15 | 0.0241 | + | constant |
| 31 | 0 | 0.807 | 0 | 0.70 | 0.30 | 0.639 | + | hinge |

## Groups

- negligible: units 0, 3, 4, 10, 11, 14, 15, 16, 22, 25, 30 (magnitude 0.04 together; a quarter of the tolerance is 0.0362)
- lines: units 5, 6, 18, 19, 24, 27, 28, 29 (each a constant or a line on the box; together one linear term)
- same line: units 1, 9, 23 (gate angle 0 degrees, offset 1.66)
- same line: units 13, 20 (gate angle 180 degrees, offset 0.32)
- the rest: units 2, 7, 8, 12, 17, 21, 26, 31
