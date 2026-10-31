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
    return 1.007 * exp(0.9979 * x0) - 0.005888
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.2%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.1 +0.0 +0.0 +0.0 -0.0 -0.0 -0.1 -0.1 -0.1 -0.2
  - largest error per slice: +0.1 +0.1 +0.0 +0.0 +0.0 +0.0 +0.1 +0.1 +0.2 +0.2
- ripple (faithful minus net): largest 0.2%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.0 -0.0 -0.0 -0.1
  - largest error per slice: +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.007 | 0.71% |
| x0 inside exp | 1 | 0.9979 | 0.43% |
| offset | 0 | -0.005888 | 0.08% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.145, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -0.466 | 0 | 0.38 | 0.62 | 0.424 | + | hinge |
| 1 | 180 | 2.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 4.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 0 | 1.1 | 0 | 0.78 | 0.22 | 0.588 | + | hinge |
| 4 | 180 | -1.99 | 0 | 0.00 | 1.00 | 0.586 | - | line |
| 5 | 0 | 1.67 | 0 | 0.92 | 0.08 | 0.288 | + | hinge |
| 6 | 0 | -2.05 | 0 | 0.00 | 1.00 | 0.13 | + | line |
| 7 | 0 | 1.84 | 0 | 0.96 | 0.04 | 0.159 | + | hinge |
| 8 | 0 | -1.95 | 0 | 0.01 | 0.99 | 0.778 | + | line |
| 9 | 180 | 3.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 0 | -1.07 | 0 | 0.23 | 0.77 | 0.411 | + | hinge |
| 11 | 0 | 0.878 | 0 | 0.72 | 0.28 | 0.617 | + | hinge |
| 12 | 0 | 1.5 | 0 | 0.88 | 0.12 | 0.408 | + | hinge |
| 13 | 180 | -1.66 | 0 | 0.08 | 0.92 | 0.0422 | - | constant |
| 14 | 180 | 7.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 0 | -1.48 | 0 | 0.13 | 0.87 | 0.335 | + | hinge |
| 16 | 0 | 4.11 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | 27.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | 0 | 1.31 | 0 | 0.83 | 0.17 | 0.517 | + | hinge |
| 19 | 0 | 2.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 20 | 180 | -1.08 | 0 | 0.23 | 0.77 | 1.47e-05 | - | constant |
| 21 | 180 | 0.187 | 0 | 0.55 | 0.45 | 0.0465 | - | constant |
| 22 | 180 | -1.81 | 0 | 0.05 | 0.95 | 0.0912 | - | line |
| 23 | 180 | -1.67 | 0 | 0.08 | 0.92 | 0.00594 | + | constant |
| 24 | 180 | -1.9 | 0 | 0.02 | 0.98 | 0.468 | + | line |
| 25 | 0 | 13.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 26 | 0 | 0.131 | 0 | 0.53 | 0.47 | 0.626 | + | hinge |
| 27 | 0 | 0.654 | 0 | 0.66 | 0.34 | 0.579 | + | hinge |
| 28 | 180 | 0.74 | 0 | 0.69 | 0.31 | 0.167 | + | hinge |
| 29 | 0 | -0.174 | 0 | 0.46 | 0.54 | 0.623 | + | hinge |
| 30 | 0 | -2.82 | 0 | 0.00 | 1.00 | 0.00204 | + | constant |
| 31 | 0 | 0.412 | 0 | 0.60 | 0.40 | 0.639 | + | hinge |

## Groups

- negligible: units 1, 2, 9, 14, 16, 17, 19, 20, 23, 25, 30 (magnitude 0.00799 together; a quarter of the tolerance is 0.0362)
- lines: units 4, 6, 8, 13, 21, 22, 24 (each a constant or a line on the box; together one linear term)
- same line: units 5, 7, 12, 18 (gate angle 0 degrees, offset 1.67)
- the rest: units 0, 3, 10, 11, 15, 26, 27, 28, 29, 31
