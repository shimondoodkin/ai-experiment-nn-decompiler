# Explain

Inputs: x0 in [-1, 3]; output: regression.

## Formula

A family program passes the check (12 nodes):

```python
def net(x0):
    return 18 / (x0 + 3.5) - 2.18
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 18.38 / (x0 + 3.57) - 2.174
```

Clean:

```python
def net(x0):
    return 18 / (x0 + 3.5) - 2.18
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, at the box edge (x0 = -1)
  - mean error in 10 equal slices of x0: +0.6 +0.1 -0.3 -0.5 -0.6 -0.7 -0.7 -0.8 -0.8 -0.8
  - largest error per slice: +1.0 +0.3 +0.4 +0.5 +0.7 +0.7 +0.8 +0.8 +0.8 +0.8
- ripple (faithful minus net): largest 1.0%, inside the box (x0 = 1.44)
  - mean error in 10 equal slices of x0: -0.2 -0.9 -0.6 +0.0 +0.5 +0.8 +0.8 +0.5 +0.1 -0.6
  - largest error per slice: +0.9 +0.9 +0.9 +0.4 +0.8 +0.9 +1.0 +0.8 +0.4 +1.0

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside reciprocal | 3.5 | 3.57 | 4.59% |
| scale | 18 | 18.38 | 3.56% |
| offset | -2.18 | -2.174 | 0.14% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0855, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.278 | 0 | 0.82 | 0.18 | 1.36e-09 | + | constant |
| 1 | 180 | 2.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 4.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 0 | 1.06 | 0 | 0.52 | 0.48 | 0.027 | - | constant |
| 4 | 180 | -0.265 | 0 | 0.69 | 0.31 | 0.302 | + | hinge |
| 5 | 0 | 0.652 | 0 | 0.41 | 0.59 | 0.193 | - | hinge |
| 6 | 180 | -1.81 | 0 | 0.30 | 0.70 | 0.324 | + | hinge |
| 7 | 0 | 1.45 | 0 | 0.61 | 0.39 | 0.169 | - | hinge |
| 8 | 180 | -10.1 | 0 | 0.00 | 1.00 | 0.177 | + | line |
| 9 | 180 | 3.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 180 | -42.4 | 0 | 0.00 | 1.00 | 0.0254 | + | constant |
| 11 | 180 | -1.98 | 0 | 0.25 | 0.75 | 0.0143 | + | constant |
| 12 | 0 | 1.45 | 0 | 0.61 | 0.39 | 0.0824 | - | hinge |
| 13 | 180 | -0.648 | 0 | 0.59 | 0.41 | 0.486 | + | hinge |
| 14 | 180 | 7.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 180 | -2.22 | 0 | 0.19 | 0.81 | 0.319 | + | hinge |
| 16 | 0 | 8.08 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | 27.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | 0 | 1.05 | 0 | 0.51 | 0.49 | 0.261 | - | hinge |
| 19 | 0 | 5.92 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 20 | 180 | -1.04 | 0 | 0.49 | 0.51 | 0.665 | + | hinge |
| 21 | 180 | 0.714 | 0 | 0.93 | 0.07 | 0.0904 | + | hinge |
| 22 | 180 | -2.59 | 0 | 0.10 | 0.90 | 0.25 | + | hinge |
| 23 | 180 | -1.44 | 0 | 0.39 | 0.61 | 0.742 | + | hinge |
| 24 | 180 | 0.0864 | 0 | 0.77 | 0.23 | 0.241 | + | hinge |
| 25 | 0 | 13.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 26 | 0 | -3.63 | 0 | 0.00 | 1.00 | 0.136 | + | line |
| 27 | 0 | 1.08 | 0 | 0.52 | 0.48 | 8.24e-07 | + | constant |
| 28 | 180 | 0.411 | 0 | 0.85 | 0.15 | 0.171 | + | hinge |
| 29 | 0 | -3.71 | 0 | 0.00 | 1.00 | 0.13 | + | line |
| 30 | 180 | -1.22 | 0 | 0.44 | 0.56 | 0.00024 | + | constant |
| 31 | 180 | 0.573 | 0 | 0.89 | 0.11 | 3.36e-08 | + | constant |

## Groups

- negligible: units 0, 1, 2, 9, 11, 14, 16, 17, 19, 25, 27, 30, 31 (magnitude 0.0145 together; a quarter of the tolerance is 0.0214)
- lines: units 3, 8, 10, 26, 29 (each a constant or a line on the box; together one linear term)
- same line: units 5, 13 (gate angle 0 degrees, offset 0.652)
- same line: units 7, 12, 23 (gate angle 0 degrees, offset 1.45)
- same line: units 18, 20 (gate angle 0 degrees, offset 1.05)
- the rest: units 4, 6, 15, 21, 22, 24, 28
