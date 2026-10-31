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
    return 1.01 / (x0 + 0.004731) - 0.005107
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.2%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.1 -0.0 +0.0 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2
  - largest error per slice: +0.2 +0.0 +0.0 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: -0.0 +0.0 -0.0 -0.0 -0.0 -0.1 -0.1 -0.1 -0.2 -0.2
  - largest error per slice: +0.4 +0.3 +0.2 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.01 | 1.15% |
| constant inside reciprocal | 0 | 0.004731 | 1.08% |
| offset | 0 | -0.005107 | 0.29% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0347, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -138 | 0 | 0.00 | 1.00 | 0.000704 | + | constant |
| 1 | 180 | -3.94 | 0 | 0.02 | 0.98 | 0.281 | + | line |
| 2 | 0 | 4.69 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | 8.97 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | -0.428 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | 2.73 | 0 | 0.64 | 0.36 | 0.051 | + | hinge |
| 6 | 180 | 1.93 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 180 | 0.678 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | 0 | 2.56 | 0 | 0.59 | 0.41 | 0.000252 | + | constant |
| 9 | 180 | 1.05 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 180 | -2.06 | 0 | 0.55 | 0.45 | 0.138 | + | hinge |
| 11 | 180 | -3.22 | 0 | 0.22 | 0.78 | 0.0954 | + | hinge |
| 12 | 180 | -4.47 | 0 | 0.00 | 1.00 | 0.11 | + | line |
| 13 | 0 | 0.117 | 0 | 0.00 | 1.00 | 0.13 | - | line |
| 14 | 180 | -1.67 | 0 | 0.67 | 0.33 | 0.158 | + | hinge |
| 15 | 0 | 2.58 | 0 | 0.59 | 0.41 | 0.000794 | + | constant |
| 16 | 0 | -0.823 | 0 | 0.00 | 1.00 | 0.0281 | + | line |
| 17 | 180 | -0.129 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | 180 | -0.579 | 0 | 0.98 | 0.02 | 0.0728 | + | hinge |
| 19 | 0 | -1.06 | 0 | 0.00 | 1.00 | 0.079 | + | line |
| 20 | 180 | -0.686 | 0 | 0.95 | 0.05 | 0.131 | + | hinge |
| 21 | 180 | -1.38 | 0 | 0.75 | 0.25 | 0.172 | + | hinge |
| 22 | 180 | 3.08 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 23 | 0 | 2.98 | 0 | 0.71 | 0.29 | 0.000691 | + | constant |
| 24 | 180 | 2.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 180 | -1.85 | 0 | 0.62 | 0.38 | 0.0179 | + | hinge |
| 26 | 180 | -2.4 | 0 | 0.46 | 0.54 | 0.0735 | + | hinge |
| 27 | 180 | -0.809 | 0 | 0.91 | 0.09 | 0.162 | + | hinge |
| 28 | 180 | -0.965 | 0 | 0.86 | 0.14 | 0.174 | + | hinge |
| 29 | 180 | 1.76 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 30 | 180 | -0.322 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 31 | 180 | -1.15 | 0 | 0.81 | 0.19 | 0.179 | + | hinge |

## Groups

- negligible: units 0, 2, 3, 4, 6, 7, 8, 9, 15, 17, 22, 23, 24, 29, 30 (magnitude 0.00244 together; a quarter of the tolerance is 0.00869)
- lines: units 1, 12, 13, 16, 19 (each a constant or a line on the box; together one linear term)
- same line: units 18, 20, 27, 28 (gate angle 180 degrees, offset -0.579)
- the rest: units 5, 10, 11, 14, 21, 25, 26, 31
