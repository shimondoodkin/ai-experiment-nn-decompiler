# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return log(x0)
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.009 * log(x0 + 0.002974) + 0.0005068
```

Clean:

```python
def net(x0):
    return log(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: -0.0 -0.1 -0.1 -0.2 -0.2 -0.3 -0.3 -0.3 -0.4 -0.4
  - largest error per slice: +0.2 +0.1 +0.2 +0.2 +0.3 +0.3 +0.3 +0.4 +0.4 +0.4
- ripple (faithful minus net): largest 0.3%, at the box edge (x0 = 0.127)
  - mean error in 10 equal slices of x0: -0.0 -0.0 +0.1 +0.2 +0.2 +0.3 +0.3 +0.3 +0.3 +0.3
  - largest error per slice: +0.3 +0.3 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.002974 | 0.75% |
| scale | 1 | 1.009 | 0.53% |
| offset | 0 | 0.0005068 | 0.01% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.078, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -1.53 | 0 | 0.00 | 1.00 | 0.15 | + | line |
| 1 | 180 | -2.38 | 0 | 0.53 | 0.47 | 0.215 | - | hinge |
| 2 | 0 | 3.9 | 0 | 0.78 | 0.22 | 3.58e-05 | - | constant |
| 3 | 180 | 8.97 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | -0.0856 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 5 | 0 | 3.69 | 0 | 0.73 | 0.27 | 0.0449 | - | hinge |
| 6 | 180 | 1.93 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 180 | 0.678 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | 0 | 4.25 | 0 | 0.85 | 0.15 | 0.0285 | - | constant |
| 9 | 180 | 1.05 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 180 | -2.83 | 0 | 0.44 | 0.56 | 0.062 | - | hinge |
| 11 | 180 | -2.78 | 0 | 0.45 | 0.55 | 0.04 | - | hinge |
| 12 | 180 | -0.629 | 0 | 0.89 | 0.11 | 0.259 | - | hinge |
| 13 | 0 | 2.98 | 0 | 0.59 | 0.41 | 6.07e-05 | - | constant |
| 14 | 180 | -3.18 | 0 | 0.37 | 0.63 | 0.15 | - | hinge |
| 15 | 0 | 5.62 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 16 | 0 | -0.395 | 0 | 0.00 | 1.00 | 0.351 | + | line |
| 17 | 180 | -0.229 | 0 | 0.97 | 0.03 | 0.212 | - | hinge |
| 18 | 180 | -1.12 | 0 | 0.79 | 0.21 | 0.245 | - | hinge |
| 19 | 0 | -0.769 | 0 | 0.00 | 1.00 | 0.557 | + | line |
| 20 | 180 | -0.46 | 0 | 0.93 | 0.07 | 0.258 | - | hinge |
| 21 | 180 | -1.88 | 0 | 0.64 | 0.36 | 0.229 | - | hinge |
| 22 | 180 | 3.08 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 23 | 0 | 0.102 | 0 | 0.00 | 1.00 | 0.366 | + | line |
| 24 | 180 | 2.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 180 | -0.704 | 0 | 0.87 | 0.13 | 0.000219 | - | constant |
| 26 | 180 | -0.918 | 0 | 0.83 | 0.17 | 0.00053 | - | constant |
| 27 | 180 | -1.46 | 0 | 0.72 | 0.28 | 0.239 | - | hinge |
| 28 | 180 | -0.328 | 0 | 0.95 | 0.05 | 0.243 | - | hinge |
| 29 | 180 | 1.76 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 30 | 180 | -0.155 | 0 | 0.99 | 0.01 | 0.139 | - | hinge |
| 31 | 180 | -0.85 | 0 | 0.84 | 0.16 | 0.257 | - | hinge |

## Groups

- negligible: units 2, 3, 4, 6, 7, 9, 13, 15, 22, 24, 25, 26, 29 (magnitude 0.000846 together; a quarter of the tolerance is 0.0195)
- lines: units 0, 8, 16, 19, 23 (each a constant or a line on the box; together one linear term)
- same line: units 10, 11 (gate angle 180 degrees, offset -2.83)
- same line: units 12, 17, 20, 28, 30, 31 (gate angle 180 degrees, offset -0.629)
- the rest: units 1, 5, 14, 18, 21, 27
