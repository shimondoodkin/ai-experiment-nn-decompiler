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
    return 0.996 / (x0 - 0.002356) - 0.001462
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.2%, inside the box (x0 = 1.18)
  - mean error in 10 equal slices of x0: +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1
  - largest error per slice: +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.1 +0.1 +0.1
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 0.602)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.2 -0.1 -0.1 -0.2 -0.2 -0.1 -0.2 -0.1
  - largest error per slice: +0.5 +0.3 +0.3 +0.3 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside reciprocal | 0 | -0.002356 | 0.54% |
| scale | 1 | 0.996 | 0.46% |
| offset | 0 | -0.001462 | 0.08% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0349, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 3.14 | 0 | 0.76 | 0.24 | 4.71e-07 | - | constant |
| 1 | 180 | 2.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 4.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 0 | 2.61 | 0 | 0.60 | 0.40 | 0.0544 | + | hinge |
| 4 | 180 | -0.602 | 0 | 0.97 | 0.03 | 0.107 | + | hinge |
| 5 | 0 | 0.504 | 0 | 0.00 | 1.00 | 0.225 | - | line |
| 6 | 180 | -0.723 | 0 | 0.94 | 0.06 | 0.163 | + | hinge |
| 7 | 0 | 1.09 | 0 | 0.17 | 0.83 | 0.0427 | - | line |
| 8 | 180 | -8.68 | 0 | 0.00 | 1.00 | 0.0856 | + | line |
| 9 | 180 | 3.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 180 | -7.9 | 0 | 0.00 | 1.00 | 0.0326 | + | line |
| 11 | 180 | -1.63 | 0 | 0.68 | 0.32 | 0.176 | + | hinge |
| 12 | 0 | 2.78 | 0 | 0.65 | 0.35 | 0.00266 | + | constant |
| 13 | 180 | -2.02 | 0 | 0.57 | 0.43 | 0.16 | + | hinge |
| 14 | 180 | 7.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 180 | -0.879 | 0 | 0.89 | 0.11 | 0.195 | + | hinge |
| 16 | 0 | 4.13 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | 27.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | 0 | 2.52 | 0 | 0.58 | 0.42 | 0.00726 | + | constant |
| 19 | 0 | 4.69 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 20 | 180 | -1.32 | 0 | 0.76 | 0.24 | 0.196 | + | hinge |
| 21 | 180 | 0.402 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 22 | 180 | -3.17 | 0 | 0.24 | 0.76 | 0.12 | + | hinge |
| 23 | 180 | -1.08 | 0 | 0.83 | 0.17 | 0.214 | + | hinge |
| 24 | 180 | 0.146 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 0 | 13.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 26 | 0 | -5.13 | 0 | 0.00 | 1.00 | 0.011 | + | constant |
| 27 | 0 | 0.51 | 0 | 0.00 | 1.00 | 0.0461 | - | line |
| 28 | 180 | 0.661 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 29 | 0 | -30.8 | 0 | 0.00 | 1.00 | 0.00107 | + | constant |
| 30 | 180 | -2.36 | 0 | 0.47 | 0.53 | 0.0516 | + | hinge |
| 31 | 0 | 0.593 | 0 | 0.03 | 0.97 | 0.00613 | - | constant |

## Groups

- negligible: units 0, 1, 2, 9, 12, 14, 16, 17, 18, 19, 21, 24, 25, 28, 29, 31 (magnitude 0.0171 together; a quarter of the tolerance is 0.00872)
- lines: units 5, 7, 8, 10, 26, 27 (each a constant or a line on the box; together one linear term)
- same line: units 4, 6, 15 (gate angle 180 degrees, offset -0.602)
- the rest: units 3, 11, 13, 20, 22, 23, 30
