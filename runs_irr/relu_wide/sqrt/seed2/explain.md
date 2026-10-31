# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return sqrt(x0) + 0.02
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.005 * sqrt(x0 + 0.001891) - 0.00378
```

Clean:

```python
def net(x0):
    return sqrt(x0) + 0.02
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: +1.0 +1.0 +0.9 +0.9 +0.8 +0.8 +0.8 +0.7 +0.7 +0.7
  - largest error per slice: +1.0 +1.0 +0.9 +0.9 +0.9 +0.8 +0.8 +0.8 +0.7 +0.7
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 0.026)
  - mean error in 10 equal slices of x0: -0.0 -0.0 +0.0 +0.1 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2
  - largest error per slice: +0.5 +0.2 +0.1 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.001891 | 2.22% |
| offset | 0.02 | -0.00378 | 1.22% |
| scale | 1 | 1.005 | 0.51% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0391, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -1.52 | 0 | 0.00 | 1.00 | 0.21 | + | line |
| 1 | 180 | -4 | 0 | 0.00 | 1.00 | 0.0323 | + | line |
| 2 | 0 | 0.0786 | 0 | 0.02 | 0.98 | 0.0135 | + | constant |
| 3 | 180 | 8.97 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | -0.897 | 0 | 0.78 | 0.22 | 0.0024 | - | constant |
| 5 | 0 | 4.97 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | 180 | 1.93 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 180 | 0.678 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | 0 | 4.17 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 9 | 180 | 1.05 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 180 | -1.83 | 0 | 0.54 | 0.46 | 0.0124 | - | constant |
| 11 | 180 | -3.33 | 0 | 0.17 | 0.83 | 0.00107 | + | constant |
| 12 | 180 | -0.725 | 0 | 0.82 | 0.18 | 0.0922 | - | hinge |
| 13 | 0 | 4.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 14 | 180 | -2.58 | 0 | 0.35 | 0.65 | 0.0958 | - | hinge |
| 15 | 0 | 3.25 | 0 | 0.81 | 0.19 | 0.0221 | - | hinge |
| 16 | 0 | -0.52 | 0 | 0.00 | 1.00 | 0.387 | + | line |
| 17 | 180 | 0.114 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 18 | 180 | -0.238 | 0 | 0.94 | 0.06 | 0.0873 | - | hinge |
| 19 | 0 | -0.819 | 0 | 0.00 | 1.00 | 0.628 | + | line |
| 20 | 180 | -0.099 | 0 | 0.98 | 0.02 | 0.0824 | - | hinge |
| 21 | 180 | -2.04 | 0 | 0.49 | 0.51 | 0.0809 | - | hinge |
| 22 | 180 | 3.08 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 23 | 0 | 0.000272 | 0 | 0.00 | 1.00 | 0.406 | + | line |
| 24 | 180 | 2.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 180 | -1.08 | 0 | 0.73 | 0.27 | 0.0932 | - | hinge |
| 26 | 180 | 0.444 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 27 | 180 | -0.446 | 0 | 0.89 | 0.11 | 0.0915 | - | hinge |
| 28 | 180 | -1.52 | 0 | 0.62 | 0.38 | 0.0948 | - | hinge |
| 29 | 180 | 1.76 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 30 | 180 | -0.0257 | 0 | 0.99 | 0.01 | 0.0717 | - | hinge |
| 31 | 180 | -4 | 0 | 0.00 | 1.00 | 0.438 | + | line |

## Groups

- negligible: units 3, 4, 5, 6, 7, 8, 9, 11, 13, 17, 22, 24, 26, 29 (magnitude 0.00347 together; a quarter of the tolerance is 0.00978)
- lines: units 0, 1, 2, 10, 16, 19, 23, 31 (each a constant or a line on the box; together one linear term)
- same line: units 18, 20, 30 (gate angle 180 degrees, offset -0.238)
- the rest: units 12, 14, 15, 21, 25, 27, 28
