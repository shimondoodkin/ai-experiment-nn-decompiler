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
    return 1.011 * log(x0 + 0.003194) - 0.003463
```

Clean:

```python
def net(x0):
    return log(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: +0.1 +0.0 -0.1 -0.1 -0.2 -0.2 -0.3 -0.3 -0.3 -0.4
  - largest error per slice: +0.2 +0.1 +0.1 +0.2 +0.2 +0.3 +0.3 +0.3 +0.4 +0.4
- ripple (faithful minus net): largest 0.8%, at the box edge (x0 = 0.1)
  - mean error in 10 equal slices of x0: +0.1 -0.0 +0.2 +0.1 +0.1 +0.3 +0.3 +0.3 +0.6 +0.4
  - largest error per slice: +0.8 +0.5 +0.3 +0.4 +0.3 +0.4 +0.4 +0.5 +0.6 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.003194 | 0.81% |
| scale | 1 | 1.011 | 0.65% |
| offset | 0 | -0.003463 | 0.09% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0777, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.727 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | 180 | 1.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 2.02 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | 0.52 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | 4.63 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | 1.13 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 0 | 10 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | 0.0606 | 0 | 0.00 | 1.00 | 0.503 | + | line |
| 8 | 0 | 6.24 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 0 | 0.103 | 0 | 0.00 | 1.00 | 0.0768 | + | line |
| 10 | 180 | 1.16 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 11 | 0 | 0.67 | 0 | 0.12 | 0.88 | 0.00163 | + | constant |
| 12 | 180 | -0.368 | 0 | 0.95 | 0.05 | 0.419 | - | hinge |
| 13 | 180 | 1.22 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 14 | 180 | 0.767 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 15 | 180 | 2.97 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 16 | 180 | 0.215 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | -2.85 | 0 | 0.44 | 0.56 | 4.32e-05 | - | constant |
| 18 | 0 | 2.98 | 0 | 0.59 | 0.41 | 0.000483 | - | constant |
| 19 | 0 | 0.65 | 0 | 0.11 | 0.89 | 0.00099 | + | constant |
| 20 | 0 | 0.1 | 0 | 0.00 | 1.00 | 0.179 | + | line |
| 21 | 180 | -2.43 | 0 | 0.52 | 0.48 | 0.381 | - | hinge |
| 22 | 0 | -0.668 | 0 | 0.00 | 1.00 | 0.729 | + | line |
| 23 | 180 | -0.204 | 0 | 0.98 | 0.02 | 0.337 | - | hinge |
| 24 | 180 | 0.562 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 180 | -1.62 | 0 | 0.69 | 0.31 | 0.4 | - | hinge |
| 26 | 180 | -0.637 | 0 | 0.89 | 0.11 | 0.434 | - | hinge |
| 27 | 0 | 3.53 | 0 | 0.70 | 0.30 | 0.148 | - | hinge |
| 28 | 180 | -1.04 | 0 | 0.81 | 0.19 | 0.421 | - | hinge |
| 29 | 180 | -0.0925 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 30 | 180 | 1.55 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 31 | 0 | 0.201 | 0 | 0.02 | 0.98 | 0.151 | + | line |

## Groups

- negligible: units 0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 24, 29, 30 (magnitude 0.00314 together; a quarter of the tolerance is 0.0194)
- lines: units 7, 9, 20, 22, 31 (each a constant or a line on the box; together one linear term)
- same line: units 12, 23 (gate angle 180 degrees, offset -0.368)
- the rest: units 21, 25, 26, 27, 28
