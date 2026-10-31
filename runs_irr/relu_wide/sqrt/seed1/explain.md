# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return sqrt(x0) + 0.01834
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.01 * sqrt(x0 + 0.002256) - 0.01388
```

Clean:

```python
def net(x0):
    return sqrt(x0) + 0.01834
```

% of the output range:
- distortion (clean minus faithful): largest 1.3%, at the box edge (x0 = 0.112)
  - mean error in 10 equal slices of x0: +1.3 +1.2 +1.1 +1.0 +0.9 +0.9 +0.8 +0.7 +0.7 +0.6
  - largest error per slice: +1.3 +1.2 +1.1 +1.0 +1.0 +0.9 +0.8 +0.8 +0.7 +0.6
- ripple (faithful minus net): largest 0.8%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: -0.1 -0.2 -0.1 -0.0 +0.1 +0.1 +0.2 +0.2 +0.3 +0.3
  - largest error per slice: +0.8 +0.5 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2 +0.4 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.002256 | 2.43% |
| offset | 0.01834 | -0.01388 | 1.65% |
| scale | 1 | 1.01 | 1.02% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0391, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.639 | 0 | 0.84 | 0.16 | 0.00378 | - | constant |
| 1 | 180 | 2.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 4.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 0 | 0.197 | 0 | 0.05 | 0.95 | 0.0205 | + | line |
| 4 | 180 | -0.05 | 0 | 0.99 | 0.01 | 0.108 | - | hinge |
| 5 | 0 | 4.66 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | 180 | -2.06 | 0 | 0.48 | 0.52 | 0.102 | - | hinge |
| 7 | 0 | 0.767 | 0 | 0.19 | 0.81 | 0.335 | - | hinge |
| 8 | 0 | -6.28 | 0 | 0.00 | 1.00 | 0.134 | + | line |
| 9 | 180 | 3.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 0 | -2.81 | 0 | 0.00 | 1.00 | 0.19 | + | line |
| 11 | 0 | 4.16 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 12 | 0 | 0.197 | 0 | 0.05 | 0.95 | 0.0193 | + | constant |
| 13 | 180 | -0.442 | 0 | 0.89 | 0.11 | 0.119 | - | hinge |
| 14 | 180 | 7.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 180 | -2.55 | 0 | 0.36 | 0.64 | 0.0596 | - | hinge |
| 16 | 0 | 16.6 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | 27.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | 0 | 0.194 | 0 | 0.05 | 0.95 | 0.126 | + | line |
| 19 | 0 | 6.56 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 20 | 180 | -0.191 | 0 | 0.95 | 0.05 | 0.121 | - | hinge |
| 21 | 180 | 0.402 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 22 | 180 | -1.08 | 0 | 0.73 | 0.27 | 0.0949 | - | hinge |
| 23 | 180 | -3.08 | 0 | 0.23 | 0.77 | 0.108 | - | hinge |
| 24 | 180 | 0.146 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 0 | 13.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 26 | 0 | -1.03 | 0 | 0.00 | 1.00 | 0.48 | + | line |
| 27 | 0 | 6.08 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 28 | 180 | 0.661 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 29 | 0 | -1.06 | 0 | 0.00 | 1.00 | 0.449 | + | line |
| 30 | 180 | -1.52 | 0 | 0.62 | 0.38 | 0.0976 | - | hinge |
| 31 | 0 | -0.766 | 0 | 0.00 | 1.00 | 0.0552 | + | line |

## Groups

- negligible: units 0, 1, 2, 5, 9, 11, 14, 16, 17, 19, 21, 24, 25, 27, 28 (magnitude 0.00378 together; a quarter of the tolerance is 0.00977)
- lines: units 3, 8, 10, 12, 18, 26, 29, 31 (each a constant or a line on the box; together one linear term)
- same line: units 4, 20 (gate angle 180 degrees, offset -0.05)
- the rest: units 6, 7, 13, 15, 22, 23, 30
