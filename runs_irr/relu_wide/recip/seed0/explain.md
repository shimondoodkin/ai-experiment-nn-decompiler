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
    return 1.002 / (x0 + 0.001732) - 0.003709
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.3 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2
  - largest error per slice: +0.4 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2
- ripple (faithful minus net): largest 0.3%, at the box edge (x0 = 0.535)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.0 -0.1 -0.1 -0.1 -0.2 -0.1 -0.2 -0.2
  - largest error per slice: +0.3 +0.2 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside reciprocal | 0 | 0.001732 | 0.40% |
| scale | 1 | 1.002 | 0.23% |
| offset | 0 | -0.003709 | 0.21% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0348, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 86.6 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 3.49 | 0 | 0.86 | 0.14 | 0.00858 | + | constant |
| 2 | 180 | 0.849 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | 1.27 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | -12.3 | 0 | 0.00 | 1.00 | 0.0632 | + | line |
| 6 | 180 | -1.54 | 0 | 0.70 | 0.30 | 0.133 | + | hinge |
| 7 | 0 | 3.53 | 0 | 0.87 | 0.13 | 0.000417 | + | constant |
| 8 | 180 | -2.12 | 0 | 0.54 | 0.46 | 0.12 | + | hinge |
| 9 | 0 | 1.81 | 0 | 0.37 | 0.63 | 0.0128 | + | constant |
| 10 | 180 | -0.671 | 0 | 0.95 | 0.05 | 0.116 | + | hinge |
| 11 | 180 | 4.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 180 | 0.756 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | 180 | 0.778 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 14 | 180 | -1.31 | 0 | 0.77 | 0.23 | 0.163 | + | hinge |
| 15 | 180 | -1.8 | 0 | 0.63 | 0.37 | 0.123 | + | hinge |
| 16 | 0 | 0.508 | 0 | 0.00 | 1.00 | 0.129 | - | line |
| 17 | 0 | 0.669 | 0 | 0.05 | 0.95 | 0.017 | + | constant |
| 18 | 180 | -0.928 | 0 | 0.88 | 0.12 | 0.165 | + | hinge |
| 19 | 180 | -0.787 | 0 | 0.92 | 0.08 | 0.15 | + | hinge |
| 20 | 180 | -1.42 | 0 | 0.74 | 0.26 | 0.00256 | + | constant |
| 21 | 0 | 4.48 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 22 | 180 | -1.1 | 0 | 0.83 | 0.17 | 0.17 | + | hinge |
| 23 | 0 | 0.509 | 0 | 0.00 | 1.00 | 0.189 | - | line |
| 24 | 180 | -2.51 | 0 | 0.43 | 0.57 | 0.11 | + | hinge |
| 25 | 0 | 7.33 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 26 | 0 | 3.12 | 0 | 0.75 | 0.25 | 0.0015 | + | constant |
| 27 | 180 | 0.557 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 180 | -0.577 | 0 | 0.98 | 0.02 | 0.069 | + | hinge |
| 29 | 180 | -3.01 | 0 | 0.28 | 0.72 | 0.0978 | + | hinge |
| 30 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 31 | 0 | 0.516 | 0 | 0.00 | 1.00 | 0.107 | - | line |

## Groups

- negligible: units 0, 1, 2, 3, 4, 7, 11, 12, 13, 20, 21, 25, 26, 27, 30 (magnitude 0.0131 together; a quarter of the tolerance is 0.00871)
- lines: units 5, 9, 16, 17, 23, 31 (each a constant or a line on the box; together one linear term)
- same line: units 10, 18, 19, 22, 28 (gate angle 180 degrees, offset -0.671)
- the rest: units 6, 8, 14, 15, 24, 29
