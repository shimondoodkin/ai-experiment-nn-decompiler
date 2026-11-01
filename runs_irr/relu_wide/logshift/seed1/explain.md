# Explain

Inputs: x0 in [-1.5, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return log(x0 + 2)
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.005 * log(x0 + 2.003) - 0.0001125
```

Clean:

```python
def net(x0):
    return log(x0 + 2)
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: -0.1 -0.1 -0.2 -0.2 -0.3 -0.3 -0.3 -0.3 -0.4 -0.4
  - largest error per slice: +0.1 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3 +0.3 +0.4 +0.4
- ripple (faithful minus net): largest 0.4%, inside the box (x0 = 3.49)
  - mean error in 10 equal slices of x0: +0.0 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.3 +0.3 +0.3
  - largest error per slice: +0.3 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.005 | 0.36% |
| constant inside log | 2 | 2.003 | 0.24% |
| offset | 0 | -0.0001125 | 0.00% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0496, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.818 | 0 | 0.87 | 0.13 | 3.73e-08 | - | constant |
| 1 | 180 | 2.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 4.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 0 | 1.69 | 0 | 0.58 | 0.42 | 0.00111 | - | constant |
| 4 | 180 | -0.956 | 0 | 0.55 | 0.45 | 0.332 | - | hinge |
| 5 | 0 | 0.328 | 0 | 0.33 | 0.67 | 0.383 | - | hinge |
| 6 | 180 | -1.3 | 0 | 0.49 | 0.51 | 0.00832 | - | constant |
| 7 | 0 | 1.63 | 0 | 0.57 | 0.43 | 0.11 | - | hinge |
| 8 | 0 | -5.57 | 0 | 0.00 | 1.00 | 0.223 | + | line |
| 9 | 180 | 3.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 0 | -2.64 | 0 | 0.00 | 1.00 | 0.301 | + | line |
| 11 | 180 | 0.211 | 0 | 0.77 | 0.23 | 0.204 | - | hinge |
| 12 | 0 | 1 | 0 | 0.46 | 0.54 | 0.0159 | + | constant |
| 13 | 180 | -2.25 | 0 | 0.32 | 0.68 | 0.119 | - | hinge |
| 14 | 180 | 7.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 180 | -1.3 | 0 | 0.49 | 0.51 | 0.00216 | - | constant |
| 16 | 0 | 15.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | 27.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | 0 | 0.968 | 0 | 0.45 | 0.55 | 0.164 | + | hinge |
| 19 | 0 | 7.38 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 20 | 180 | -1.98 | 0 | 0.37 | 0.63 | 0.0153 | - | constant |
| 21 | 180 | 1.02 | 0 | 0.91 | 0.09 | 0.155 | - | hinge |
| 22 | 180 | -2.98 | 0 | 0.18 | 0.82 | 0.158 | - | hinge |
| 23 | 180 | -4 | 0 | 0.00 | 1.00 | 0.0929 | + | line |
| 24 | 180 | 0.662 | 0 | 0.85 | 0.15 | 0.188 | - | hinge |
| 25 | 0 | 13.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 26 | 0 | -1.55 | 0 | 0.00 | 1.00 | 0.553 | + | line |
| 27 | 0 | -0.834 | 0 | 0.12 | 0.88 | 0.00401 | - | constant |
| 28 | 180 | 1.29 | 0 | 0.96 | 0.04 | 0.101 | - | hinge |
| 29 | 0 | -1.56 | 0 | 0.00 | 1.00 | 0.53 | + | line |
| 30 | 180 | -1.19 | 0 | 0.51 | 0.49 | 6.88e-06 | - | constant |
| 31 | 0 | 4.37 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 0, 1, 2, 3, 6, 9, 14, 15, 16, 17, 19, 25, 27, 30, 31 (magnitude 0.0156 together; a quarter of the tolerance is 0.0124)
- lines: units 8, 10, 12, 20, 23, 26, 29 (each a constant or a line on the box; together one linear term)
- same line: units 4, 18 (gate angle 180 degrees, offset -0.956)
- same line: units 21, 28 (gate angle 180 degrees, offset 1.02)
- the rest: units 5, 7, 11, 13, 22, 24
