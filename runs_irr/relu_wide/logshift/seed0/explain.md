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
    return log(x0 + 2.002) - 0.002579
```

Clean:

```python
def net(x0):
    return log(x0 + 2)
```

% of the output range:
- distortion (clean minus faithful): largest 0.1%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: -0.0 +0.0 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1
  - largest error per slice: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1
- ripple (faithful minus net): largest 0.3%, at the box edge (x0 = -1.4)
  - mean error in 10 equal slices of x0: +0.1 +0.1 +0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0
  - largest error per slice: +0.3 +0.3 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 2 | 2.002 | 0.16% |
| offset | 0 | -0.002579 | 0.10% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0495, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 86.6 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 2.23 | 0 | 0.68 | 0.32 | 0.0624 | - | hinge |
| 2 | 180 | 0.67 | 0 | 0.85 | 0.15 | 0.185 | - | hinge |
| 3 | 180 | 1.61 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | -2.94 | 0 | 0.00 | 1.00 | 0.585 | + | line |
| 6 | 180 | -1.75 | 0 | 0.41 | 0.59 | 0.0731 | - | hinge |
| 7 | 0 | 9.85 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 8 | 180 | 1.96 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 0 | 4.48 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 10 | 180 | 1.63 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 11 | 180 | 4.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 180 | 1.02 | 0 | 0.91 | 0.09 | 0.153 | - | hinge |
| 13 | 180 | 0.227 | 0 | 0.77 | 0.23 | 0.383 | - | hinge |
| 14 | 180 | -4.13 | 0 | 0.00 | 1.00 | 0.134 | + | line |
| 15 | 180 | -0.511 | 0 | 0.64 | 0.36 | 4.22e-12 | - | constant |
| 16 | 0 | 2.87 | 0 | 0.80 | 0.20 | 0.0287 | - | hinge |
| 17 | 0 | -0.225 | 0 | 0.23 | 0.77 | 0.616 | + | hinge |
| 18 | 180 | -3.32 | 0 | 0.12 | 0.88 | 0.07 | - | line |
| 19 | 180 | -1.32 | 0 | 0.49 | 0.51 | 0.139 | - | hinge |
| 20 | 0 | -1.81 | 0 | 0.00 | 1.00 | 0.162 | + | line |
| 21 | 0 | -0.214 | 0 | 0.23 | 0.77 | 0.000957 | + | constant |
| 22 | 180 | -3.97 | 0 | 0.01 | 0.99 | 0.0429 | + | line |
| 23 | 0 | 0.833 | 0 | 0.42 | 0.58 | 0.193 | - | hinge |
| 24 | 180 | 1.65 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 0 | 7.33 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 26 | 0 | 0.316 | 0 | 0.33 | 0.67 | 0.224 | + | hinge |
| 27 | 180 | 1.29 | 0 | 0.96 | 0.04 | 0.0989 | - | hinge |
| 28 | 180 | -0.305 | 0 | 0.67 | 0.33 | 0.302 | - | hinge |
| 29 | 180 | -3.97 | 0 | 0.01 | 0.99 | 0.0518 | + | line |
| 30 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 31 | 0 | -0.428 | 0 | 0.20 | 0.80 | 0.000414 | - | constant |

## Groups

- negligible: units 0, 3, 4, 7, 8, 9, 10, 11, 15, 21, 24, 25, 30, 31 (magnitude 0.00137 together; a quarter of the tolerance is 0.0124)
- lines: units 5, 14, 18, 20, 22, 29 (each a constant or a line on the box; together one linear term)
- same line: units 12, 27 (gate angle 180 degrees, offset 1.02)
- same line: units 13, 17 (gate angle 180 degrees, offset 0.227)
- same line: units 26, 28 (gate angle 0 degrees, offset 0.316)
- the rest: units 1, 2, 6, 16, 19, 23
