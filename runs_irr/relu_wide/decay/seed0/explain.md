# Explain

Inputs: x0 in [-1, 3]; output: regression.

## Formula

A family program passes the check (12 nodes):

```python
def net(x0):
    return 18 / (x0 + 3.5) - 2.184
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 18.3 / (x0 + 3.564) - 2.163
```

Clean:

```python
def net(x0):
    return 18 / (x0 + 3.5) - 2.184
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, at the box edge (x0 = -1)
  - mean error in 10 equal slices of x0: +0.6 +0.0 -0.3 -0.5 -0.7 -0.8 -0.8 -0.9 -0.9 -0.9
  - largest error per slice: +1.0 +0.3 +0.4 +0.6 +0.7 +0.8 +0.9 +0.9 +0.9 +0.9
- ripple (faithful minus net): largest 1.0%, at the box edge (x0 = 3)
  - mean error in 10 equal slices of x0: -0.2 -0.9 -0.6 -0.0 +0.5 +0.8 +0.9 +0.7 +0.2 -0.5
  - largest error per slice: +0.9 +1.0 +0.8 +0.4 +0.7 +1.0 +1.0 +0.8 +0.4 +1.0

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside reciprocal | 3.5 | 3.564 | 4.21% |
| scale | 18 | 18.3 | 2.81% |
| offset | -2.184 | -2.163 | 0.49% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0854, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 86.6 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 1.04 | 0 | 0.51 | 0.49 | 0.308 | - | hinge |
| 2 | 180 | 0.722 | 0 | 0.93 | 0.07 | 0.0839 | + | hinge |
| 3 | 180 | 1.27 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | -39.2 | 0 | 0.00 | 1.00 | 0.0483 | + | line |
| 6 | 180 | -1.42 | 0 | 0.39 | 0.61 | 0.00414 | + | constant |
| 7 | 0 | -0.539 | 0 | 0.12 | 0.88 | 2.72e-09 | + | constant |
| 8 | 180 | -1.6 | 0 | 0.35 | 0.65 | 0.269 | + | hinge |
| 9 | 0 | 4.39 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 10 | 180 | -0.379 | 0 | 0.66 | 0.34 | 0.381 | + | hinge |
| 11 | 180 | 4.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 180 | 0.437 | 0 | 0.86 | 0.14 | 0.152 | + | hinge |
| 13 | 180 | 1.25 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 14 | 180 | -2.33 | 0 | 0.17 | 0.83 | 0.317 | + | hinge |
| 15 | 180 | -2.53 | 0 | 0.12 | 0.88 | 0.0534 | + | line |
| 16 | 0 | 0.38 | 0 | 0.34 | 0.66 | 0.169 | - | hinge |
| 17 | 0 | -0.199 | 0 | 0.20 | 0.80 | 0.49 | + | hinge |
| 18 | 180 | -1.04 | 0 | 0.49 | 0.51 | 0.592 | + | hinge |
| 19 | 180 | -1.33 | 0 | 0.42 | 0.58 | 0.244 | + | hinge |
| 20 | 0 | -56.9 | 0 | 0.00 | 1.00 | 0.00764 | + | constant |
| 21 | 0 | 3.01 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 22 | 180 | -2.98 | 0 | 0.00 | 1.00 | 0.335 | + | line |
| 23 | 0 | 0.73 | 0 | 0.43 | 0.57 | 0.312 | - | hinge |
| 24 | 180 | -1.94 | 0 | 0.26 | 0.74 | 0.299 | + | hinge |
| 25 | 0 | 7.33 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 26 | 0 | 0.74 | 0 | 0.43 | 0.57 | 0.0829 | - | hinge |
| 27 | 180 | -0.0431 | 0 | 0.74 | 0.26 | 0.24 | + | hinge |
| 28 | 180 | -0.727 | 0 | 0.57 | 0.43 | 0.601 | + | hinge |
| 29 | 180 | -2.83 | 0 | 0.04 | 0.96 | 0.368 | + | line |
| 30 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 31 | 0 | 1.62 | 0 | 0.66 | 0.34 | 0.000173 | - | constant |

## Groups

- negligible: units 0, 3, 4, 6, 7, 9, 11, 13, 20, 21, 25, 30, 31 (magnitude 0.012 together; a quarter of the tolerance is 0.0213)
- lines: units 5, 15, 22, 29 (each a constant or a line on the box; together one linear term)
- same line: units 1, 18 (gate angle 0 degrees, offset 1.04)
- same line: units 10, 16 (gate angle 180 degrees, offset -0.379)
- same line: units 23, 26, 28 (gate angle 0 degrees, offset 0.73)
- the rest: units 2, 8, 12, 14, 17, 19, 24, 27
