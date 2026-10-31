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
    return 1.007 * log(x0 + 0.001922) + 0.0003912
```

Clean:

```python
def net(x0):
    return log(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.3%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: +0.0 -0.0 -0.1 -0.1 -0.2 -0.2 -0.2 -0.3 -0.3 -0.3
  - largest error per slice: +0.1 +0.1 +0.1 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3 +0.3
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 0.132)
  - mean error in 10 equal slices of x0: +0.0 -0.0 +0.1 +0.1 +0.2 +0.3 +0.2 +0.3 +0.3 +0.4
  - largest error per slice: +0.5 +0.4 +0.2 +0.2 +0.3 +0.3 +0.3 +0.4 +0.4 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.001922 | 0.49% |
| scale | 1 | 1.007 | 0.41% |
| offset | 0 | 0.0003912 | 0.01% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0778, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 86.6 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 4.07 | 0 | 0.81 | 0.19 | 0.0483 | - | hinge |
| 2 | 180 | 0.849 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | 1.27 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | -2.56 | 0 | 0.00 | 1.00 | 0.244 | + | line |
| 6 | 180 | -2.52 | 0 | 0.50 | 0.50 | 0.248 | - | hinge |
| 7 | 0 | 6.6 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 8 | 180 | 0.302 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 0 | 0.904 | 0 | 0.17 | 0.83 | 0.0455 | + | line |
| 10 | 180 | 0.00568 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 11 | 180 | 4.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 180 | 0.756 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | 180 | 0.778 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 14 | 180 | -0.897 | 0 | 0.84 | 0.16 | 0.319 | - | hinge |
| 15 | 180 | -3.23 | 0 | 0.36 | 0.64 | 0.236 | - | hinge |
| 16 | 0 | 1.59 | 0 | 0.30 | 0.70 | 0.305 | - | hinge |
| 17 | 0 | 0.1 | 0 | 0.00 | 1.00 | 0.791 | + | line |
| 18 | 180 | -0.422 | 0 | 0.93 | 0.07 | 0.366 | - | hinge |
| 19 | 180 | -1.93 | 0 | 0.63 | 0.37 | 0.249 | - | hinge |
| 20 | 0 | 0.633 | 0 | 0.11 | 0.89 | 0.00166 | + | constant |
| 21 | 0 | 3.27 | 0 | 0.65 | 0.35 | 2.91e-05 | + | constant |
| 22 | 180 | -1.26 | 0 | 0.76 | 0.24 | 0.285 | - | hinge |
| 23 | 0 | 1.14 | 0 | 0.21 | 0.79 | 0.0436 | - | line |
| 24 | 180 | -0.171 | 0 | 0.99 | 0.01 | 0.209 | - | hinge |
| 25 | 0 | 7.33 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 26 | 0 | 0.424 | 0 | 0.07 | 0.93 | 0.688 | + | hinge |
| 27 | 180 | 0.557 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 180 | -0.274 | 0 | 0.97 | 0.03 | 0.286 | - | hinge |
| 29 | 180 | -0.624 | 0 | 0.89 | 0.11 | 0.319 | - | hinge |
| 30 | 180 | 1.52 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 31 | 0 | 5.08 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 20, 21, 25, 27, 30, 31 (magnitude 0.00169 together; a quarter of the tolerance is 0.0195)
- lines: units 5, 9, 17, 23 (each a constant or a line on the box; together one linear term)
- same line: units 18, 24, 26, 28, 29 (gate angle 180 degrees, offset -0.422)
- the rest: units 1, 6, 14, 15, 16, 19, 22
