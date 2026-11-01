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
    return 1.011 / (x0 + 0.006005) - 0.01262
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.8%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.6 +0.4 +0.5 +0.5 +0.5 +0.5 +0.5 +0.6 +0.6 +0.6
  - largest error per slice: +0.8 +0.5 +0.5 +0.5 +0.5 +0.5 +0.6 +0.6 +0.6 +0.6
- ripple (faithful minus net): largest 1.0%, at the box edge (x0 = 0.574)
  - mean error in 10 equal slices of x0: -0.3 -0.3 -0.3 -0.3 -0.3 -0.5 -0.2 -0.4 -0.5 -0.3
  - largest error per slice: +1.0 +0.8 +0.6 +0.5 +0.5 +0.5 +0.4 +0.5 +0.5 +0.7

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside reciprocal | 0 | 0.006005 | 1.38% |
| scale | 1 | 1.011 | 1.28% |
| offset | 0 | -0.01262 | 0.73% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0345, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.727 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | 180 | 1.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 2.02 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | 0.52 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | 4.63 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | 1.13 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 0 | 0.498 | 0 | 0.00 | 1.00 | 0.117 | - | line |
| 7 | 180 | -5.15 | 0 | 0.00 | 1.00 | 5.71e-05 | + | constant |
| 8 | 0 | 1.65 | 0 | 0.33 | 0.67 | 0.451 | + | hinge |
| 9 | 0 | 2.12 | 0 | 0.46 | 0.54 | 0.21 | + | hinge |
| 10 | 180 | 1.16 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 11 | 0 | 0.498 | 0 | 0.00 | 1.00 | 0.0991 | - | line |
| 12 | 180 | -3.93 | 0 | 0.02 | 0.98 | 0.65 | + | hinge |
| 13 | 180 | 1.22 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 14 | 180 | 0.767 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 15 | 180 | 2.97 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 16 | 180 | 0.215 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 0 | -0.161 | 0 | 0.00 | 1.00 | 0.167 | - | line |
| 18 | 0 | 0.424 | 0 | 0.00 | 1.00 | 0.224 | - | line |
| 19 | 0 | 0.499 | 0 | 0.00 | 1.00 | 0.182 | - | line |
| 20 | 0 | 7.2 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 21 | 180 | -1.22 | 0 | 0.79 | 0.21 | 0.314 | + | hinge |
| 22 | 0 | -1.96 | 0 | 0.00 | 1.00 | 0.0746 | + | line |
| 23 | 180 | -0.256 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 24 | 180 | 0.562 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 180 | -2.81 | 0 | 0.34 | 0.66 | 0.189 | + | hinge |
| 26 | 180 | -0.887 | 0 | 0.89 | 0.11 | 0.3 | + | hinge |
| 27 | 0 | 4.21 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 28 | 180 | -0.66 | 0 | 0.95 | 0.05 | 0.211 | + | hinge |
| 29 | 180 | -0.0925 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 30 | 180 | 1.55 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 31 | 0 | 1.22 | 0 | 0.21 | 0.79 | 0.00114 | - | constant |

## Groups

- negligible: units 0, 1, 2, 3, 4, 5, 7, 10, 13, 14, 15, 16, 20, 23, 24, 27, 29, 30, 31 (magnitude 0.00119 together; a quarter of the tolerance is 0.00862)
- lines: units 6, 11, 17, 18, 19, 22 (each a constant or a line on the box; together one linear term)
- the rest: units 8, 9, 12, 21, 25, 26, 28
