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
    return 1.004 * sqrt(x0 + 0.002309) - 0.005341
```

Clean:

```python
def net(x0):
    return sqrt(x0) + 0.02
```

% of the output range:
- distortion (clean minus faithful): largest 1.2%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: +1.0 +1.1 +1.0 +1.0 +1.0 +1.0 +0.9 +0.9 +0.9 +0.9
  - largest error per slice: +1.2 +1.1 +1.1 +1.0 +1.0 +1.0 +0.9 +0.9 +0.9 +0.9
- ripple (faithful minus net): largest 0.7%, at the box edge (x0 = 0.016)
  - mean error in 10 equal slices of x0: +0.0 -0.1 -0.0 -0.1 +0.0 +0.1 +0.1 +0.1 +0.3 +0.2
  - largest error per slice: +0.7 +0.4 +0.3 +0.3 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.002309 | 2.47% |
| offset | 0.02 | -0.005341 | 1.30% |
| scale | 1 | 1.004 | 0.41% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0389, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.727 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | 180 | 1.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 2.02 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | 0.52 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | 4.63 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | 1.13 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 0 | 2.38 | 0 | 0.60 | 0.40 | 0.00547 | - | constant |
| 7 | 0 | -0.299 | 0 | 0.00 | 1.00 | 0.45 | + | line |
| 8 | 180 | -1.29 | 0 | 0.68 | 0.32 | 0.129 | - | hinge |
| 9 | 0 | 3.35e-05 | 0 | 0.00 | 1.00 | 0.128 | + | line |
| 10 | 180 | 1.16 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 11 | 0 | 0.432 | 0 | 0.11 | 0.89 | 0.0191 | + | constant |
| 12 | 180 | -4.52 | 0 | 0.00 | 1.00 | 0.404 | + | line |
| 13 | 180 | 1.22 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 14 | 180 | 0.767 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 15 | 180 | 2.97 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 16 | 180 | 0.215 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | -1.55 | 0 | 0.61 | 0.39 | 1.65e-05 | - | constant |
| 18 | 0 | 4.85 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 19 | 0 | 0.0485 | 0 | 0.01 | 0.99 | 0.0546 | + | line |
| 20 | 0 | -0.00176 | 0 | 0.00 | 1.00 | 0.153 | + | line |
| 21 | 180 | -0.788 | 0 | 0.80 | 0.20 | 0.126 | - | hinge |
| 22 | 0 | -0.87 | 0 | 0.00 | 1.00 | 0.743 | + | line |
| 23 | 180 | -0.0447 | 0 | 0.99 | 0.01 | 0.102 | - | hinge |
| 24 | 180 | 0.562 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 180 | -3.39 | 0 | 0.15 | 0.85 | 0.0186 | - | constant |
| 26 | 180 | -0.176 | 0 | 0.96 | 0.04 | 0.112 | - | hinge |
| 27 | 0 | 2.84 | 0 | 0.71 | 0.29 | 0.0547 | - | hinge |
| 28 | 180 | -0.416 | 0 | 0.89 | 0.11 | 0.119 | - | hinge |
| 29 | 180 | -1.96 | 0 | 0.51 | 0.49 | 0.135 | - | hinge |
| 30 | 180 | 1.55 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 31 | 0 | 0.18 | 0 | 0.04 | 0.96 | 0.121 | + | line |

## Groups

- negligible: units 0, 1, 2, 3, 4, 5, 6, 10, 13, 14, 15, 16, 17, 18, 24, 30 (magnitude 0.00549 together; a quarter of the tolerance is 0.00973)
- lines: units 7, 9, 11, 12, 19, 20, 22, 25, 31 (each a constant or a line on the box; together one linear term)
- same line: units 23, 26 (gate angle 180 degrees, offset -0.0447)
- the rest: units 8, 21, 27, 28, 29
