# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (7 nodes):

```python
def net(x0, x1):
    return x0 * x1
```

## Faithful fit

The quadratic formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no quadratic formula follows).

Faithful:

```python
def net(x0, x1):
    return -0.0009795 * x0 * x0 - 0.001726 * x1 * x1 + 0.9966 * x0 * x1 + 0.005168 * x0 + 0.003494 * x1 - 0.003962
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.6%, at the box edge (x0 = -1.99, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2; largest: +0.1 +0.1 +0.1 +0.2 +0.2 +0.3 +0.4 +0.4 +0.5 +0.6
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.1 +0.1 +0.2 +0.3 +0.2 +0.1 +0.0 -0.0; largest: +0.0 +0.1 +0.1 +0.1 +0.1 +0.2 +0.4 +0.6 +0.4 +0.2 +0.1 +0.1
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.1 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3 +0.4 +0.4
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 -0.1 -0.1 -0.0 +0.1 -0.0 -0.1 -0.1 -0.1 -0.1 +0.0 +0.0; largest: +0.2 +0.3 +0.3 +0.3 +0.4 +0.2 +0.3 +0.4 +0.3 +0.3 +0.3 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9966 | 0.17% |
| x0 | 0 | 0.005168 | 0.13% |
| x1 | 0 | 0.003494 | 0.09% |
| x1² | 0 | -0.001726 | 0.09% |
| constant | 0 | -0.003962 | 0.05% |
| x0² | 0 | -0.0009795 | 0.05% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.157, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -49 | -0.856 | 0 | 0.24 | 0.76 | 0.589 | - | hinge |
| 1 | 116 | -1.38 | 0 | 0.13 | 0.87 | 0.152 | - | hinge |
| 2 | 142 | 1.65 | 0 | 0.91 | 0.09 | 0.312 | - | hinge |
| 3 | 135 | -0.126 | 0 | 0.46 | 0.54 | 0.851 | - | hinge |
| 4 | 141 | 3.84 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 5 | 99 | 2.42 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | -56 | 0.816 | 0 | 0.74 | 0.26 | 1.56e-18 | - | constant |
| 7 | 46 | 3 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | -3 | 2.71 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 6 | 2.33 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | -136 | 0.848 | 0 | 0.75 | 0.25 | 0.363 | + | hinge |
| 11 | -93 | 3.58 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | -124 | 2.22 | 0 | 0.98 | 0.02 | 0.142 | + | hinge |
| 13 | 45 | 0.282 | 0 | 0.60 | 0.40 | 0.703 | + | hinge |
| 14 | 135 | 1.24 | 0 | 0.84 | 0.16 | 0.591 | - | hinge |
| 15 | 54 | -1.02 | 0 | 0.21 | 0.79 | 0.281 | + | hinge |
| 16 | -49 | -0.498 | 0 | 0.34 | 0.66 | 0.345 | - | hinge |
| 17 | -135 | 0.301 | 0 | 0.60 | 0.40 | 0.733 | + | hinge |
| 18 | -17 | -2.04 | 0 | 0.02 | 0.98 | 0.167 | - | line |
| 19 | -46 | -2.41 | 0 | 0.01 | 0.99 | 0.301 | - | line |
| 20 | -42 | -0.865 | 0 | 0.24 | 0.76 | 0.827 | - | hinge |
| 21 | 27 | -1.83 | 0 | 0.06 | 0.94 | 0.224 | + | hinge |
| 22 | 126 | 1.67 | 0 | 0.92 | 0.08 | 0.252 | - | hinge |
| 23 | -168 | -1.72 | 0 | 0.06 | 0.94 | 0.00331 | + | constant |
| 24 | -131 | -0.862 | 0 | 0.24 | 0.76 | 0.55 | + | hinge |
| 25 | -137 | 0.948 | 0 | 0.78 | 0.22 | 0.0471 | + | constant |
| 26 | -44 | 1.16 | 0 | 0.83 | 0.17 | 0.535 | - | hinge |
| 27 | 45 | 0.561 | 0 | 0.68 | 0.32 | 0.631 | + | hinge |
| 28 | -52 | 1.88 | 0 | 0.95 | 0.05 | 0.225 | - | hinge |
| 29 | 42 | 0.876 | 0 | 0.76 | 0.24 | 0.377 | + | hinge |
| 30 | 132 | -0.436 | 0 | 0.36 | 0.64 | 0.544 | - | hinge |
| 31 | 45 | 1.54 | 0 | 0.90 | 0.10 | 0.318 | + | hinge |
| 32 | -137 | 1.12 | 0 | 0.82 | 0.18 | 0.42 | + | hinge |
| 33 | 121 | -0.922 | 0 | 0.23 | 0.77 | 0.031 | - | constant |
| 34 | -130 | -1.21 | 0 | 0.16 | 0.84 | 0.612 | + | hinge |
| 35 | 120 | 2.89 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 36 | 40 | 2.04 | 0 | 0.96 | 0.04 | 0.188 | + | hinge |
| 37 | 45 | -0.00426 | 0 | 0.50 | 0.50 | 0.843 | + | hinge |
| 38 | 42 | 1.77 | 0 | 0.93 | 0.07 | 0.192 | + | hinge |
| 39 | 137 | 0.483 | 0 | 0.66 | 0.34 | 0.621 | - | hinge |
| 40 | -135 | 0.592 | 0 | 0.69 | 0.31 | 0.64 | + | hinge |
| 41 | -126 | 6.3 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 42 | -45 | -0.159 | 0 | 0.44 | 0.56 | 0.861 | - | hinge |
| 43 | 138 | -0.82 | 0 | 0.25 | 0.75 | 0.816 | - | hinge |
| 44 | 131 | -0.804 | 0 | 0.26 | 0.74 | 0.477 | - | hinge |
| 45 | -94 | 3.08 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 46 | 89 | 27 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 47 | -135 | -2.2 | 0 | 0.02 | 0.98 | 0.0473 | + | constant |
| 48 | 77 | 2.5 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 49 | -121 | -1.87 | 0 | 0.05 | 0.95 | 0.534 | + | hinge |
| 50 | -45 | 2.29 | 0 | 0.98 | 0.02 | 0.216 | - | hinge |
| 51 | 45 | -1.65 | 0 | 0.09 | 0.91 | 0.42 | + | hinge |
| 52 | -133 | 1.88 | 0 | 0.94 | 0.06 | 0.291 | + | hinge |
| 53 | -35 | 1.88 | 0 | 0.95 | 0.05 | 0.166 | - | hinge |
| 54 | -2 | 2.45 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 55 | 46 | 2.4 | 0 | 0.99 | 0.01 | 0.137 | + | hinge |
| 56 | -42 | 0.467 | 0 | 0.65 | 0.35 | 0.467 | - | hinge |
| 57 | 53 | -1.42 | 0 | 0.12 | 0.88 | 0.338 | + | hinge |
| 58 | 41 | 1.24 | 0 | 0.84 | 0.16 | 0.329 | + | hinge |
| 59 | -137 | 1.45 | 0 | 0.88 | 0.12 | 0.362 | + | hinge |
| 60 | 135 | 2.15 | 0 | 0.97 | 0.03 | 0.301 | - | hinge |
| 61 | -44 | 1.5 | 0 | 0.89 | 0.11 | 0.432 | - | hinge |
| 62 | -74 | -2.04 | 0 | 0.02 | 0.98 | 0.0819 | - | line |
| 63 | -154 | 2.21 | 0 | 0.98 | 0.02 | 0.0854 | + | hinge |

## Groups

- negligible: units 4, 5, 6, 7, 8, 9, 11, 23, 33, 35, 41, 45, 46, 48, 54 (magnitude 0.0343 together; a quarter of the tolerance is 0.0393)
- lines: units 18, 19, 25, 47, 62 (each a constant or a line on the box; together one linear term)
- ring: 5 units (2, 22, 28, 53, 61) with gates at distance 1.50-1.88, switch width ~0, magnitude ~0.25 each, negative
- same line: units 0, 20 (gate angle -49 degrees, offset -0.856)
- same line: units 10, 15 (gate angle -136 degrees, offset 0.848)
- same line: units 16, 39 (gate angle -49 degrees, offset -0.498)
- same line: units 24, 29 (gate angle -131 degrees, offset -0.862)
- same line: units 30, 56 (gate angle 132 degrees, offset -0.436)
- same line: units 34, 58 (gate angle -130 degrees, offset -1.21)
- same line: units 43, 44 (gate angle 138 degrees, offset -0.82)
- same line: units 51, 57, 59 (gate angle 45 degrees, offset -1.65)
- the rest: units 1, 3, 12, 13, 14, 17, 21, 26, 27, 31, 32, 36, 37, 38, 40, 42, 49, 50, 52, 55, 60, 63
