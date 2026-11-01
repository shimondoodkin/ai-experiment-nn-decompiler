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
    return -0.003207 * x0 * x0 - 0.00297 * x1 * x1 + 0.9981 * x0 * x1 + 0.003385 * x0 - 0.002711 * x1 + 0.005012
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = -1.99, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.1 -0.1 -0.0 -0.0 +0.0 +0.0 +0.1 +0.1 +0.2 +0.2; largest: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.3 +0.4
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.0 -0.1 -0.1 -0.1; largest: +0.2 +0.3 +0.2 +0.2 +0.3 +0.2 +0.2 +0.4 +0.2 +0.1 +0.1 +0.1
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = 1.49, x1 = 1.95)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.1 +0.1 +0.1 +0.0 +0.0 +0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.2 +0.3 +0.3 +0.3 +0.3 +0.3 +0.4 +0.4 +0.4 +0.4
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.1 -0.1 -0.0 -0.0 +0.0 +0.0 -0.0 +0.1 +0.1 +0.1 +0.1; largest: +0.3 +0.4 +0.4 +0.3 +0.3 +0.2 +0.2 +0.4 +0.3 +0.3 +0.4 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0² | 0 | -0.003207 | 0.16% |
| x1² | 0 | -0.00297 | 0.15% |
| x0·x1 | 1 | 0.9981 | 0.10% |
| x0 | 0 | 0.003385 | 0.08% |
| x1 | 0 | -0.002711 | 0.07% |
| constant | 0 | 0.005012 | 0.06% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.158, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -137 | -0.538 | 0 | 0.33 | 0.67 | 0.227 | + | hinge |
| 1 | -134 | -0.377 | 0 | 0.38 | 0.62 | 0.359 | + | hinge |
| 2 | -57 | -1.73 | 0 | 0.07 | 0.93 | 0.215 | - | hinge |
| 3 | 43 | 1.08 | 0 | 0.81 | 0.19 | 0.559 | + | hinge |
| 4 | 47 | -1.84 | 0 | 0.06 | 0.94 | 0.234 | + | hinge |
| 5 | 134 | 2.09 | 0 | 0.97 | 0.03 | 0.218 | - | hinge |
| 6 | -165 | 2.78 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 7 | -146 | 5.1 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | 134 | -0.514 | 0 | 0.33 | 0.67 | 1.04 | - | hinge |
| 9 | 46 | -0.841 | 0 | 0.25 | 0.75 | 1.08 | + | hinge |
| 10 | -72 | 2.59 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 11 | -44 | 0.854 | 0 | 0.76 | 0.24 | 0.633 | - | hinge |
| 12 | -136 | 2.38 | 0 | 0.99 | 0.01 | 0.144 | + | hinge |
| 13 | 134 | 0.378 | 0 | 0.63 | 0.37 | 0.723 | - | hinge |
| 14 | -134 | -0.212 | 0 | 0.43 | 0.57 | 0.666 | + | hinge |
| 15 | 134 | -0.216 | 0 | 0.43 | 0.57 | 0.863 | - | hinge |
| 16 | -134 | 0.564 | 0 | 0.68 | 0.32 | 0.59 | + | hinge |
| 17 | -139 | 2.11 | 0 | 0.97 | 0.03 | 0.0401 | + | constant |
| 18 | -62 | 0.813 | 0 | 0.73 | 0.27 | 0.0635 | - | constant |
| 19 | 134 | 2.36 | 0 | 0.99 | 0.01 | 0.156 | - | hinge |
| 20 | -72 | 2.7 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 21 | -78 | -1.67 | 0 | 0.08 | 0.92 | 0.0286 | - | constant |
| 22 | -32 | 1.63 | 0 | 0.91 | 0.09 | 0.204 | - | hinge |
| 23 | -135 | -2.8 | 0 | 0.00 | 1.00 | 0.289 | - | line |
| 24 | 134 | 1.34 | 0 | 0.86 | 0.14 | 0.401 | - | hinge |
| 25 | -43 | -1 | 0 | 0.21 | 0.79 | 0.909 | - | hinge |
| 26 | 52 | 4.24 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 27 | 48 | 2.19 | 0 | 0.98 | 0.02 | 0.277 | + | hinge |
| 28 | 124 | -1.41 | 0 | 0.13 | 0.87 | 0.699 | - | hinge |
| 29 | -46 | -0.075 | 0 | 0.47 | 0.53 | 0.899 | - | hinge |
| 30 | 10 | 1.91 | 0 | 0.97 | 0.03 | 0.021 | + | constant |
| 31 | -134 | 1.14 | 0 | 0.82 | 0.18 | 0.473 | + | hinge |
| 32 | -135 | 1.42 | 0 | 0.88 | 0.12 | 0.364 | + | hinge |
| 33 | -52 | -1.01 | 0 | 0.21 | 0.79 | 0.581 | - | hinge |
| 34 | 42 | -1.62 | 0 | 0.09 | 0.91 | 0.252 | + | hinge |
| 35 | 72 | -1.82 | 0 | 0.05 | 0.95 | 0.332 | + | hinge |
| 36 | -153 | 3.18 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 37 | 16 | 3.14 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 38 | 58 | 1.25 | 0 | 0.84 | 0.16 | 0.173 | + | hinge |
| 39 | 50 | 1.73 | 0 | 0.93 | 0.07 | 0.41 | + | hinge |
| 40 | -134 | 0.0172 | 0 | 0.51 | 0.49 | 0.657 | + | hinge |
| 41 | -137 | 2.02 | 0 | 0.96 | 0.04 | 0.169 | + | hinge |
| 42 | -40 | 1.22 | 0 | 0.84 | 0.16 | 0.419 | - | hinge |
| 43 | -59 | 1.11 | 0 | 0.81 | 0.19 | 0.137 | - | hinge |
| 44 | -165 | 2.6 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 45 | 160 | -2.04 | 0 | 0.03 | 0.97 | 0.841 | - | hinge |
| 46 | 134 | 0.67 | 0 | 0.71 | 0.29 | 0.632 | - | hinge |
| 47 | 51 | -1.69 | 0 | 0.08 | 0.92 | 0.344 | + | hinge |
| 48 | 136 | 1.81 | 0 | 0.94 | 0.06 | 0.248 | - | hinge |
| 49 | -116 | -0.81 | 0 | 0.27 | 0.73 | 0.101 | + | hinge |
| 50 | 27 | -1.07 | 0 | 0.20 | 0.80 | 4.05e-11 | + | constant |
| 51 | -57 | 2.18 | 0 | 0.98 | 0.02 | 0.203 | - | hinge |
| 52 | 17 | 1.52 | 0 | 0.89 | 0.11 | 2.05e-15 | + | constant |
| 53 | -133 | -0.542 | 0 | 0.33 | 0.67 | 0.516 | + | hinge |
| 54 | -46 | -1.57 | 0 | 0.10 | 0.90 | 0.804 | - | hinge |
| 55 | 41 | -1.74 | 0 | 0.07 | 0.93 | 0.603 | + | hinge |
| 56 | -136 | -1.39 | 0 | 0.13 | 0.87 | 0.673 | + | hinge |
| 57 | -135 | -0.761 | 0 | 0.27 | 0.73 | 0.784 | + | hinge |
| 58 | 46 | -0.276 | 0 | 0.41 | 0.59 | 0.923 | + | hinge |
| 59 | 72 | -1.28 | 0 | 0.16 | 0.84 | 0.148 | + | hinge |
| 60 | -54 | 1.78 | 0 | 0.93 | 0.07 | 0.289 | - | hinge |
| 61 | -149 | 3.04 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 62 | -153 | -1.58 | 0 | 0.09 | 0.91 | 0.316 | + | hinge |
| 63 | -5 | 2.6 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 6, 7, 10, 20, 21, 26, 30, 36, 37, 44, 50, 52, 61, 63 (magnitude 0.0496 together; a quarter of the tolerance is 0.0396)
- lines: units 17, 18, 23 (each a constant or a line on the box; together one linear term)
- ring: 3 units (22, 48, 60) with gates at distance 1.63-1.81, switch width ~0, magnitude ~0.25 each, negative
- same line: units 0, 1, 14, 53 (gate angle -137 degrees, offset -0.538)
- same line: units 4, 34, 41, 47, 55 (gate angle 47 degrees, offset -1.84)
- same line: units 25, 33 (gate angle -43 degrees, offset -1)
- the rest: units 2, 3, 5, 8, 9, 11, 12, 13, 15, 16, 19, 24, 27, 28, 29, 31, 32, 35, 38, 39, 40, 42, 43, 45, 46, 49, 51, 54, 56, 57, 58, 59, 62
