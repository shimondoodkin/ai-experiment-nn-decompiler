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
    return 0.002822 * x0 * x0 + 0.002484 * x1 * x1 + 0.9965 * x0 * x1 - 0.002039 * x0 + 0.001766 * x1 - 0.006895
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.1 +0.1 +0.1 +0.1 +0.0 +0.0 -0.0 -0.1 -0.1 -0.1; largest: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.3 +0.4 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 +0.1 +0.0 -0.0 -0.1 -0.1 +0.0 +0.0 +0.1 +0.0 -0.0 +0.0; largest: +0.1 +0.1 +0.1 +0.3 +0.5 +0.3 +0.1 +0.1 +0.1 +0.1 +0.3 +0.1
- ripple (faithful minus net): largest 0.4%, inside the box (x0 = 1.78, x1 = -1.75)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.0 +0.0 +0.0 +0.0; largest: +0.2 +0.2 +0.3 +0.2 +0.2 +0.2 +0.2 +0.3 +0.4 +0.3
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.1 -0.1 -0.0 +0.0 -0.0 -0.1 -0.1 -0.1 -0.0 +0.0 -0.0; largest: +0.2 +0.2 +0.2 +0.2 +0.3 +0.2 +0.2 +0.2 +0.2 +0.2 +0.4 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9965 | 0.18% |
| x0² | 0 | 0.002822 | 0.14% |
| x1² | 0 | 0.002484 | 0.13% |
| constant | 0 | -0.006895 | 0.09% |
| x0 | 0 | -0.002039 | 0.05% |
| x1 | 0 | 0.001766 | 0.04% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.158, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 135 | 2.31 | 0 | 0.98 | 0.02 | 0.207 | - | hinge |
| 1 | -132 | 0.949 | 0 | 0.78 | 0.22 | 0.287 | + | hinge |
| 2 | 135 | 0.011 | 0 | 0.51 | 0.49 | 0.728 | - | hinge |
| 3 | 88 | 2.11 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | 127 | 1.88 | 0 | 0.94 | 0.06 | 0.208 | - | hinge |
| 5 | -124 | -0.921 | 0 | 0.23 | 0.77 | 0.111 | + | hinge |
| 6 | -142 | -1.56 | 0 | 0.10 | 0.90 | 0.526 | + | hinge |
| 7 | -172 | 2.29 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | 45 | 0.247 | 0 | 0.59 | 0.41 | 0.545 | + | hinge |
| 9 | -135 | 0.211 | 0 | 0.57 | 0.43 | 0.589 | + | hinge |
| 10 | 87 | 2.26 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 11 | 126 | -0.951 | 0 | 0.22 | 0.78 | 0.212 | - | hinge |
| 12 | -82 | 2.33 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 13 | -40 | -1.16 | 0 | 0.17 | 0.83 | 0.45 | - | hinge |
| 14 | 164 | 2.5 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 87 | 2.14 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 16 | -138 | 1.39 | 0 | 0.87 | 0.13 | 0.265 | + | hinge |
| 17 | -135 | 0.685 | 0 | 0.71 | 0.29 | 0.501 | + | hinge |
| 18 | 133 | -1.99 | 0 | 0.04 | 0.96 | 0.289 | - | hinge |
| 19 | 50 | 1.52 | 0 | 0.90 | 0.10 | 0.185 | + | hinge |
| 20 | -46 | 2.26 | 0 | 0.98 | 0.02 | 0.251 | - | hinge |
| 21 | -50 | 1.73 | 0 | 0.92 | 0.08 | 0.341 | - | hinge |
| 22 | -135 | 1.88 | 0 | 0.94 | 0.06 | 0.226 | + | hinge |
| 23 | 44 | 1 | 0 | 0.79 | 0.21 | 0.405 | + | hinge |
| 24 | 143 | 1.88 | 0 | 0.94 | 0.06 | 0.189 | - | hinge |
| 25 | 55 | -1.29 | 0 | 0.15 | 0.85 | 0.267 | + | hinge |
| 26 | 43 | -1.58 | 0 | 0.10 | 0.90 | 0.503 | + | hinge |
| 27 | -44 | 0.933 | 0 | 0.78 | 0.22 | 0.596 | - | hinge |
| 28 | -47 | -1.13 | 0 | 0.18 | 0.82 | 0.788 | - | hinge |
| 29 | -139 | 0.969 | 0 | 0.78 | 0.22 | 0.274 | + | hinge |
| 30 | 45 | 0.0188 | 0 | 0.51 | 0.49 | 0.671 | + | hinge |
| 31 | 138 | 1.49 | 0 | 0.89 | 0.11 | 0.372 | - | hinge |
| 32 | -33 | 1.75 | 0 | 0.93 | 0.07 | 0.143 | - | hinge |
| 33 | 46 | 1.74 | 0 | 0.93 | 0.07 | 0.156 | + | hinge |
| 34 | 45 | 1.94 | 0 | 0.95 | 0.05 | 0.209 | + | hinge |
| 35 | -136 | 2.43 | 0 | 0.99 | 0.01 | 0.12 | + | hinge |
| 36 | 63 | 2.72 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 37 | 45 | 0.465 | 0 | 0.65 | 0.35 | 0.515 | + | hinge |
| 38 | -48 | 1.32 | 0 | 0.86 | 0.14 | 0.445 | - | hinge |
| 39 | -49 | 3.21 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 40 | 135 | 0.846 | 0 | 0.76 | 0.24 | 0.565 | - | hinge |
| 41 | -47 | 0.56 | 0 | 0.68 | 0.32 | 0.536 | - | hinge |
| 42 | -169 | 2.58 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 43 | -87 | 2.62 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 44 | -54 | -1.45 | 0 | 0.12 | 0.88 | 0.41 | - | hinge |
| 45 | 71 | -1.9 | 0 | 0.04 | 0.96 | 0.0571 | + | constant |
| 46 | 46 | 2.44 | 0 | 0.99 | 0.01 | 0.109 | + | hinge |
| 47 | -136 | -0.775 | 0 | 0.26 | 0.74 | 0.722 | + | hinge |
| 48 | 48 | 1.26 | 0 | 0.85 | 0.15 | 0.264 | + | hinge |
| 49 | -135 | 0.445 | 0 | 0.64 | 0.36 | 0.582 | + | hinge |
| 50 | 44 | -1.67 | 0 | 0.08 | 0.92 | 0.464 | + | hinge |
| 51 | -136 | 2.14 | 0 | 0.97 | 0.03 | 0.178 | + | hinge |
| 52 | -137 | 3.02 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 53 | -136 | 1.19 | 0 | 0.83 | 0.17 | 0.226 | + | hinge |
| 54 | 59 | -1.59 | 0 | 0.09 | 0.91 | 0.192 | + | hinge |
| 55 | 135 | -0.245 | 0 | 0.42 | 0.58 | 0.798 | - | hinge |
| 56 | 144 | -1.3 | 0 | 0.15 | 0.85 | 0.421 | - | hinge |
| 57 | -45 | -0.28 | 0 | 0.40 | 0.60 | 0.871 | - | hinge |
| 58 | 135 | 0.56 | 0 | 0.68 | 0.32 | 0.637 | - | hinge |
| 59 | 45 | 0.625 | 0 | 0.70 | 0.30 | 0.218 | + | hinge |
| 60 | 44 | 2.19 | 0 | 0.98 | 0.02 | 0.147 | + | hinge |
| 61 | 139 | -0.566 | 0 | 0.32 | 0.68 | 0.457 | - | hinge |
| 62 | -142 | -1.29 | 0 | 0.15 | 0.85 | 0.477 | + | hinge |
| 63 | 101 | 2.34 | 0 | 1.00 | 0.00 | 0 | + | constant |

## Groups

- negligible: units 3, 7, 10, 12, 14, 15, 36, 39, 42, 43, 52, 63 (magnitude 0 together; a quarter of the tolerance is 0.0394)
- lines: units 45 (each a constant or a line on the box; together one linear term)
- ring: 3 units (1, 23, 29) with gates at distance 0.95-1.00, switch width ~0, magnitude ~0.29 each, positive
- ring: 5 units (22, 33, 34, 51, 60) with gates at distance 1.74-2.19, switch width ~0, magnitude ~0.18 each, positive
- ring: 4 units (4, 21, 24, 32) with gates at distance 1.73-1.88, switch width ~0, magnitude ~0.2 each, negative
- same line: units 11, 27 (gate angle 126 degrees, offset -0.951)
- same line: units 13, 28 (gate angle -40 degrees, offset -1.16)
- same line: units 16, 26, 50, 53 (gate angle -138 degrees, offset 1.39)
- same line: units 37, 47, 59 (gate angle 45 degrees, offset 0.465)
- same line: units 41, 61 (gate angle -47 degrees, offset 0.56)
- same line: units 48, 62 (gate angle 48 degrees, offset 1.26)
- the rest: units 0, 2, 5, 6, 8, 9, 17, 18, 19, 20, 25, 30, 31, 35, 38, 40, 44, 46, 49, 54, 55, 56, 57, 58
