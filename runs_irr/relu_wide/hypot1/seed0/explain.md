# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (21 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 1)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(0.9975 * x0 * x0 + 1.001 * x1 * x1 - 0.009499 * x0 * x1 + 0.0003614 * x0 - 0.005193 * x1 + 1.003) - 0.006293
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 1)
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, at the box edge (x0 = 2, x1 = 1.98)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.2 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3; largest: +0.3 +0.3 +0.4 +0.4 +0.5 +0.6 +0.6 +0.7 +0.7 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.4 +0.5 +0.4 +0.3 +0.2 +0.3 +0.4 +0.4 +0.2 +0.1 +0.1 +0.2; largest: +0.6 +0.7 +0.6 +0.4 +0.3 +0.4 +0.5 +0.6 +0.4 +0.2 +0.2 +0.4
- ripple (faithful minus net): largest 0.5%, inside the box (x0 = 0.229, x1 = 0.256)
  - by distance from (-0.000169, 0.00259), rings 0.282 wide, mean: -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.0 -0.0 -0.0 +0.0; largest: +0.4 +0.5 +0.4 +0.5 +0.4 +0.4 +0.4 +0.4 +0.5 +0.4
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.2 -0.1 -0.0 +0.0 -0.0 -0.1 -0.1 -0.0 +0.1 +0.1 -0.0; largest: +0.4 +0.5 +0.5 +0.3 +0.4 +0.4 +0.4 +0.4 +0.5 +0.5 +0.5 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 inside sqrt | 0 | -0.009499 | 0.32% |
| offset | 0 | -0.006293 | 0.31% |
| x1 inside sqrt | 0 | -0.005193 | 0.12% |
| x0² inside sqrt | 1 | 0.9975 | 0.11% |
| constant inside sqrt | 1 | 1.003 | 0.07% |
| x1² inside sqrt | 1 | 1.001 | 0.04% |
| x0 inside sqrt | 0 | 0.0003614 | 0.01% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0399, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 130 | 1.62 | 0 | 0.91 | 0.09 | 0.0433 | + | hinge |
| 1 | -142 | 1.08 | 0 | 0.81 | 0.19 | 0.0623 | + | hinge |
| 2 | 142 | 1.99 | 0 | 0.96 | 0.04 | 0.0113 | + | constant |
| 3 | 90 | 1.15 | 0 | 0.79 | 0.21 | 0.00242 | + | constant |
| 4 | 91 | 4.1 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 5 | -170 | 0.284 | 0 | 0.57 | 0.43 | 0.0865 | + | hinge |
| 6 | -146 | -0.422 | 0 | 0.37 | 0.63 | 0.0988 | + | hinge |
| 7 | 166 | 0.23 | 0 | 0.56 | 0.44 | 0.168 | + | hinge |
| 8 | 62 | 0.0271 | 0 | 0.51 | 0.49 | 0.303 | + | hinge |
| 9 | -154 | 0.135 | 0 | 0.54 | 0.46 | 0.264 | + | hinge |
| 10 | 63 | 0.598 | 0 | 0.67 | 0.33 | 0.0837 | + | hinge |
| 11 | 96 | -0.875 | 0 | 0.28 | 0.72 | 0.136 | + | hinge |
| 12 | 45 | 1.72 | 0 | 0.93 | 0.07 | 2.29e-16 | + | constant |
| 13 | -47 | -0.375 | 0 | 0.38 | 0.62 | 0.428 | + | hinge |
| 14 | -152 | -1.36 | 0 | 0.14 | 0.86 | 0.111 | + | hinge |
| 15 | 98 | -0.11 | 0 | 0.47 | 0.53 | 0.263 | + | hinge |
| 16 | -145 | 1.43 | 0 | 0.88 | 0.12 | 0.0579 | + | hinge |
| 17 | -121 | 0.546 | 0 | 0.66 | 0.34 | 0.182 | + | hinge |
| 18 | 133 | -0.593 | 0 | 0.31 | 0.69 | 0.345 | + | hinge |
| 19 | 22 | 0.842 | 0 | 0.73 | 0.27 | 0.0641 | + | hinge |
| 20 | -48 | 0.975 | 0 | 0.78 | 0.22 | 0.13 | + | hinge |
| 21 | -54 | 1.4 | 0 | 0.87 | 0.13 | 0.0756 | + | hinge |
| 22 | -139 | 2.25 | 0 | 0.98 | 0.02 | 0.0162 | + | constant |
| 23 | 25 | 1.08 | 0 | 0.80 | 0.20 | 0.0775 | + | hinge |
| 24 | -178 | 2.13 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 61 | -0.245 | 0 | 0.43 | 0.57 | 0.275 | + | hinge |
| 26 | 33 | -1.8 | 0 | 0.06 | 0.94 | 0.188 | + | hinge |
| 27 | -67 | 1.82 | 0 | 0.94 | 0.06 | 0.0297 | + | hinge |
| 28 | -81 | -0.594 | 0 | 0.35 | 0.65 | 0.205 | + | hinge |
| 29 | -155 | 0.848 | 0 | 0.73 | 0.27 | 0.118 | + | hinge |
| 30 | 30 | 0.179 | 0 | 0.55 | 0.45 | 0.129 | + | hinge |
| 31 | 165 | 0.614 | 0 | 0.66 | 0.34 | 0.0569 | + | hinge |
| 32 | -14 | 0.566 | 0 | 0.65 | 0.35 | 0.0998 | + | hinge |
| 33 | 65 | 0.812 | 0 | 0.73 | 0.27 | 0.08 | + | hinge |
| 34 | 41 | 2.15 | 0 | 0.97 | 0.03 | 0.0225 | + | hinge |
| 35 | -139 | 3.3 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 36 | 73 | 1.42 | 0 | 0.87 | 0.13 | 0.0143 | + | constant |
| 37 | 63 | 0.334 | 0 | 0.60 | 0.40 | 0.247 | + | hinge |
| 38 | -82 | 0.467 | 0 | 0.62 | 0.38 | 0.168 | + | hinge |
| 39 | -81 | -0.219 | 0 | 0.44 | 0.56 | 0.276 | + | hinge |
| 40 | 133 | 0.735 | 0 | 0.73 | 0.27 | 0.194 | + | hinge |
| 41 | -48 | -0.0314 | 0 | 0.49 | 0.51 | 0.327 | + | hinge |
| 42 | -9 | -2.94 | 0 | 0.00 | 1.00 | 0.227 | + | line |
| 43 | -85 | 2.19 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 44 | -52 | -1.17 | 0 | 0.17 | 0.83 | 0.295 | + | hinge |
| 45 | 87 | 2.13 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 46 | 39 | 2.47 | 0 | 0.99 | 0.01 | 0.00517 | + | constant |
| 47 | -162 | -0.354 | 0 | 0.41 | 0.59 | 0.289 | + | hinge |
| 48 | 15 | 0.0451 | 0 | 0.51 | 0.49 | 0.189 | + | hinge |
| 49 | -155 | 0.47 | 0 | 0.63 | 0.37 | 0.23 | + | hinge |
| 50 | -3 | -0.751 | 0 | 0.31 | 0.69 | 0.0888 | + | hinge |
| 51 | -136 | 3.44 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 52 | 166 | -0.138 | 0 | 0.47 | 0.53 | 0.23 | + | hinge |
| 53 | -119 | 0.912 | 0 | 0.76 | 0.24 | 0.0491 | + | hinge |
| 54 | 63 | -3.33 | 0 | 0.00 | 1.00 | 0.184 | + | line |
| 55 | 91 | 2.11 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 56 | 134 | -0.285 | 0 | 0.40 | 0.60 | 0.368 | + | hinge |
| 57 | -42 | -0.0184 | 0 | 0.49 | 0.51 | 0.127 | + | hinge |
| 58 | -129 | 3.04 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 59 | 70 | 1.14 | 0 | 0.80 | 0.20 | 0.0264 | + | hinge |
| 60 | 30 | 1.68 | 0 | 0.92 | 0.08 | 0.0327 | + | hinge |
| 61 | 154 | 1.32 | 0 | 0.85 | 0.15 | 1.65e-05 | + | constant |
| 62 | -159 | -0.639 | 0 | 0.33 | 0.67 | 0.187 | + | hinge |
| 63 | -80 | 2.35 | 0 | 1.00 | 0.00 | 0 | + | constant |

## Groups

- negligible: units 3, 4, 12, 24, 35, 43, 45, 46, 51, 55, 58, 61, 63 (magnitude 0.0076 together; a quarter of the tolerance is 0.00998)
- lines: units 2, 22, 36, 42, 54 (each a constant or a line on the box; together one linear term)
- ring: 4 units (10, 17, 31, 32) with gates at distance 0.55-0.61, switch width ~0, magnitude ~0.092 each, positive
- ring: 5 units (19, 20, 29, 33, 40) with gates at distance 0.73-0.98, switch width ~0, magnitude ~0.12 each, positive
- ring: 5 units (0, 16, 21, 27, 60) with gates at distance 1.40-1.82, switch width ~0, magnitude ~0.043 each, positive
- same line: units 41, 57 (gate angle -48 degrees, offset -0.0314)
- the rest: units 1, 5, 6, 7, 8, 9, 11, 13, 14, 15, 18, 23, 25, 26, 28, 30, 34, 37, 38, 39, 44, 47, 48, 49, 50, 52, 53, 56, 59, 62
