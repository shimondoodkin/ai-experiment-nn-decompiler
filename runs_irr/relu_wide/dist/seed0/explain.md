# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (18 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.007 * x0 * x0 + 1.007 * x1 * x1 + 0.000406 * x0 * x1 + 0.00228 * x0 - 0.001974 * x1 + 0.001583) - 0.0004637
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0, 0), rings 0.423 wide, mean: -0.1 -0.1 -0.1 -0.1 -0.2 -0.2 -0.2 -0.3 -0.3 -0.3; largest: +0.7 +0.1 +0.1 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3 +0.4
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.2 -0.2 -0.1 -0.2 -0.1 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2; largest: +0.5 +0.6 +0.3 +0.7 +0.4 +0.3 +0.4 +0.7 +0.3 +0.3 +0.4 +0.6
- ripple (faithful minus net): largest 0.2%, at the box edge (x0 = 0.666, x1 = 2.95)
  - by distance from (-0.00113, 0.00098), rings 0.423 wide, mean: +0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.1 +0.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.2 +0.1 +0.2 +0.1 +0.1 +0.1 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.001583 | 0.53% |
| x0² inside sqrt | 1 | 1.007 | 0.25% |
| x1² inside sqrt | 1 | 1.007 | 0.25% |
| x0 inside sqrt | 0 | 0.00228 | 0.03% |
| x1 inside sqrt | 0 | -0.001974 | 0.02% |
| offset | 0 | -0.0004637 | 0.01% |
| x0·x1 inside sqrt | 0 | 0.000406 | 0.01% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0844, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 92 | 3.13 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | -125 | 0.0386 | 0 | 0.51 | 0.49 | 0.364 | + | hinge |
| 2 | 154 | 4.03 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 81 | 0.0107 | 0 | 0.50 | 0.50 | 0.0981 | + | hinge |
| 4 | 67 | 5.94 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 176 | 0.00291 | 0 | 0.50 | 0.50 | 0.142 | + | hinge |
| 6 | -179 | 3.02 | 0 | 1.00 | 0.00 | 5.38e-10 | - | constant |
| 7 | 175 | -0.00382 | 0 | 0.50 | 0.50 | 0.204 | + | hinge |
| 8 | 67 | 0.102 | 0 | 0.52 | 0.48 | 0.201 | + | hinge |
| 9 | -170 | 0.000956 | 0 | 0.50 | 0.50 | 0.427 | + | hinge |
| 10 | 84 | 0.0619 | 0 | 0.51 | 0.49 | 0.0362 | + | constant |
| 11 | 100 | 3.5 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 37 | 4.2 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | -40 | -0.0265 | 0 | 0.49 | 0.51 | 0.539 | + | hinge |
| 14 | 0 | 3.01 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 15 | 115 | -0.000156 | 0 | 0.50 | 0.50 | 0.658 | + | hinge |
| 16 | -168 | 3.74 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | -113 | 0.0441 | 0 | 0.51 | 0.49 | 0.458 | + | hinge |
| 18 | 129 | 2.09 | 0 | 0.87 | 0.13 | 0.000126 | + | constant |
| 19 | 55 | 0.0726 | 0 | 0.52 | 0.48 | 0.154 | + | hinge |
| 20 | -74 | 8.09e-06 | 0 | 0.50 | 0.50 | 0.544 | + | hinge |
| 21 | -40 | 0.142 | 0 | 0.53 | 0.47 | 0.111 | + | hinge |
| 22 | -166 | 3.66 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 23 | 40 | 0.0678 | 0 | 0.52 | 0.48 | 0.242 | + | hinge |
| 24 | -180 | 3.02 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 25 | 78 | -0.00225 | 0 | 0.50 | 0.50 | 0.212 | + | hinge |
| 26 | 16 | -0.0101 | 0 | 0.50 | 0.50 | 0.166 | + | hinge |
| 27 | -41 | 4.25 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 28 | -81 | 8.22e-05 | 0 | 0.50 | 0.50 | 0.282 | + | hinge |
| 29 | -139 | 0.0553 | 0 | 0.51 | 0.49 | 0.299 | + | hinge |
| 30 | 34 | -0.000454 | 0 | 0.50 | 0.50 | 0.454 | + | hinge |
| 31 | 158 | -0.000571 | 0 | 0.50 | 0.50 | 0.583 | + | hinge |
| 32 | -0 | 3.01 | 0 | 1.00 | 0.00 | 5.71e-09 | + | constant |
| 33 | 83 | -0.0122 | 0 | 0.50 | 0.50 | 0.234 | + | hinge |
| 34 | 82 | 3.38 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 35 | -171 | 3.44 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 36 | 74 | -0.00168 | 0 | 0.50 | 0.50 | 0.323 | + | hinge |
| 37 | 48 | 0.00314 | 0 | 0.50 | 0.50 | 0.553 | + | hinge |
| 38 | -88 | 0.0645 | 0 | 0.51 | 0.49 | 0.277 | + | hinge |
| 39 | -89 | -0.0631 | 0 | 0.49 | 0.51 | 0.281 | + | hinge |
| 40 | 113 | 3.97 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 41 | -48 | -0.00166 | 0 | 0.50 | 0.50 | 0.578 | + | hinge |
| 42 | -2 | -2.87 | 0 | 0.02 | 0.98 | 0.0166 | + | constant |
| 43 | -89 | 3.07 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 44 | 79 | 4.83 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 45 | 84 | 3.28 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 46 | 33 | 4.16 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 47 | -154 | -0.067 | 0 | 0.49 | 0.51 | 0.33 | + | hinge |
| 48 | 2 | 0.00174 | 0 | 0.50 | 0.50 | 0.468 | + | hinge |
| 49 | -155 | 0.0644 | 0 | 0.51 | 0.49 | 0.35 | + | hinge |
| 50 | 18 | 0.0102 | 0 | 0.50 | 0.50 | 0.287 | + | hinge |
| 51 | -140 | 4.53 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 52 | 167 | 0.000697 | 0 | 0.50 | 0.50 | 0.61 | + | hinge |
| 53 | -178 | 3.17 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 54 | 124 | -3.41 | 0 | 0.02 | 0.98 | 0.00108 | + | constant |
| 55 | 97 | 3.42 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 56 | 124 | 0.000736 | 0 | 0.50 | 0.50 | 0.635 | + | hinge |
| 57 | -31 | 0.000918 | 0 | 0.50 | 0.50 | 0.599 | + | hinge |
| 58 | -138 | 4.85 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 59 | 60 | 0.00388 | 0 | 0.50 | 0.50 | 0.212 | + | hinge |
| 60 | 2 | 3.11 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 61 | 154 | 0.0321 | 0 | 0.51 | 0.49 | 0.0562 | + | hinge |
| 62 | 178 | 3.09 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 63 | -84 | 3.29 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 0, 2, 4, 6, 11, 12, 14, 16, 18, 22, 24, 27, 32, 34, 35, 40, 42, 43, 44, 45, 46, 51, 53, 54, 55, 58, 60, 62, 63 (magnitude 0.0178 together; a quarter of the tolerance is 0.0211)
- lines: units 10 (each a constant or a line on the box; together one linear term)
- ring: 4 units (19, 23, 38, 49) with gates at distance 0.06-0.07, switch width ~0, magnitude ~0.26 each, positive
- same line: units 1, 3, 5, 7, 8, 9, 13, 15, 17, 20, 21, 25, 26, 28, 29, 30, 31, 33, 36, 37, 39, 41, 47, 48, 50, 52, 56, 57, 59, 61 (gate angle -125 degrees, offset 0.0386)
