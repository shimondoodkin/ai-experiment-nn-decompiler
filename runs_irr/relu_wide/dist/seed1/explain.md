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
    return sqrt(1.001 * x0 * x0 + 1.002 * x1 * x1 + 0.0002484 * x0 * x1 - 0.0065 * x0 - 0.001612 * x1 + 0.001711) - 0.00252
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0, 0), rings 0.423 wide, mean: -0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 -0.0 -0.0 -0.0; largest: +0.7 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 +0.1 +0.0 -0.0 -0.0 -0.0 -0.1 -0.1 -0.0 +0.0 +0.0 +0.1; largest: +0.4 +0.6 +0.3 +0.7 +0.4 +0.4 +0.4 +0.7 +0.3 +0.3 +0.2 +0.5
- ripple (faithful minus net): largest 0.2%, inside the box (x0 = -0.0905, x1 = -0.00709)
  - by distance from (0.00325, 0.000804), rings 0.423 wide, mean: -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2 +0.2
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 +0.0 +0.0 +0.0 -0.0; largest: +0.2 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.001711 | 0.56% |
| x0 inside sqrt | 0 | -0.0065 | 0.08% |
| x1² inside sqrt | 1 | 1.002 | 0.07% |
| offset | 0 | -0.00252 | 0.06% |
| x0² inside sqrt | 1 | 1.001 | 0.04% |
| x1 inside sqrt | 0 | -0.001612 | 0.02% |
| x0·x1 inside sqrt | 0 | 0.0002484 | 0.01% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.084, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -11 | 3.62 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | -3 | 3.17 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 148 | 0.0476 | 0 | 0.51 | 0.49 | 0.503 | + | hinge |
| 3 | 89 | 3 | 0 | 1.00 | 0.00 | 2.13e-07 | + | constant |
| 4 | -30 | 4.12 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 25 | 3.99 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | -65 | 0.0943 | 0 | 0.52 | 0.48 | 0.141 | + | hinge |
| 7 | -29 | 3.98 | 0 | 1.00 | 0.00 | 7.43e-16 | + | constant |
| 8 | -4 | 3.23 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 1 | 3.39 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | -116 | 0.0505 | 0 | 0.51 | 0.49 | 0.449 | + | hinge |
| 11 | -91 | 3.66 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 156 | 0.00486 | 0 | 0.50 | 0.50 | 0.501 | + | hinge |
| 13 | 64 | 0.102 | 0 | 0.52 | 0.48 | 0.233 | + | hinge |
| 14 | 171 | 3.29 | 0 | 1.00 | 0.00 | 1.97e-05 | + | constant |
| 15 | 84 | 0.00164 | 0 | 0.50 | 0.50 | 0.555 | + | hinge |
| 16 | -73 | 0.00113 | 0 | 0.50 | 0.50 | 0.514 | + | hinge |
| 17 | -152 | 0.0464 | 0 | 0.51 | 0.49 | 0.451 | + | hinge |
| 18 | -17 | 0.00657 | 0 | 0.50 | 0.50 | 0.343 | + | hinge |
| 19 | -80 | -0.000773 | 0 | 0.50 | 0.50 | 0.341 | + | hinge |
| 20 | 13 | 2.12 | 0 | 0.86 | 0.14 | 1.63e-05 | - | constant |
| 21 | -10 | -0.0812 | 0 | 0.49 | 0.51 | 0.178 | + | hinge |
| 22 | 131 | 0.00314 | 0 | 0.50 | 0.50 | 0.577 | + | hinge |
| 23 | 18 | 2.25 | 0 | 0.89 | 0.11 | 2.16e-14 | - | constant |
| 24 | -168 | 0.00603 | 0 | 0.50 | 0.50 | 0.339 | + | hinge |
| 25 | -161 | 0.17 | 0 | 0.53 | 0.47 | 0.0294 | + | constant |
| 26 | -40 | 4.23 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 27 | 38 | 0.0634 | 0 | 0.52 | 0.48 | 0.356 | + | hinge |
| 28 | -34 | 4.19 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 29 | 28 | 0.0979 | 0 | 0.52 | 0.48 | 0.224 | + | hinge |
| 30 | 126 | 4.28 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 31 | 92 | 3.12 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 32 | 179 | 3.06 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 33 | 7 | 3.77 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 34 | -124 | 0.00338 | 0 | 0.50 | 0.50 | 0.455 | + | hinge |
| 35 | 6 | 3.64 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 36 | 19 | 0.0118 | 0 | 0.50 | 0.50 | 0.526 | + | hinge |
| 37 | 46 | 0.00627 | 0 | 0.50 | 0.50 | 0.436 | + | hinge |
| 38 | 74 | -0.00117 | 0 | 0.50 | 0.50 | 0.616 | + | hinge |
| 39 | 176 | 3.25 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 40 | -142 | 0.0836 | 0 | 0.52 | 0.48 | 0.282 | + | hinge |
| 41 | -136 | 6.83 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 42 | -55 | 0.00473 | 0 | 0.50 | 0.50 | 0.178 | + | hinge |
| 43 | 139 | 0.00162 | 0 | 0.50 | 0.50 | 0.596 | + | hinge |
| 44 | 87 | 3.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 45 | -87 | 0.00184 | 0 | 0.50 | 0.50 | 0.494 | + | hinge |
| 46 | -32 | 3.9 | 0 | 1.00 | 0.00 | 1.63e-18 | + | constant |
| 47 | 178 | 5.42 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 48 | 21 | 4.21 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 49 | -147 | 0.0175 | 0 | 0.50 | 0.50 | 0.0798 | + | hinge |
| 50 | -10 | 0.036 | 0 | 0.51 | 0.49 | 0.401 | + | hinge |
| 51 | 4 | 3.45 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 52 | -129 | 0.00619 | 0 | 0.50 | 0.50 | 0.312 | + | hinge |
| 53 | 5 | 0.00699 | 0 | 0.50 | 0.50 | 0.443 | + | hinge |
| 54 | -0 | 3.02 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 55 | 90 | 3.01 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 56 | -0 | 3.07 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 57 | 115 | 0.0388 | 0 | 0.51 | 0.49 | 0.387 | + | hinge |
| 58 | 29 | 4.12 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 59 | -171 | 3.65 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 60 | 178 | 0.00868 | 0 | 0.50 | 0.50 | 0.358 | + | hinge |
| 61 | -32 | 0.126 | 0 | 0.52 | 0.48 | 0.205 | + | hinge |
| 62 | -58 | 0.00279 | 0 | 0.50 | 0.50 | 0.41 | + | hinge |
| 63 | 175 | 3.15 | 0 | 1.00 | 0.00 | 3.14e-06 | + | constant |

## Groups

- negligible: units 0, 1, 3, 4, 5, 7, 8, 9, 11, 14, 20, 23, 26, 28, 30, 31, 32, 33, 35, 39, 41, 44, 46, 47, 48, 51, 54, 55, 56, 58, 59, 63 (magnitude 3.93e-05 together; a quarter of the tolerance is 0.021)
- lines: units 25 (each a constant or a line on the box; together one linear term)
- ring: 3 units (15, 43, 45) with gates at distance 0.00-0.00, switch width ~0, magnitude ~0.56 each, positive
- ring: 5 units (18, 24, 37, 52, 53) with gates at distance 0.01-0.01, switch width ~0, magnitude ~0.34 each, positive
- ring: 4 units (6, 13, 29, 40) with gates at distance 0.08-0.10, switch width ~0, magnitude ~0.23 each, positive
- same line: units 2, 12, 61 (gate angle 148 degrees, offset 0.0476)
- same line: units 10, 34, 38 (gate angle -116 degrees, offset 0.0505)
- same line: units 16, 19, 22, 42, 57, 62 (gate angle -73 degrees, offset 0.00113)
- same line: units 17, 27, 36, 49 (gate angle -152 degrees, offset 0.0464)
- same line: units 21, 50, 60 (gate angle -10 degrees, offset -0.0812)
