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
    return sqrt(1.001 * x0 * x0 + 1.001 * x1 * x1 + 0.0001022 * x0 * x1 - 0.0001464 * x0 + 1.595e-05 * x1 + 0.001276) + 0.0002663
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0, 0), rings 0.423 wide, mean: -0.1 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.1 -0.1; largest: +0.7 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.1 +0.1 +0.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.4 +0.5 +0.3 +0.7 +0.4 +0.3 +0.3 +0.6 +0.3 +0.2 +0.2 +0.5
- ripple (faithful minus net): largest 0.2%, at the box edge (x0 = 1.68, x1 = 2.96)
  - by distance from (7.31e-05, -7.97e-06), rings 0.423 wide, mean: +0.0 +0.0 +0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.1 +0.1 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1 +0.2 +0.1 +0.1 +0.1

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.001276 | 0.44% |
| x0² inside sqrt | 1 | 1.001 | 0.04% |
| x1² inside sqrt | 1 | 1.001 | 0.04% |
| offset | 0 | 0.0002663 | 0.01% |
| x0·x1 inside sqrt | 0 | 0.0001022 | 0.00% |
| x0 inside sqrt | 0 | -0.0001464 | 0.00% |
| x1 inside sqrt | 0 | 1.595e-05 | 0.00% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0841, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -160 | 3.81 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | -105 | 0.0331 | 0 | 0.50 | 0.50 | 0.469 | + | hinge |
| 2 | -84 | 3.51 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 22 | -0.0492 | 0 | 0.49 | 0.51 | 0.358 | + | hinge |
| 4 | 74 | 0.138 | 0 | 0.53 | 0.47 | 0.12 | + | hinge |
| 5 | 115 | 0.00646 | 0 | 0.50 | 0.50 | 0.547 | + | hinge |
| 6 | 179 | 3.03 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 7 | 173 | -0.0594 | 0 | 0.49 | 0.51 | 0.346 | + | hinge |
| 8 | 110 | 3.91 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 9 | 37 | 0.00888 | 0 | 0.50 | 0.50 | 0.284 | + | hinge |
| 10 | -60 | 4.1 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 11 | -39 | 0.0212 | 0 | 0.50 | 0.50 | 0.598 | + | hinge |
| 12 | -150 | 4.1 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 13 | 174 | 2.61 | 0 | 0.94 | 0.06 | 2.57e-05 | - | constant |
| 14 | -137 | 0.00187 | 0 | 0.50 | 0.50 | 0.508 | + | hinge |
| 15 | 121 | 4.46 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 16 | -117 | 0.0013 | 0 | 0.50 | 0.50 | 0.626 | + | hinge |
| 17 | -179 | 3.08 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 18 | -82 | 3.37 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 19 | -157 | 3.93 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 20 | -12 | 3.59 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 21 | 176 | 3.2 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 22 | 0 | 3.04 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 23 | 0 | 3.04 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 24 | 101 | 0.0763 | 0 | 0.52 | 0.48 | 0.246 | + | hinge |
| 25 | 173 | 3.59 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 26 | 104 | 3.3 | 0 | 0.99 | 0.01 | 4.75e-06 | + | constant |
| 27 | 22 | 0.0667 | 0 | 0.51 | 0.49 | 0.301 | + | hinge |
| 28 | 108 | 3.5 | 0 | 1.00 | 0.00 | 1.23e-09 | + | constant |
| 29 | -43 | 4.27 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 30 | 3 | 0.00376 | 0 | 0.50 | 0.50 | 0.478 | + | hinge |
| 31 | -168 | 0.00306 | 0 | 0.50 | 0.50 | 0.589 | + | hinge |
| 32 | -129 | 4.31 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 33 | -71 | 0.00613 | 0 | 0.50 | 0.50 | 0.366 | + | hinge |
| 34 | -11 | 3.07 | 0 | 0.98 | 0.02 | 1.82e-05 | - | constant |
| 35 | 93 | -0.00134 | 0 | 0.50 | 0.50 | 0.343 | + | hinge |
| 36 | 179 | 3.48 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 37 | 13 | 3.67 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 38 | 80 | 0.0113 | 0 | 0.50 | 0.50 | 0.108 | + | hinge |
| 39 | 86 | -0.000924 | 0 | 0.50 | 0.50 | 0.466 | + | hinge |
| 40 | -149 | 0.00549 | 0 | 0.50 | 0.50 | 0.615 | + | hinge |
| 41 | 174 | 0.0736 | 0 | 0.51 | 0.49 | 0.302 | + | hinge |
| 42 | -29 | 0.00434 | 0 | 0.50 | 0.50 | 0.664 | + | hinge |
| 43 | -56 | -0.00795 | 0 | 0.50 | 0.50 | 0.622 | + | hinge |
| 44 | -168 | 3.61 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 45 | 157 | 0.0122 | 0 | 0.50 | 0.50 | 0.213 | + | hinge |
| 46 | 133 | 0.00201 | 0 | 0.50 | 0.50 | 0.608 | + | hinge |
| 47 | 87 | 1.87 | 0 | 0.81 | 0.19 | 0.000122 | - | constant |
| 48 | 141 | 0.186 | 0 | 0.54 | 0.46 | 0.0761 | + | hinge |
| 49 | -121 | 4.11 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 50 | 9 | 3.04 | 0 | 0.99 | 0.01 | 1.02e-11 | + | constant |
| 51 | -17 | 0.003 | 0 | 0.50 | 0.50 | 0.616 | + | hinge |
| 52 | 143 | 4.34 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 53 | -132 | 0.00635 | 0 | 0.50 | 0.50 | 0.274 | + | hinge |
| 54 | -166 | 3.99 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 55 | -42 | 3.63 | 0 | 0.99 | 0.01 | 2.39e-19 | - | constant |
| 56 | -90 | 3.3 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 57 | -111 | 0.0123 | 0 | 0.50 | 0.50 | 0.221 | + | hinge |
| 58 | 54 | 0.00471 | 0 | 0.50 | 0.50 | 0.584 | + | hinge |
| 59 | 101 | -0.0493 | 0 | 0.49 | 0.51 | 0.37 | + | hinge |
| 60 | -56 | 0.212 | 0 | 0.54 | 0.46 | 0.0395 | + | constant |
| 61 | 180 | 3 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 62 | -179 | 2.98 | 0 | 1.00 | 0.00 | 2.97e-07 | + | constant |
| 63 | 26 | 4.77 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 0, 2, 6, 8, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 32, 34, 36, 37, 44, 47, 49, 50, 52, 54, 55, 56, 61, 62, 63 (magnitude 0.000171 together; a quarter of the tolerance is 0.021)
- lines: units 60 (each a constant or a line on the box; together one linear term)
- same line: units 1, 3, 4, 5, 7, 9, 11, 14, 16, 24, 27, 30, 31, 33, 35, 38, 39, 40, 41, 42, 43, 45, 46, 48, 51, 53, 57, 58, 59 (gate angle -105 degrees, offset 0.0331)
