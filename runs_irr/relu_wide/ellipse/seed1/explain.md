# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

A conic program passes the check (28 nodes):

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.2526 * x0 * x0 + x1 * x1 + 0.004257 * x0 * x1 - 0.2551 * x0 - 0.005127 * x1 < 0.9414
```

Clean:

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

Share of the box where they decide differently: clean and faithful 0.22% (the dirt), faithful and the net 0.05% (what no conic follows), clean and the net 0.20%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | 0.004257 | 4.02% |
| x0² | 0.25 | 0.2526 | 2.46% |
| x1 | 0 | -0.005127 | 1.62% |
| x0 | -0.25 | -0.2551 | 1.61% |
| threshold | 0.95 | 0.9414 | 0.91% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (12.4, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -81 | 0.639 | 0 | 0.61 | 0.39 | 6.7 | - | hinge |
| 1 | 116 | 0.416 | 0 | 0.58 | 0.42 | 20.4 | - | hinge |
| 2 | 144 | 0.404 | 0 | 0.58 | 0.42 | 32.6 | - | hinge |
| 3 | 92 | 0.476 | 0 | 0.58 | 0.42 | 17.3 | - | hinge |
| 4 | -77 | -9.48 | 0 | 0.00 | 1.00 | 7.35 | + | line |
| 5 | -67 | -11.7 | 0 | 0.00 | 1.00 | 6.48 | + | line |
| 6 | -60 | -16.7 | 0 | 0.00 | 1.00 | 4.53 | + | constant |
| 7 | -141 | 1.4 | 0 | 0.77 | 0.23 | 0.124 | - | constant |
| 8 | -76 | -14.4 | 0 | 0.00 | 1.00 | 5.11 | + | constant |
| 9 | 59 | 0.605 | 0 | 0.62 | 0.38 | 16.7 | - | hinge |
| 10 | -116 | 1.76 | 0 | 0.82 | 0.18 | 0.0397 | - | constant |
| 11 | -94 | 0.23 | 0 | 0.54 | 0.46 | 63.2 | - | hinge |
| 12 | -157 | 3.63 | 0 | 1.00 | 0.00 | 0.00133 | - | constant |
| 13 | 82 | 0.449 | 0 | 0.58 | 0.42 | 40.6 | - | hinge |
| 14 | 118 | 0.333 | 0 | 0.57 | 0.43 | 63.5 | - | hinge |
| 15 | 93 | 0.338 | 0 | 0.56 | 0.44 | 29.3 | - | hinge |
| 16 | -88 | 0.343 | 0 | 0.56 | 0.44 | 27.7 | - | hinge |
| 17 | -149 | 0.292 | 0 | 0.56 | 0.44 | 36.1 | - | hinge |
| 18 | -52 | 0.812 | 0 | 0.67 | 0.33 | 13.4 | - | hinge |
| 19 | -86 | 0.426 | 0 | 0.57 | 0.43 | 21.9 | - | hinge |
| 20 | -63 | 0.905 | 0 | 0.67 | 0.33 | 6.36 | - | hinge |
| 21 | -49 | -13.2 | 0 | 0.00 | 1.00 | 5.57 | + | constant |
| 22 | 147 | 0.458 | 0 | 0.59 | 0.41 | 22.9 | - | hinge |
| 23 | -170 | -18.3 | 0 | 0.00 | 1.00 | 3.19 | + | constant |
| 24 | 94 | 0.306 | 0 | 0.55 | 0.45 | 47.9 | - | hinge |
| 25 | -139 | 0.341 | 0 | 0.57 | 0.43 | 33.1 | - | hinge |
| 26 | -59 | 0.597 | 0 | 0.61 | 0.39 | 62.8 | - | hinge |
| 27 | 40 | 0.938 | 0 | 0.70 | 0.30 | 21.3 | - | hinge |
| 28 | -74 | 0.471 | 0 | 0.58 | 0.42 | 35.6 | - | hinge |
| 29 | 28 | 1.13 | 0 | 0.71 | 0.29 | 16.6 | - | hinge |
| 30 | 129 | 0.396 | 0 | 0.59 | 0.41 | 17.1 | - | hinge |
| 31 | 71 | 0.616 | 0 | 0.61 | 0.39 | 31.1 | - | hinge |
| 32 | 166 | 0.385 | 0 | 0.57 | 0.43 | 52.6 | - | hinge |
| 33 | 103 | 0.367 | 0 | 0.57 | 0.43 | 25.1 | - | hinge |
| 34 | -113 | 0.382 | 0 | 0.57 | 0.43 | 35.2 | - | hinge |
| 35 | -52 | -15.2 | 0 | 0.00 | 1.00 | 5.27 | + | constant |
| 36 | -16 | 1.08 | 0 | 0.69 | 0.31 | 20.4 | - | hinge |
| 37 | -49 | -44.9 | 0 | 0.00 | 1.00 | 1.78 | + | constant |
| 38 | 77 | 0.478 | 0 | 0.58 | 0.42 | 44.1 | - | hinge |
| 39 | 129 | 0.36 | 0 | 0.58 | 0.42 | 37.5 | - | hinge |
| 40 | -158 | 0.44 | 0 | 0.58 | 0.42 | 30.4 | - | hinge |
| 41 | -81 | 0.402 | 0 | 0.57 | 0.43 | 60 | - | hinge |
| 42 | -73 | 0.538 | 0 | 0.59 | 0.41 | 33.7 | - | hinge |
| 43 | 103 | 0.388 | 0 | 0.57 | 0.43 | 32.5 | - | hinge |
| 44 | 90 | 0.454 | 0 | 0.58 | 0.42 | 17.8 | - | hinge |
| 45 | -118 | 0.549 | 0 | 0.60 | 0.40 | 14.6 | - | hinge |
| 46 | -43 | -14.1 | 0 | 0.00 | 1.00 | 5.24 | + | constant |
| 47 | -135 | 0.337 | 0 | 0.58 | 0.42 | 12.2 | - | hinge |
| 48 | -132 | 1.31 | 0 | 0.76 | 0.24 | 3.5e-12 | + | constant |
| 49 | -101 | 0.311 | 0 | 0.55 | 0.45 | 47.3 | - | hinge |
| 50 | 13 | 1.16 | 0 | 0.70 | 0.30 | 19.5 | - | hinge |
| 51 | 117 | -7.85 | 0 | 0.00 | 1.00 | 3 | + | constant |
| 52 | -108 | 0.366 | 0 | 0.56 | 0.44 | 38.3 | - | hinge |
| 53 | 57 | 0.584 | 0 | 0.62 | 0.38 | 30.9 | - | hinge |
| 54 | 62 | 0.567 | 0 | 0.61 | 0.39 | 53.3 | - | hinge |
| 55 | 88 | 0.413 | 0 | 0.57 | 0.43 | 24.7 | - | hinge |
| 56 | -51 | 0.645 | 0 | 0.64 | 0.36 | 32.3 | - | hinge |
| 57 | 66 | -12.9 | 0 | 0.00 | 1.00 | 5.11 | + | constant |
| 58 | 36 | 0.921 | 0 | 0.69 | 0.31 | 27.9 | - | hinge |
| 59 | -119 | 0.348 | 0 | 0.56 | 0.44 | 65.3 | - | hinge |
| 60 | 110 | 0.314 | 0 | 0.56 | 0.44 | 64.1 | - | hinge |
| 61 | -42 | 0.835 | 0 | 0.68 | 0.32 | 27.7 | - | hinge |
| 62 | -92 | 0.326 | 0 | 0.55 | 0.45 | 30 | - | hinge |
| 63 | -131 | 0.347 | 0 | 0.58 | 0.42 | 50.9 | - | hinge |

## Groups

- negligible: units 7, 10, 12, 37, 48, 51 (magnitude 4.95 together; a quarter of the tolerance is 3.11)
- lines: units 4, 5, 6, 8, 21, 23, 35, 46, 57 (each a constant or a line on the box; together one linear term)
- ring: 12 units (1, 2, 3, 13, 19, 22, 28, 30, 38, 40, 44, 55) with gates at distance 0.40-0.48, switch width ~0, magnitude ~24 each, negative
- same line: units 0, 11, 16, 34, 41, 42, 45, 49, 52, 59, 62 (gate angle -81 degrees, offset 0.639)
- same line: units 9, 31, 53, 54 (gate angle 59 degrees, offset 0.605)
- same line: units 14, 15, 24, 33, 43, 60 (gate angle 118 degrees, offset 0.333)
- same line: units 18, 26, 56, 61 (gate angle -52 degrees, offset 0.812)
- same line: units 25, 47, 63 (gate angle -139 degrees, offset 0.341)
- same line: units 27, 29, 58 (gate angle 40 degrees, offset 0.938)
- the rest: units 17, 20, 32, 36, 39, 50
