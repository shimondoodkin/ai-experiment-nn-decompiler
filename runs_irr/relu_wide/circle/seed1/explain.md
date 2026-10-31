# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

A conic program passes the check (18 nodes):

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.9995 * x0 * x0 + x1 * x1 + 0.004702 * x0 * x1 + 0.006505 * x0 + 0.01301 * x1 < 3.963
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.31% (the dirt), faithful and the net 0.06% (what no conic follows), clean and the net 0.30%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | 0.004702 | 1.07% |
| x1 | 0 | 0.01301 | 0.99% |
| x0 | 0 | 0.006505 | 0.49% |
| threshold | 3.95 | 3.963 | 0.33% |
| x0² | 1 | 0.9995 | 0.11% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (7.97, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -52 | 0.933 | 0 | 0.69 | 0.31 | 10.2 | - | hinge |
| 1 | 99 | -40.9 | 0 | 0.00 | 1.00 | 1.69 | + | constant |
| 2 | 132 | 0.802 | 0 | 0.67 | 0.33 | 32.6 | - | hinge |
| 3 | 102 | 0.685 | 0 | 0.62 | 0.38 | 17 | - | hinge |
| 4 | -93 | -20.1 | 0 | 0.00 | 1.00 | 3.21 | + | constant |
| 5 | -56 | -34.9 | 0 | 0.00 | 1.00 | 2.42 | + | constant |
| 6 | -52 | -49.9 | 0 | 0.00 | 1.00 | 1.6 | + | constant |
| 7 | 24 | -132 | 0 | 0.00 | 1.00 | 0.363 | + | constant |
| 8 | -96 | -32.5 | 0 | 0.00 | 1.00 | 2.2 | + | constant |
| 9 | 19 | 0.658 | 0 | 0.62 | 0.38 | 16.9 | - | hinge |
| 10 | -124 | 0.721 | 0 | 0.64 | 0.36 | 34.8 | - | hinge |
| 11 | -115 | 0.672 | 0 | 0.62 | 0.38 | 38.3 | - | hinge |
| 12 | -156 | 3.71 | 0 | 1.00 | 0.00 | 0.0013 | - | constant |
| 13 | 49 | 0.719 | 0 | 0.65 | 0.35 | 36 | - | hinge |
| 14 | 133 | 0.601 | 0 | 0.63 | 0.37 | 41.4 | - | hinge |
| 15 | 87 | 0.625 | 0 | 0.61 | 0.39 | 16.9 | - | hinge |
| 16 | -77 | 0.647 | 0 | 0.61 | 0.39 | 19.1 | - | hinge |
| 17 | -157 | 0.591 | 0 | 0.61 | 0.39 | 30.6 | - | hinge |
| 18 | 3 | 0.677 | 0 | 0.61 | 0.39 | 15 | - | hinge |
| 19 | -70 | 0.662 | 0 | 0.62 | 0.38 | 16.9 | - | hinge |
| 20 | -15 | 0.75 | 0 | 0.63 | 0.37 | 9.81 | - | hinge |
| 21 | 3 | -19.3 | 0 | 0.00 | 1.00 | 3.1 | + | constant |
| 22 | 111 | 0.721 | 0 | 0.63 | 0.37 | 24.2 | - | hinge |
| 23 | 174 | -21.9 | 0 | 0.00 | 1.00 | 2.94 | + | constant |
| 24 | 124 | 0.761 | 0 | 0.66 | 0.34 | 34.1 | - | hinge |
| 25 | -166 | 0.658 | 0 | 0.61 | 0.39 | 23.2 | - | hinge |
| 26 | -32 | 0.701 | 0 | 0.64 | 0.36 | 41.3 | - | hinge |
| 27 | 38 | 0.77 | 0 | 0.66 | 0.34 | 27.9 | - | hinge |
| 28 | -92 | 0.571 | 0 | 0.59 | 0.41 | 23.1 | - | hinge |
| 29 | 21 | 0.704 | 0 | 0.63 | 0.37 | 23.2 | - | hinge |
| 30 | 138 | 0.709 | 0 | 0.65 | 0.35 | 10.9 | - | hinge |
| 31 | 67 | 0.607 | 0 | 0.61 | 0.39 | 26.9 | - | hinge |
| 32 | 168 | 0.626 | 0 | 0.61 | 0.39 | 33.9 | - | hinge |
| 33 | 84 | 0.805 | 0 | 0.64 | 0.36 | 6.92 | - | hinge |
| 34 | -139 | 0.701 | 0 | 0.65 | 0.35 | 20.9 | - | hinge |
| 35 | 4 | -31.5 | 0 | 0.00 | 1.00 | 2.01 | + | constant |
| 36 | -45 | 0.681 | 0 | 0.65 | 0.35 | 32 | - | hinge |
| 37 | -18 | -88.4 | 0 | 0.00 | 1.00 | 0.887 | + | constant |
| 38 | 62 | 0.581 | 0 | 0.61 | 0.39 | 35.9 | - | hinge |
| 39 | 156 | 0.726 | 0 | 0.63 | 0.37 | 28.9 | - | hinge |
| 40 | -146 | 0.704 | 0 | 0.64 | 0.36 | 37.2 | - | hinge |
| 41 | -61 | 0.695 | 0 | 0.63 | 0.37 | 38.7 | - | hinge |
| 42 | -55 | 0.621 | 0 | 0.63 | 0.37 | 30.7 | - | hinge |
| 43 | 117 | 0.794 | 0 | 0.65 | 0.35 | 22.1 | - | hinge |
| 44 | 94 | 0.581 | 0 | 0.60 | 0.40 | 15.4 | - | hinge |
| 45 | -76 | -19.7 | 0 | 0.00 | 1.00 | 2.78 | + | constant |
| 46 | 47 | -111 | 0 | 0.00 | 1.00 | 0.757 | + | constant |
| 47 | 33 | 2.44 | 0 | 0.91 | 0.09 | 5.4e-17 | - | constant |
| 48 | 28 | 1.52 | 0 | 0.78 | 0.22 | 0.945 | - | constant |
| 49 | -161 | 0.636 | 0 | 0.61 | 0.39 | 27.9 | - | hinge |
| 50 | -4 | 0.56 | 0 | 0.59 | 0.41 | 22.6 | - | hinge |
| 51 | 31 | -18 | 0 | 0.00 | 1.00 | 1.78 | + | constant |
| 52 | -103 | 0.699 | 0 | 0.62 | 0.38 | 24.5 | - | hinge |
| 53 | 8 | 0.622 | 0 | 0.61 | 0.39 | 21.5 | - | hinge |
| 54 | 32 | 0.742 | 0 | 0.65 | 0.35 | 39.2 | - | hinge |
| 55 | 77 | 0.603 | 0 | 0.60 | 0.40 | 23.9 | - | hinge |
| 56 | -40 | 0.656 | 0 | 0.64 | 0.36 | 33.3 | - | hinge |
| 57 | 96 | -18.7 | 0 | 0.00 | 1.00 | 3.32 | + | constant |
| 58 | 41 | 0.779 | 0 | 0.67 | 0.33 | 33.3 | - | hinge |
| 59 | -131 | 0.641 | 0 | 0.64 | 0.36 | 42 | - | hinge |
| 60 | 148 | 0.695 | 0 | 0.64 | 0.36 | 35.2 | - | hinge |
| 61 | -23 | 0.683 | 0 | 0.62 | 0.38 | 29.6 | - | hinge |
| 62 | -73 | 0.587 | 0 | 0.60 | 0.40 | 19.6 | - | hinge |
| 63 | -174 | 0.73 | 0 | 0.62 | 0.38 | 22.6 | - | hinge |

## Groups

- negligible: units 1, 6, 7, 12, 37, 46, 47, 48, 51 (magnitude 8.03 together; a quarter of the tolerance is 1.99)
- lines: units 4, 5, 8, 21, 23, 35, 45, 57 (each a constant or a line on the box; together one linear term)
- ring: 38 units (3, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 22, 25, 26, 28, 29, 31, 32, 34, 36, 38, 39, 40, 41, 42, 44, 49, 50, 52, 53, 54, 55, 56, 59, 60, 61, 62, 63) with gates at distance 0.56-0.74, switch width ~0, magnitude ~27 each, negative
- same line: units 2, 24, 30, 43 (gate angle 132 degrees, offset 0.802)
- same line: units 27, 58 (gate angle 38 degrees, offset 0.77)
- the rest: units 0, 20, 33
