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
    return 0.9862 * x0 * x0 + x1 * x1 + 0.01279 * x0 * x1 - 0.001755 * x0 + 0.009979 * x1 < 3.95
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.25% (the dirt), faithful and the net 0.11% (what no conic follows), clean and the net 0.27%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0² | 1 | 0.9862 | 3.12% |
| threshold | 3.977 | 3.95 | 0.68% |
| x0·x1 | 0.01 | 0.01279 | 0.63% |
| x0 | 0 | -0.001755 | 0.13% |
| x1 | 0.01 | 0.009979 | 0.00% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (8.08, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -47 | 7.29 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | 10 | 5 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 26 | 0.711 | 0 | 0.63 | 0.37 | 36.4 | - | hinge |
| 3 | -91 | 0.572 | 0 | 0.59 | 0.41 | 17.7 | - | hinge |
| 4 | -71 | 0.74 | 0 | 0.63 | 0.37 | 31.6 | - | hinge |
| 5 | 27 | 6.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | 73 | -14.1 | 0 | 0.00 | 1.00 | 5.07 | + | line |
| 7 | 65 | -12.2 | 0 | 0.00 | 1.00 | 5.88 | + | line |
| 8 | -61 | 0.845 | 0 | 0.66 | 0.34 | 17.5 | - | hinge |
| 9 | 112 | 0.712 | 0 | 0.63 | 0.37 | 35.2 | - | hinge |
| 10 | 163 | -17.9 | 0 | 0.00 | 1.00 | 2.41 | + | constant |
| 11 | 88 | 0.563 | 0 | 0.60 | 0.40 | 17.1 | - | hinge |
| 12 | 133 | 0.709 | 0 | 0.65 | 0.35 | 46.3 | - | hinge |
| 13 | -123 | 0.71 | 0 | 0.64 | 0.36 | 35.3 | - | hinge |
| 14 | 101 | -38.4 | 0 | 0.00 | 1.00 | 1.78 | + | constant |
| 15 | -143 | 0.714 | 0 | 0.65 | 0.35 | 23.1 | - | hinge |
| 16 | 73 | 0.697 | 0 | 0.62 | 0.38 | 21.4 | - | hinge |
| 17 | -32 | -20.9 | 0 | 0.00 | 1.00 | 2.25 | + | constant |
| 18 | -41 | 0.639 | 0 | 0.64 | 0.36 | 43.3 | - | hinge |
| 19 | -33 | -30.4 | 0 | 0.00 | 1.00 | 2 | + | constant |
| 20 | -50 | -25.3 | 0 | 0.00 | 1.00 | 3.19 | + | constant |
| 21 | 39 | 0.627 | 0 | 0.63 | 0.37 | 38.6 | - | hinge |
| 22 | 69 | 0.734 | 0 | 0.63 | 0.37 | 13.9 | - | hinge |
| 23 | 0 | 0.62 | 0 | 0.60 | 0.40 | 19.9 | - | hinge |
| 24 | 36 | 6.47 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 25 | 43 | 0.654 | 0 | 0.64 | 0.36 | 23.7 | - | hinge |
| 26 | 32 | 0.719 | 0 | 0.64 | 0.36 | 33.2 | - | hinge |
| 27 | 171 | 5.34 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 118 | 0.687 | 0 | 0.63 | 0.37 | 39.6 | - | hinge |
| 29 | 85 | 0.581 | 0 | 0.60 | 0.40 | 16.4 | - | hinge |
| 30 | -26 | -17.2 | 0 | 0.00 | 1.00 | 4.2 | + | line |
| 31 | 64 | -11.1 | 0 | 0.00 | 1.00 | 6.42 | + | line |
| 32 | -17 | 0.571 | 0 | 0.60 | 0.40 | 33.3 | - | hinge |
| 33 | -78 | 0.682 | 0 | 0.62 | 0.38 | 10.3 | - | hinge |
| 34 | 108 | 0.698 | 0 | 0.62 | 0.38 | 21.8 | - | hinge |
| 35 | 159 | 0.673 | 0 | 0.62 | 0.38 | 34.8 | - | hinge |
| 36 | 97 | -22.2 | 0 | 0.00 | 1.00 | 2.93 | + | constant |
| 37 | 50 | -14 | 0 | 0.00 | 1.00 | 4.82 | + | line |
| 38 | -82 | 0.656 | 0 | 0.61 | 0.39 | 30.3 | - | hinge |
| 39 | -116 | 0.726 | 0 | 0.63 | 0.37 | 36.1 | - | hinge |
| 40 | -21 | 0.596 | 0 | 0.61 | 0.39 | 31.4 | - | hinge |
| 41 | 178 | 0.543 | 0 | 0.59 | 0.41 | 28 | - | hinge |
| 42 | -54 | 0.675 | 0 | 0.64 | 0.36 | 39.1 | - | hinge |
| 43 | 84 | 0.617 | 0 | 0.61 | 0.39 | 18.8 | - | hinge |
| 44 | -115 | -56.6 | 0 | 0.00 | 1.00 | 0.75 | + | constant |
| 45 | -97 | 0.674 | 0 | 0.61 | 0.39 | 25.7 | - | hinge |
| 46 | 10 | 0.665 | 0 | 0.61 | 0.39 | 29.6 | - | hinge |
| 47 | 13 | 0.636 | 0 | 0.61 | 0.39 | 19.9 | - | hinge |
| 48 | -139 | 0.668 | 0 | 0.64 | 0.36 | 44.6 | - | hinge |
| 49 | -150 | 0.651 | 0 | 0.62 | 0.38 | 27 | - | hinge |
| 50 | -166 | -60.3 | 0 | 0.00 | 1.00 | 0.534 | + | constant |
| 51 | 166 | 0.555 | 0 | 0.60 | 0.40 | 17.4 | - | hinge |
| 52 | 50 | 0.738 | 0 | 0.66 | 0.34 | 28.9 | - | hinge |
| 53 | 56 | 0.685 | 0 | 0.64 | 0.36 | 31.9 | - | hinge |
| 54 | 125 | -19.2 | 0 | 0.00 | 1.00 | 4.05 | + | line |
| 55 | -34 | 0.717 | 0 | 0.64 | 0.36 | 24.3 | - | hinge |
| 56 | -156 | 0.639 | 0 | 0.62 | 0.38 | 37.8 | - | hinge |
| 57 | -169 | 0.699 | 0 | 0.62 | 0.38 | 30.5 | - | hinge |
| 58 | 92 | 0.688 | 0 | 0.62 | 0.38 | 19.2 | - | hinge |
| 59 | 56 | -17.8 | 0 | 0.00 | 1.00 | 3.12 | + | constant |
| 60 | 38 | -13.1 | 0 | 0.00 | 1.00 | 5.76 | + | line |
| 61 | 150 | 0.599 | 0 | 0.62 | 0.38 | 18.3 | - | hinge |
| 62 | -47 | 0.676 | 0 | 0.65 | 0.35 | 19.2 | - | hinge |
| 63 | 143 | 0.587 | 0 | 0.63 | 0.37 | 43 | - | hinge |

## Groups

- negligible: units 0, 1, 5, 14, 19, 24, 27, 44, 50 (magnitude 5.06 together; a quarter of the tolerance is 2.02)
- lines: units 6, 7, 10, 17, 20, 30, 31, 36, 37, 54, 59, 60 (each a constant or a line on the box; together one linear term)
- ring: 38 units (2, 3, 9, 11, 12, 13, 15, 16, 18, 21, 23, 25, 26, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 51, 53, 55, 56, 57, 58, 61, 62, 63) with gates at distance 0.54-0.73, switch width ~0, magnitude ~30 each, negative
- same line: units 4, 8, 33 (gate angle -71 degrees, offset 0.74)
- the rest: units 22, 52
