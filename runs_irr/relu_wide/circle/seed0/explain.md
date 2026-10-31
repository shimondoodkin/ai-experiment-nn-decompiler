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
    return 0.9957 * x0 * x0 + x1 * x1 + 0.008473 * x0 * x1 - 0.01179 * x0 - 0.002069 * x1 < 3.992
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.14% (the dirt), faithful and the net 0.07% (what no conic follows), clean and the net 0.18%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | 0.008473 | 1.90% |
| x0² | 1 | 0.9957 | 0.97% |
| x0 | -0.02 | -0.01179 | 0.62% |
| threshold | 4 | 3.992 | 0.20% |
| x1 | 0 | -0.002069 | 0.16% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (7.85, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 120 | 0.683 | 0 | 0.63 | 0.37 | 33.2 | - | hinge |
| 1 | -119 | 0.714 | 0 | 0.63 | 0.37 | 34.1 | - | hinge |
| 2 | 147 | 0.698 | 0 | 0.64 | 0.36 | 34.1 | - | hinge |
| 3 | 98 | 0.631 | 0 | 0.61 | 0.39 | 19.6 | - | hinge |
| 4 | 123 | 0.724 | 0 | 0.65 | 0.35 | 35.5 | - | hinge |
| 5 | -173 | 0.632 | 0 | 0.61 | 0.39 | 21.4 | - | hinge |
| 6 | -158 | 1.01 | 0 | 0.68 | 0.32 | 3.93 | - | hinge |
| 7 | 165 | 0.703 | 0 | 0.62 | 0.38 | 24.6 | - | hinge |
| 8 | 59 | 0.736 | 0 | 0.64 | 0.36 | 29.9 | - | hinge |
| 9 | -152 | 0.805 | 0 | 0.65 | 0.35 | 15.9 | - | hinge |
| 10 | 77 | 0.708 | 0 | 0.62 | 0.38 | 17.8 | - | hinge |
| 11 | 124 | -20.6 | 0 | 0.00 | 1.00 | 0.972 | + | constant |
| 12 | 147 | -38.5 | 0 | 0.00 | 1.00 | 1.81 | + | constant |
| 13 | -52 | -11.6 | 0 | 0.00 | 1.00 | 5.09 | + | line |
| 14 | -169 | -16.4 | 0 | 0.00 | 1.00 | 3.43 | + | constant |
| 15 | 113 | -18.4 | 0 | 0.00 | 1.00 | 1.88 | + | constant |
| 16 | -131 | 0.766 | 0 | 0.66 | 0.34 | 35.5 | - | hinge |
| 17 | -93 | 0.729 | 0 | 0.62 | 0.38 | 22.8 | - | hinge |
| 18 | 127 | -15.1 | 0 | 0.00 | 1.00 | 5.28 | + | line |
| 19 | -3 | 0.618 | 0 | 0.60 | 0.40 | 25.1 | - | hinge |
| 20 | -45 | 0.72 | 0 | 0.66 | 0.34 | 40.3 | - | hinge |
| 21 | -68 | 0.747 | 0 | 0.63 | 0.37 | 32 | - | hinge |
| 22 | -140 | 0.718 | 0 | 0.65 | 0.35 | 37.6 | - | hinge |
| 23 | 33 | 0.747 | 0 | 0.65 | 0.35 | 39.2 | - | hinge |
| 24 | 178 | 0.647 | 0 | 0.61 | 0.39 | 19.5 | - | hinge |
| 25 | 73 | 0.961 | 0 | 0.67 | 0.33 | 6.3 | - | hinge |
| 26 | 93 | -19.8 | 0 | 0.00 | 1.00 | 2.85 | + | constant |
| 27 | -36 | 0.715 | 0 | 0.65 | 0.35 | 38.2 | - | hinge |
| 28 | -79 | 0.693 | 0 | 0.62 | 0.38 | 11.6 | - | hinge |
| 29 | -125 | 0.679 | 0 | 0.64 | 0.36 | 28.2 | - | hinge |
| 30 | 65 | 0.717 | 0 | 0.63 | 0.37 | 31.1 | - | hinge |
| 31 | 145 | 0.739 | 0 | 0.65 | 0.35 | 36.5 | - | hinge |
| 32 | -27 | 0.677 | 0 | 0.63 | 0.37 | 28.9 | - | hinge |
| 33 | 41 | 0.7 | 0 | 0.65 | 0.35 | 36.2 | - | hinge |
| 34 | 50 | 0.716 | 0 | 0.65 | 0.35 | 39.6 | - | hinge |
| 35 | -165 | 0.648 | 0 | 0.61 | 0.39 | 29.6 | - | hinge |
| 36 | 84 | 0.69 | 0 | 0.62 | 0.38 | 22.7 | - | hinge |
| 37 | 13 | 0.7 | 0 | 0.62 | 0.38 | 26.3 | - | hinge |
| 38 | -63 | 0.682 | 0 | 0.63 | 0.37 | 32.1 | - | hinge |
| 39 | -139 | -23.8 | 0 | 0.00 | 1.00 | 2.95 | + | constant |
| 40 | 135 | 0.655 | 0 | 0.64 | 0.36 | 42.7 | - | hinge |
| 41 | -50 | 0.789 | 0 | 0.67 | 0.33 | 16.3 | - | hinge |
| 42 | 139 | -38.4 | 0 | 0.00 | 1.00 | 1.99 | + | constant |
| 43 | -86 | 0.617 | 0 | 0.60 | 0.40 | 22.4 | - | hinge |
| 44 | -117 | -38.6 | 0 | 0.00 | 1.00 | 1.9 | + | constant |
| 45 | 39 | 0.893 | 0 | 0.69 | 0.31 | 13.9 | - | hinge |
| 46 | -13 | 0.595 | 0 | 0.60 | 0.40 | 34.4 | - | hinge |
| 47 | -171 | -15.2 | 0 | 0.00 | 1.00 | 4.16 | + | line |
| 48 | 23 | 0.755 | 0 | 0.64 | 0.36 | 30.4 | - | hinge |
| 49 | -135 | 0.768 | 0 | 0.66 | 0.34 | 26.5 | - | hinge |
| 50 | 32 | -11.9 | 0 | 0.00 | 1.00 | 1.78 | + | constant |
| 51 | -151 | 0.693 | 0 | 0.63 | 0.37 | 36.7 | - | hinge |
| 52 | 169 | -12.6 | 0 | 0.00 | 1.00 | 4.43 | + | line |
| 53 | -108 | 0.753 | 0 | 0.63 | 0.37 | 21.6 | - | hinge |
| 54 | 128 | -14.5 | 0 | 0.00 | 1.00 | 4.94 | + | line |
| 55 | 113 | 0.661 | 0 | 0.62 | 0.38 | 20.7 | - | hinge |
| 56 | 132 | -18.1 | 0 | 0.00 | 1.00 | 4.46 | + | line |
| 57 | -52 | 0.742 | 0 | 0.66 | 0.34 | 21.4 | - | hinge |
| 58 | 160 | 0.62 | 0 | 0.61 | 0.39 | 29.3 | - | hinge |
| 59 | 106 | 0.638 | 0 | 0.61 | 0.39 | 30.1 | - | hinge |
| 60 | 0 | 4.19 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 61 | 144 | -18 | 0 | 0.00 | 1.00 | 3.78 | + | constant |
| 62 | -151 | 2.61 | 0 | 0.93 | 0.07 | 0.00538 | - | constant |
| 63 | -110 | -16.4 | 0 | 0.00 | 1.00 | 1.41 | + | constant |

## Groups

- negligible: units 11, 12, 15, 44, 50, 60, 62, 63 (magnitude 9.75 together; a quarter of the tolerance is 1.96)
- lines: units 13, 14, 18, 26, 39, 42, 47, 52, 54, 56, 61 (each a constant or a line on the box; together one linear term)
- ring: 40 units (0, 1, 2, 3, 4, 5, 7, 8, 10, 16, 17, 19, 20, 21, 22, 23, 24, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 43, 46, 48, 49, 51, 53, 55, 57, 58, 59) with gates at distance 0.59-0.79, switch width ~0, magnitude ~30 each, negative
- same line: units 6, 9 (gate angle -158 degrees, offset 1.01)
- the rest: units 25, 28, 45
