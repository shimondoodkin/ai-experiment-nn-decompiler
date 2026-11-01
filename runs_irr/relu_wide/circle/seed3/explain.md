# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

A conic program passes the check (25 nodes):

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 + 0.05 * x0 < 4
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.994 * x0 * x0 + x1 * x1 - 0.005127 * x0 * x1 + 0.04057 * x0 - 0.02438 * x1 < 4.049
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 + 0.05 * x0 < 4
```

Share of the box where they decide differently: clean and faithful 0.52% (the dirt), faithful and the net 0.10% (what no conic follows), clean and the net 0.52%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x1 | 0 | -0.02438 | 1.83% |
| x0² | 1 | 0.994 | 1.35% |
| threshold | 4 | 4.049 | 1.23% |
| x0·x1 | 0 | -0.005127 | 1.15% |
| x0 | 0.05 | 0.04057 | 0.71% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (7.86, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -146 | 0.59 | 0 | 0.62 | 0.38 | 23.8 | - | hinge |
| 1 | -87 | 0.596 | 0 | 0.60 | 0.40 | 20.9 | - | hinge |
| 2 | -95 | -17.3 | 0 | 0.00 | 1.00 | 3.49 | + | constant |
| 3 | 48 | 0.731 | 0 | 0.66 | 0.34 | 33.8 | - | hinge |
| 4 | 78 | 0.489 | 0 | 0.59 | 0.41 | 18.7 | - | hinge |
| 5 | 73 | 0.587 | 0 | 0.60 | 0.40 | 32.2 | - | hinge |
| 6 | -179 | 0.577 | 0 | 0.60 | 0.40 | 29.7 | - | hinge |
| 7 | -179 | -13.8 | 0 | 0.00 | 1.00 | 3.21 | + | constant |
| 8 | 110 | 0.72 | 0 | 0.63 | 0.37 | 15.7 | - | hinge |
| 9 | 34 | -18.6 | 0 | 0.00 | 1.00 | 3.07 | + | constant |
| 10 | -46 | 0.608 | 0 | 0.63 | 0.37 | 46.6 | - | hinge |
| 11 | -53 | 0.687 | 0 | 0.64 | 0.36 | 16.2 | - | hinge |
| 12 | -131 | 0.738 | 0 | 0.66 | 0.34 | 35.8 | - | hinge |
| 13 | 143 | 0.647 | 0 | 0.64 | 0.36 | 39 | - | hinge |
| 14 | -149 | 0.718 | 0 | 0.64 | 0.36 | 8.58 | - | hinge |
| 15 | 106 | 0.692 | 0 | 0.62 | 0.38 | 15.3 | - | hinge |
| 16 | -124 | 0.764 | 0 | 0.65 | 0.35 | 23.5 | - | hinge |
| 17 | -166 | 0.627 | 0 | 0.61 | 0.39 | 20.5 | - | hinge |
| 18 | -84 | 0.572 | 0 | 0.59 | 0.41 | 14.1 | - | hinge |
| 19 | 151 | 0.63 | 0 | 0.62 | 0.38 | 59 | - | hinge |
| 20 | -123 | -36 | 0 | 0.00 | 1.00 | 2.47 | + | constant |
| 21 | -104 | -22 | 0 | 0.00 | 1.00 | 3.3 | + | constant |
| 22 | -15 | 0.58 | 0 | 0.60 | 0.40 | 37.5 | - | hinge |
| 23 | 75 | -6.86 | 0 | 0.00 | 1.00 | 0.000252 | + | constant |
| 24 | 94 | 0.599 | 0 | 0.60 | 0.40 | 21.5 | - | hinge |
| 25 | -24 | -18.1 | 0 | 0.00 | 1.00 | 0.205 | + | constant |
| 26 | 103 | -34.7 | 0 | 0.00 | 1.00 | 1.88 | + | constant |
| 27 | 21 | 0.671 | 0 | 0.62 | 0.38 | 46.4 | - | hinge |
| 28 | 97 | 0.731 | 0 | 0.62 | 0.38 | 7.2 | - | hinge |
| 29 | -40 | 0.543 | 0 | 0.62 | 0.38 | 23.4 | - | hinge |
| 30 | 9 | 0.524 | 0 | 0.59 | 0.41 | 28.6 | - | hinge |
| 31 | -155 | 0.521 | 0 | 0.60 | 0.40 | 25.4 | - | hinge |
| 32 | -119 | 0.846 | 0 | 0.66 | 0.34 | 33.1 | - | hinge |
| 33 | -87 | 0.545 | 0 | 0.59 | 0.41 | 20.5 | - | hinge |
| 34 | 40 | 0.627 | 0 | 0.63 | 0.37 | 25.5 | - | hinge |
| 35 | 80 | 0.578 | 0 | 0.60 | 0.40 | 14.3 | - | hinge |
| 36 | 162 | 3.24 | 0 | 0.99 | 0.01 | 0.00367 | - | constant |
| 37 | 31 | 0.662 | 0 | 0.63 | 0.37 | 52.4 | - | hinge |
| 38 | 80 | 0.576 | 0 | 0.60 | 0.40 | 16.2 | - | hinge |
| 39 | 60 | 0.794 | 0 | 0.65 | 0.35 | 24.9 | - | hinge |
| 40 | -154 | 0.643 | 0 | 0.62 | 0.38 | 9.44 | - | hinge |
| 41 | 162 | 0.61 | 0 | 0.61 | 0.39 | 23.7 | - | hinge |
| 42 | -36 | 0.582 | 0 | 0.62 | 0.38 | 27 | - | hinge |
| 43 | -61 | 1.13 | 0 | 0.72 | 0.28 | 3.76 | - | constant |
| 44 | -149 | 0.585 | 0 | 0.61 | 0.39 | 40.7 | - | hinge |
| 45 | -138 | -84.1 | 0 | 0.00 | 1.00 | 0.543 | + | constant |
| 46 | 115 | 0.696 | 0 | 0.63 | 0.37 | 32.3 | - | hinge |
| 47 | 162 | -94.2 | 0 | 0.00 | 1.00 | 0.982 | + | constant |
| 48 | 137 | 0.641 | 0 | 0.64 | 0.36 | 56.5 | - | hinge |
| 49 | -106 | -21.3 | 0 | 0.00 | 1.00 | 3.68 | + | constant |
| 50 | 79 | 0.662 | 0 | 0.61 | 0.39 | 5.51 | - | hinge |
| 51 | -59 | 0.638 | 0 | 0.62 | 0.38 | 49.2 | - | hinge |
| 52 | -92 | -23.1 | 0 | 0.00 | 1.00 | 2.88 | + | constant |
| 53 | -111 | -34.6 | 0 | 0.00 | 1.00 | 2.05 | + | constant |
| 54 | -60 | -19.5 | 0 | 0.00 | 1.00 | 0.551 | + | constant |
| 55 | 27 | -52.8 | 0 | 0.00 | 1.00 | 1.67 | + | constant |
| 56 | -106 | 0.677 | 0 | 0.62 | 0.38 | 16.9 | - | hinge |
| 57 | -99 | 0.706 | 0 | 0.62 | 0.38 | 17 | - | hinge |
| 58 | 52 | 1.09 | 0 | 0.72 | 0.28 | 4.81 | - | hinge |
| 59 | 110 | -18.7 | 0 | 0.00 | 1.00 | 4.3 | + | line |
| 60 | -96 | 0.672 | 0 | 0.61 | 0.39 | 22.1 | - | hinge |
| 61 | -163 | 0.55 | 0 | 0.60 | 0.40 | 17 | - | hinge |
| 62 | -163 | -20.4 | 0 | 0.00 | 1.00 | 2.53 | + | constant |
| 63 | -25 | 0.509 | 0 | 0.59 | 0.41 | 52.5 | - | hinge |

## Groups

- negligible: units 23, 25, 26, 36, 45, 47, 54, 55 (magnitude 5.83 together; a quarter of the tolerance is 1.96)
- lines: units 2, 7, 9, 20, 21, 43, 49, 52, 53, 59, 62 (each a constant or a line on the box; together one linear term)
- ring: 23 units (0, 1, 4, 5, 6, 10, 13, 17, 18, 22, 24, 29, 30, 31, 33, 34, 35, 38, 41, 42, 44, 51, 61) with gates at distance 0.49-0.65, switch width ~0, magnitude ~24 each, negative
- ring: 12 units (3, 8, 11, 12, 15, 16, 32, 39, 46, 56, 57, 60) with gates at distance 0.67-0.85, switch width ~0, magnitude ~23 each, negative
- same line: units 14, 40 (gate angle -149 degrees, offset 0.718)
- same line: units 27, 37 (gate angle 21 degrees, offset 0.671)
- the rest: units 19, 28, 48, 50, 58, 63
