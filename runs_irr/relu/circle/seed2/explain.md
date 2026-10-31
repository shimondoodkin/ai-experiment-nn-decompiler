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
    return 0.9959 * x0 * x0 + x1 * x1 + 0.004278 * x0 * x1 - 0.01156 * x0 - 0.007434 * x1 < 3.999
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.17% (the dirt), faithful and the net 0.19% (what no conic follows), clean and the net 0.22%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | 0.004278 | 0.96% |
| x0² | 1 | 0.9959 | 0.92% |
| x0 | 0 | -0.01156 | 0.87% |
| x1 | 0 | -0.007434 | 0.56% |
| threshold | 4 | 3.999 | 0.02% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (3.28, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -34 | 0.799 | 0 | 0.66 | 0.34 | 60.8 | - | hinge |
| 1 | 30 | 0.81 | 0 | 0.66 | 0.34 | 30.2 | - | hinge |
| 2 | 52 | 0.75 | 0 | 0.66 | 0.34 | 51 | - | hinge |
| 3 | -119 | 0.772 | 0 | 0.65 | 0.35 | 44.5 | - | hinge |
| 4 | -102 | 0.953 | 0 | 0.66 | 0.34 | 5.61 | - | hinge |
| 5 | -61 | 9.71 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 118 | 0.748 | 0 | 0.64 | 0.36 | 42.8 | - | hinge |
| 7 | 77 | 0.967 | 0 | 0.67 | 0.33 | 43.7 | - | hinge |
| 8 | -66 | 0.851 | 0 | 0.65 | 0.35 | 48.7 | - | hinge |
| 9 | 147 | 0.864 | 0 | 0.67 | 0.33 | 58.9 | - | hinge |
| 10 | 175 | 0.742 | 0 | 0.63 | 0.37 | 30.2 | - | hinge |
| 11 | -4 | -43.9 | 0 | 0.00 | 1.00 | 6.47 | + | line |
| 12 | -2 | 0.847 | 0 | 0.64 | 0.36 | 34 | - | hinge |
| 13 | -97 | 0.798 | 0 | 0.63 | 0.37 | 16.8 | - | hinge |
| 14 | -146 | 1 | 0 | 0.70 | 0.30 | 45.4 | - | hinge |
| 15 | -136 | 1.01 | 0 | 0.71 | 0.29 | 13.2 | - | hinge |

## Groups

- negligible: units 5 (magnitude 0 together; a quarter of the tolerance is 0.819)
- lines: units 11 (each a constant or a line on the box; together one linear term)
- ring: 10 units (0, 1, 2, 3, 6, 7, 8, 9, 10, 12) with gates at distance 0.74-0.97, switch width ~0, magnitude ~44 each, negative
- same line: units 4, 13 (gate angle -102 degrees, offset 0.953)
- same line: units 14, 15 (gate angle -146 degrees, offset 1)
