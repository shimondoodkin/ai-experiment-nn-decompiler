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
    return 0.9963 * x0 * x0 + x1 * x1 + 0.008071 * x0 * x1 - 0.003996 * x0 - 0.02222 * x1 < 3.991
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.23% (the dirt), faithful and the net 0.14% (what no conic follows), clean and the net 0.24%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | 0.008071 | 1.81% |
| x1 | 0 | -0.02222 | 1.67% |
| x0² | 1 | 0.9963 | 0.83% |
| x0 | 0 | -0.003996 | 0.30% |
| threshold | 4 | 3.991 | 0.22% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (3.37, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -138 | 0.764 | 0 | 0.66 | 0.34 | 45.1 | - | hinge |
| 1 | -106 | 1.08 | 0 | 0.69 | 0.31 | 18.8 | - | hinge |
| 2 | -71 | 0.945 | 0 | 0.67 | 0.33 | 42.1 | - | hinge |
| 3 | 29 | 0.872 | 0 | 0.67 | 0.33 | 47.6 | - | hinge |
| 4 | 67 | 0.923 | 0 | 0.67 | 0.33 | 41.6 | - | hinge |
| 5 | 90 | 3.72 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | -158 | 1.01 | 0 | 0.68 | 0.32 | 29.7 | - | hinge |
| 7 | -176 | 0.771 | 0 | 0.63 | 0.37 | 26.7 | - | hinge |
| 8 | 98 | 0.816 | 0 | 0.64 | 0.36 | 25 | - | hinge |
| 9 | 43 | 0.884 | 0 | 0.69 | 0.31 | 35.3 | - | hinge |
| 10 | -20 | 0.845 | 0 | 0.65 | 0.35 | 50.5 | - | hinge |
| 11 | -50 | 0.712 | 0 | 0.65 | 0.35 | 47.4 | - | hinge |
| 12 | -173 | -33.1 | 0 | 0.00 | 1.00 | 9.02 | + | line |
| 13 | 144 | 0.827 | 0 | 0.67 | 0.33 | 56.3 | - | hinge |
| 14 | -121 | 0.886 | 0 | 0.67 | 0.33 | 38.2 | - | hinge |
| 15 | 125 | 0.933 | 0 | 0.69 | 0.31 | 48.7 | - | hinge |

## Groups

- negligible: units 5 (magnitude 0 together; a quarter of the tolerance is 0.842)
- lines: units 12 (each a constant or a line on the box; together one linear term)
- ring: 12 units (0, 2, 3, 4, 7, 8, 9, 10, 11, 13, 14, 15) with gates at distance 0.71-0.95, switch width ~0, magnitude ~44 each, negative
- the rest: units 1, 6
