# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

A conic program passes the check (18 nodes):

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 3.975
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.9918 * x0 * x0 + x1 * x1 + 0.00524 * x0 * x1 - 0.003098 * x0 - 0.03379 * x1 < 3.959
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 3.975
```

Share of the box where they decide differently: clean and faithful 0.34% (the dirt), faithful and the net 0.25% (what no conic follows), clean and the net 0.40%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x1 | 0 | -0.03379 | 2.55% |
| x0² | 1 | 0.9918 | 1.86% |
| x0·x1 | 0 | 0.00524 | 1.18% |
| threshold | 3.975 | 3.959 | 0.40% |
| x0 | 0 | -0.003098 | 0.23% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (3.57, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -52 | 0.327 | 0 | 0.57 | 0.43 | 51.1 | - | hinge |
| 1 | 91 | 0.404 | 0 | 0.57 | 0.43 | 52.3 | - | hinge |
| 2 | 167 | 0.372 | 0 | 0.57 | 0.43 | 54.5 | - | hinge |
| 3 | 48 | 0.44 | 0 | 0.60 | 0.40 | 68.2 | - | hinge |
| 4 | 73 | -10.1 | 0 | 0.00 | 1.00 | 8.93 | + | line |
| 5 | -23 | 0.675 | 0 | 0.62 | 0.38 | 62.7 | - | hinge |
| 6 | 42 | -15.3 | 0 | 0.00 | 1.00 | 6.47 | + | line |
| 7 | -79 | 36.8 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | -19 | -21.6 | 0 | 0.00 | 1.00 | 3.65 | + | line |
| 9 | 27 | 0.438 | 0 | 0.58 | 0.42 | 57.8 | - | hinge |
| 10 | -24 | -29.5 | 0 | 0.00 | 1.00 | 3.15 | + | line |
| 11 | -90 | 0.366 | 0 | 0.56 | 0.44 | 50.9 | - | hinge |
| 12 | -142 | 0.65 | 0 | 0.64 | 0.36 | 72.7 | - | hinge |
| 13 | 38 | -12.7 | 0 | 0.00 | 1.00 | 7.31 | + | line |
| 14 | 140 | 0.642 | 0 | 0.64 | 0.36 | 59.6 | - | hinge |
| 15 | 59 | -10.1 | 0 | 0.00 | 1.00 | 6.03 | + | line |

## Groups

- negligible: units 7 (magnitude 0 together; a quarter of the tolerance is 0.892)
- lines: units 4, 6, 8, 10, 13, 15 (each a constant or a line on the box; together one linear term)
- ring: 6 units (0, 1, 2, 3, 9, 11) with gates at distance 0.33-0.44, switch width ~0, magnitude ~53 each, negative
- ring: 3 units (5, 12, 14) with gates at distance 0.64-0.67, switch width ~0, magnitude ~63 each, negative
