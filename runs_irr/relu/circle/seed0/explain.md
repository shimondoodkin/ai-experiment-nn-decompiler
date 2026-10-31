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
    return 0.9937 * x0 * x0 + x1 * x1 + 0.002349 * x0 * x1 + 0.009054 * x0 - 0.001911 * x1 < 4.043
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.49% (the dirt), faithful and the net 0.25% (what no conic follows), clean and the net 0.49%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0² | 1 | 0.9937 | 1.42% |
| threshold | 4 | 4.043 | 1.08% |
| x0 | 0 | 0.009054 | 0.68% |
| x0·x1 | 0 | 0.002349 | 0.53% |
| x1 | 0 | -0.001911 | 0.14% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (3.6, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 75 | 0.617 | 0 | 0.61 | 0.39 | 52.5 | - | hinge |
| 1 | -90 | 0.58 | 0 | 0.59 | 0.41 | 41.5 | - | hinge |
| 2 | 169 | 6.04 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 89 | 2.88 | 0 | 0.98 | 0.02 | 0.000851 | + | constant |
| 4 | 59 | 5.66 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 136 | -22 | 0 | 0.00 | 1.00 | 11.8 | + | line |
| 6 | 152 | -11.2 | 0 | 0.00 | 1.00 | 19.1 | + | line |
| 7 | 174 | 0.438 | 0 | 0.57 | 0.43 | 37.4 | - | hinge |
| 8 | 6 | 0.551 | 0 | 0.59 | 0.41 | 47.7 | - | hinge |
| 9 | -160 | 0.686 | 0 | 0.62 | 0.38 | 54.7 | - | hinge |
| 10 | 37 | 0.569 | 0 | 0.62 | 0.38 | 51.5 | - | hinge |
| 11 | 138 | 0.453 | 0 | 0.60 | 0.40 | 64.3 | - | hinge |
| 12 | 165 | 10.2 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | -50 | 0.537 | 0 | 0.62 | 0.38 | 64.6 | - | hinge |
| 14 | -122 | 0.613 | 0 | 0.62 | 0.38 | 41.6 | - | hinge |
| 15 | 111 | 0.779 | 0 | 0.64 | 0.36 | 45.6 | - | hinge |

## Groups

- negligible: units 2, 3, 4, 12 (magnitude 0.000851 together; a quarter of the tolerance is 0.899)
- lines: units 5, 6 (each a constant or a line on the box; together one linear term)
- ring: 7 units (0, 1, 8, 9, 10, 13, 14) with gates at distance 0.54-0.69, switch width ~0, magnitude ~52 each, negative
- the rest: units 7, 11, 15
