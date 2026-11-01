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
    return 0.2501 * x0 * x0 + x1 * x1 - 0.003348 * x0 * x1 - 0.2483 * x0 - 0.001016 * x1 < 0.9341
```

Clean:

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

Share of the box where they decide differently: clean and faithful 0.30% (the dirt), faithful and the net 0.11% (what no conic follows), clean and the net 0.31%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | -0.003348 | 3.16% |
| threshold | 0.95 | 0.9341 | 1.67% |
| x0 | -0.25 | -0.2483 | 0.54% |
| x1 | 0 | -0.001016 | 0.32% |
| x0² | 0.25 | 0.2501 | 0.09% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.907, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 84 | 0.918 | 0.505 | 0.57 | 0.05 | 0.06 | 0.05 | 0.26 | 9.68 | - | none (best 1.14) |
| 1 | -88 | 0.894 | 0.481 | 0.57 | 0.05 | 0.06 | 0.05 | 0.27 | 8.91 | - | none (best 1.06) |
| 2 | 144 | -1.88 | 0.956 | 0.05 | 0.06 | 0.10 | 0.10 | 0.69 | 4.24 | + | none (best 0.506) |
| 3 | 152 | -2 | 1.12 | 0.03 | 0.06 | 0.11 | 0.13 | 0.67 | 3.79 | + | none (best 0.452) |
| 4 | 102 | 0.823 | 0.479 | 0.56 | 0.05 | 0.06 | 0.05 | 0.28 | 9.49 | - | none (best 1.13) |
| 5 | -172 | -2.23 | 1.01 | 0.00 | 0.06 | 0.13 | 0.11 | 0.70 | 4 | + | none (best 0.48) |
| 6 | -152 | -1.98 | 1.03 | 0.04 | 0.06 | 0.11 | 0.12 | 0.68 | 3.22 | + | none (best 0.382) |
| 7 | 170 | -0.521 | 2.5 | 0.02 | 0.23 | 0.32 | 0.26 | 0.17 | 0.123 | - | constant |
| 8 | 126 | 0.948 | 0.526 | 0.59 | 0.07 | 0.08 | 0.05 | 0.21 | 8.21 | - | none (best 0.977) |
| 9 | -104 | 0.818 | 0.511 | 0.55 | 0.05 | 0.07 | 0.06 | 0.27 | 8.84 | - | none (best 1.05) |
| 10 | -145 | 0.742 | 1.48 | 0.35 | 0.19 | 0.22 | 0.13 | 0.11 | 2.11 | - | none (best 0.25) |
| 11 | 60 | 1.37 | 0.699 | 0.63 | 0.08 | 0.09 | 0.06 | 0.13 | 9.59 | - | none (best 1.14) |
| 12 | -134 | 1.05 | 0.616 | 0.60 | 0.08 | 0.08 | 0.06 | 0.18 | 9.06 | - | none (best 1.07) |
| 13 | -68 | 1.19 | 0.596 | 0.61 | 0.07 | 0.08 | 0.07 | 0.18 | 9.19 | - | none (best 1.09) |
| 14 | 172 | -1.97 | 2.1 | 0.00 | 0.04 | 0.26 | 0.22 | 0.48 | 0.598 | + | clamp |
| 15 | 170 | 1.26 | 0.661 | 0.60 | 0.07 | 0.08 | 0.07 | 0.18 | 7.57 | - | none (best 0.898) |

## Groups

- negligible: units 7 (magnitude 0.123 together; a quarter of the tolerance is 0.227)
- ring: 5 units (0, 1, 4, 8, 9) with gates at distance 0.82-0.95, switch width ~0.5, magnitude ~8.9 each, negative
- ring: 4 units (11, 12, 13, 15) with gates at distance 1.05-1.37, switch width ~0.64, magnitude ~9.1 each, negative
- same line: units 2, 3 (gate angle 144 degrees, offset -1.88)
- the rest: units 5, 6, 10, 14
