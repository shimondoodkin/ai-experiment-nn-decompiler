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
    return 0.2465 * x0 * x0 + x1 * x1 - 0.000444 * x0 * x1 - 0.243 * x0 + 0.0002973 * x1 < 0.9335
```

Clean:

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

Share of the box where they decide differently: clean and faithful 0.24% (the dirt), faithful and the net 0.11% (what no conic follows), clean and the net 0.23%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0² | 0.25 | 0.2465 | 3.32% |
| x0 | -0.25 | -0.243 | 2.21% |
| threshold | 0.95 | 0.9335 | 1.74% |
| x0·x1 | 0 | -0.000444 | 0.42% |
| x1 | 0 | 0.0002973 | 0.09% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.874, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | -42.7 | 67.4 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0407 | - | constant |
| 1 | 59 | 1.43 | 0.693 | 0.64 | 0.08 | 0.09 | 0.06 | 0.12 | 9.69 | - | none (best 1.15) |
| 2 | 101 | 0.817 | 0.495 | 0.56 | 0.05 | 0.06 | 0.05 | 0.28 | 9.44 | - | none (best 1.12) |
| 3 | 126 | 0.949 | 0.586 | 0.58 | 0.07 | 0.08 | 0.06 | 0.20 | 8.68 | - | none (best 1.03) |
| 4 | 158 | -2.02 | 1.3 | 0.01 | 0.06 | 0.15 | 0.15 | 0.63 | 2.05 | + | none (best 0.244) |
| 5 | -118 | 0.885 | 0.535 | 0.57 | 0.06 | 0.08 | 0.06 | 0.23 | 8.6 | - | none (best 1.02) |
| 6 | -7 | -1.29 | 0.756 | 0.16 | 0.08 | 0.09 | 0.08 | 0.59 | 3.92 | + | none (best 0.464) |
| 7 | 175 | 1.18 | 1.12 | 0.51 | 0.12 | 0.14 | 0.12 | 0.12 | 4.02 | - | none (best 0.478) |
| 8 | 6 | -55.1 | 70.9 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.022 | - | constant |
| 9 | -23 | 2.11 | 0.999 | 0.70 | 0.11 | 0.10 | 0.05 | 0.03 | 9.22 | - | none (best 1.09) |
| 10 | -153 | -2.02 | 1.02 | 0.03 | 0.06 | 0.11 | 0.11 | 0.69 | 3.35 | + | none (best 0.399) |
| 11 | -68 | 1.21 | 0.589 | 0.61 | 0.07 | 0.08 | 0.06 | 0.18 | 8.66 | - | none (best 1.03) |
| 12 | -96 | 0.828 | 0.511 | 0.55 | 0.05 | 0.06 | 0.06 | 0.28 | 9.3 | - | none (best 1.1) |
| 13 | 26 | -1.21 | 0.663 | 0.16 | 0.07 | 0.09 | 0.08 | 0.60 | 4.46 | + | none (best 0.531) |
| 14 | 83 | 0.941 | 0.473 | 0.58 | 0.05 | 0.06 | 0.05 | 0.26 | 9.36 | - | none (best 1.11) |
| 15 | 95 | -0.903 | 0.507 | 0.26 | 0.05 | 0.06 | 0.05 | 0.56 | 5.1 | + | none (best 0.607) |

## Groups

- negligible: units 0, 8 (magnitude 0.0627 together; a quarter of the tolerance is 0.218)
- ring: 5 units (2, 3, 5, 12, 14) with gates at distance 0.82-0.95, switch width ~0.51, magnitude ~9.3 each, negative
- same line: units 4, 9 (gate angle 158 degrees, offset -2.02)
- same line: units 6, 7 (gate angle -7 degrees, offset -1.29)
- the rest: units 1, 10, 11, 13, 15
