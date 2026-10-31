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
    return 0.9999 * x0 * x0 + x1 * x1 + 0.008164 * x0 * x1 - 0.01883 * x0 - 0.00167 * x1 < 3.982
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.23% (the dirt), faithful and the net 0.17% (what no conic follows), clean and the net 0.28%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | 0.008164 | 1.83% |
| x0 | 0 | -0.01883 | 1.41% |
| threshold | 4 | 3.982 | 0.45% |
| x1 | 0 | -0.00167 | 0.13% |
| x0² | 1 | 0.9999 | 0.02% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.835, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -33 | 1.78 | 0.792 | 0.70 | 0.09 | 0.08 | 0.06 | 0.08 | 9.35 | - | none (best 1.11) |
| 1 | -135 | -30.7 | 71.2 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.04 | - | constant |
| 2 | 16 | 1.79 | 0.842 | 0.67 | 0.09 | 0.11 | 0.08 | 0.06 | 9.02 | - | none (best 1.07) |
| 3 | -63 | 1.77 | 0.825 | 0.67 | 0.10 | 0.10 | 0.06 | 0.07 | 10.3 | - | none (best 1.22) |
| 4 | -23 | -1.78 | 0.824 | 0.07 | 0.06 | 0.10 | 0.09 | 0.67 | 6.05 | + | none (best 0.72) |
| 5 | 39 | 1.74 | 1.08 | 0.64 | 0.12 | 0.12 | 0.07 | 0.06 | 6.04 | - | none (best 0.72) |
| 6 | 92 | 1.76 | 0.826 | 0.66 | 0.09 | 0.10 | 0.09 | 0.07 | 9.91 | - | none (best 1.18) |
| 7 | 61 | 1.79 | 0.848 | 0.68 | 0.10 | 0.10 | 0.06 | 0.07 | 9.97 | - | none (best 1.19) |
| 8 | -113 | 1.77 | 0.858 | 0.66 | 0.10 | 0.11 | 0.07 | 0.07 | 9.88 | - | none (best 1.18) |
| 9 | 126 | 1.79 | 0.825 | 0.70 | 0.09 | 0.09 | 0.05 | 0.07 | 10.5 | - | none (best 1.24) |
| 10 | -174 | 1.77 | 0.801 | 0.66 | 0.08 | 0.10 | 0.09 | 0.07 | 9.14 | - | none (best 1.09) |
| 11 | 89 | -1.76 | 0.876 | 0.06 | 0.09 | 0.11 | 0.09 | 0.64 | 5.91 | + | none (best 0.706) |
| 12 | 155 | 1.64 | 1.25 | 0.57 | 0.14 | 0.16 | 0.08 | 0.04 | 4.05 | - | none (best 0.482) |
| 13 | -134 | -23.1 | 64.4 | 0.00 | 0.00 | 0.73 | 0.27 | 0.00 | 0.072 | - | constant |
| 14 | -144 | 1.74 | 0.79 | 0.69 | 0.09 | 0.08 | 0.06 | 0.08 | 9.43 | - | none (best 1.12) |
| 15 | 172 | -1.77 | 0.867 | 0.06 | 0.09 | 0.11 | 0.09 | 0.65 | 6.13 | + | none (best 0.73) |

## Groups

- negligible: units 1, 13 (magnitude 0.112 together; a quarter of the tolerance is 0.209)
- ring: 10 units (0, 2, 3, 5, 6, 7, 8, 9, 10, 14) with gates at distance 1.74-1.79, switch width ~0.83, magnitude ~9.7 each, negative
- same line: units 4, 12 (gate angle -23 degrees, offset -1.78)
- the rest: units 11, 15
