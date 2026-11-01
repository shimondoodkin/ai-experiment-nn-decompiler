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
    return 0.2428 * x0 * x0 + x1 * x1 + 0.0008726 * x0 * x1 - 0.2409 * x0 - 0.00174 * x1 < 0.9324
```

Clean:

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

Share of the box where they decide differently: clean and faithful 0.18% (the dirt), faithful and the net 0.33% (what no conic follows), clean and the net 0.37%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0² | 0.25 | 0.2428 | 6.82% |
| x0 | -0.25 | -0.2409 | 2.87% |
| threshold | 0.95 | 0.9324 | 1.85% |
| x0·x1 | 0 | 0.0008726 | 0.82% |
| x1 | 0 | -0.00174 | 0.55% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.867, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 35 | -140 | 92.8 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00691 | - | constant |
| 1 | -148 | 161 | 97.1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00316 | + | constant |
| 2 | 136 | 1.13 | 0.629 | 0.61 | 0.08 | 0.08 | 0.06 | 0.17 | 12.6 | - | none (best 1.49) |
| 3 | 60 | 1.46 | 0.645 | 0.66 | 0.08 | 0.08 | 0.06 | 0.13 | 15.1 | - | none (best 1.79) |
| 4 | 34 | -136 | 90.9 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00553 | - | constant |
| 5 | -144 | 153 | 92.7 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00495 | + | constant |
| 6 | -84 | -0.876 | 0.478 | 0.27 | 0.05 | 0.06 | 0.05 | 0.57 | 15.7 | + | none (best 1.86) |
| 7 | -147 | 117 | 89.7 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0147 | + | constant |
| 8 | 33 | -144 | 91.1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00357 | - | constant |
| 9 | 28 | -1.29 | 0.712 | 0.14 | 0.07 | 0.10 | 0.08 | 0.61 | 12.5 | + | none (best 1.47) |
| 10 | 164 | -2.37 | 0.969 | 0.01 | 0.04 | 0.10 | 0.10 | 0.74 | 12.4 | + | none (best 1.49) |
| 11 | -144 | 154 | 93.1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00518 | + | constant |
| 12 | -147 | 115 | 89.7 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0117 | + | constant |
| 13 | 78 | -0.863 | 0.483 | 0.27 | 0.05 | 0.06 | 0.05 | 0.56 | 15.4 | + | none (best 1.82) |
| 14 | -144 | 148 | 93.1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00646 | + | constant |
| 15 | 109 | -1.23 | 0.586 | 0.18 | 0.06 | 0.08 | 0.06 | 0.61 | 14.9 | + | none (best 1.77) |

## Groups

- negligible: units 0, 1, 4, 5, 7, 8, 11, 12, 14 (magnitude 0.0621 together; a quarter of the tolerance is 0.217)
- the rest: units 2, 3, 6, 9, 10, 13, 15
