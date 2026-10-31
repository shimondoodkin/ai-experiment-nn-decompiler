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
    return 0.9973 * x0 * x0 + x1 * x1 + 0.01209 * x0 * x1 - 0.01681 * x0 - 0.007004 * x1 < 3.994
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.21% (the dirt), faithful and the net 0.16% (what no conic follows), clean and the net 0.27%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | 0.01209 | 2.71% |
| x0 | 0 | -0.01681 | 1.26% |
| x0² | 1 | 0.9973 | 0.61% |
| x1 | 0 | -0.007004 | 0.53% |
| threshold | 4 | 3.994 | 0.15% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.986, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -59 | 1.74 | 0.792 | 0.68 | 0.09 | 0.09 | 0.06 | 0.08 | 12.7 | - | none (best 1.51) |
| 1 | -174 | -743 | 380 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.000347 | - | constant |
| 2 | 12 | -1.72 | 0.811 | 0.07 | 0.08 | 0.10 | 0.09 | 0.66 | 13.2 | + | none (best 1.57) |
| 3 | -3 | 672 | 351 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000456 | + | constant |
| 4 | -95 | -1.72 | 0.783 | 0.08 | 0.08 | 0.10 | 0.08 | 0.66 | 12.5 | + | none (best 1.47) |
| 5 | 7 | 553 | 320 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00109 | + | constant |
| 6 | 121 | 1.76 | 0.779 | 0.69 | 0.09 | 0.09 | 0.06 | 0.08 | 12.6 | - | none (best 1.49) |
| 7 | 2 | 684 | 351 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000381 | + | constant |
| 8 | -95 | 1.75 | 0.765 | 0.66 | 0.08 | 0.10 | 0.08 | 0.08 | 12.8 | - | none (best 1.52) |
| 9 | 157 | 1.73 | 0.744 | 0.68 | 0.08 | 0.09 | 0.06 | 0.08 | 11.8 | - | none (best 1.4) |
| 10 | 156 | -1.74 | 0.752 | 0.08 | 0.06 | 0.09 | 0.08 | 0.68 | 12.4 | + | none (best 1.48) |
| 11 | 48 | -1.71 | 0.764 | 0.09 | 0.05 | 0.08 | 0.08 | 0.70 | 12.3 | + | none (best 1.46) |
| 12 | 4 | 658 | 353 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000548 | + | constant |
| 13 | 179 | -604 | 340 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.000827 | - | constant |
| 14 | -130 | -1.75 | 0.766 | 0.08 | 0.05 | 0.08 | 0.08 | 0.70 | 12.6 | + | none (best 1.49) |
| 15 | -166 | -1.75 | 0.785 | 0.07 | 0.08 | 0.10 | 0.08 | 0.67 | 12.5 | + | none (best 1.49) |

## Groups

- negligible: units 1, 3, 5, 7, 12, 13 (magnitude 0.00365 together; a quarter of the tolerance is 0.246)
- ring: 4 units (0, 6, 8, 9) with gates at distance 1.73-1.76, switch width ~0.77, magnitude ~13 each, negative
- the rest: units 2, 4, 10, 11, 14, 15
