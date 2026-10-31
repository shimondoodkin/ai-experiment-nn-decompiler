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
    return 0.9841 * x0 * x0 + x1 * x1 + 0.009324 * x0 * x1 - 0.006724 * x0 - 0.00427 * x1 < 3.93
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.30% (the dirt), faithful and the net 0.18% (what no conic follows), clean and the net 0.33%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0² | 1 | 0.9841 | 3.58% |
| x0·x1 | 0 | 0.009324 | 2.09% |
| threshold | 4 | 3.93 | 1.75% |
| x0 | 0 | -0.006724 | 0.50% |
| x1 | 0 | -0.00427 | 0.32% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.906, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -15 | -456 | 228 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.000643 | - | constant |
| 1 | 166 | 489 | 228 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000344 | + | constant |
| 2 | 175 | 1.74 | 0.788 | 0.66 | 0.08 | 0.10 | 0.08 | 0.08 | 10.4 | - | none (best 1.24) |
| 3 | 47 | 1.73 | 0.78 | 0.70 | 0.08 | 0.08 | 0.05 | 0.08 | 11.7 | - | none (best 1.39) |
| 4 | -35 | -1.74 | 0.771 | 0.08 | 0.05 | 0.08 | 0.08 | 0.70 | 10 | + | none (best 1.18) |
| 5 | -21 | 1.71 | 0.814 | 0.66 | 0.09 | 0.10 | 0.07 | 0.07 | 12.1 | - | none (best 1.43) |
| 6 | -65 | -1.72 | 0.803 | 0.08 | 0.06 | 0.10 | 0.09 | 0.67 | 11.5 | + | none (best 1.36) |
| 7 | 164 | 412 | 230 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00149 | + | constant |
| 8 | -15 | -478 | 229 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.0004 | - | constant |
| 9 | 14 | 1.75 | 0.793 | 0.67 | 0.08 | 0.10 | 0.08 | 0.07 | 11.3 | - | none (best 1.34) |
| 10 | -100 | -1.71 | 0.788 | 0.08 | 0.08 | 0.10 | 0.08 | 0.66 | 11.4 | + | none (best 1.35) |
| 11 | 165 | 451 | 228 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000815 | + | constant |
| 12 | -155 | 1.74 | 0.797 | 0.67 | 0.09 | 0.10 | 0.06 | 0.08 | 10.9 | - | none (best 1.3) |
| 13 | 59 | -1.69 | 0.796 | 0.08 | 0.06 | 0.09 | 0.09 | 0.67 | 11.5 | + | none (best 1.36) |
| 14 | 125 | -1.73 | 0.781 | 0.08 | 0.06 | 0.08 | 0.09 | 0.69 | 11.1 | + | none (best 1.32) |
| 15 | 91 | -1.73 | 0.787 | 0.08 | 0.08 | 0.10 | 0.08 | 0.66 | 11.4 | + | none (best 1.36) |

## Groups

- negligible: units 0, 1, 7, 8, 11 (magnitude 0.00369 together; a quarter of the tolerance is 0.227)
- ring: 5 units (2, 3, 5, 9, 12) with gates at distance 1.71-1.75, switch width ~0.79, magnitude ~11 each, negative
- the rest: units 4, 6, 10, 13, 14, 15
