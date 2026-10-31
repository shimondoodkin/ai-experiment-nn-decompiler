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
    return 0.998 * x0 * x0 + x1 * x1 - 0.0004206 * x0 * x1 + 0.009071 * x0 + 0.02151 * x1 < 3.994
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.26% (the dirt), faithful and the net 0.28% (what no conic follows), clean and the net 0.34%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x1 | 0 | 0.02151 | 1.61% |
| x0 | 0 | 0.009071 | 0.68% |
| x0² | 1 | 0.998 | 0.45% |
| threshold | 4 | 3.994 | 0.15% |
| x0·x1 | 0 | -0.0004206 | 0.09% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.936, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 88 | 1.8 | 0.877 | 0.66 | 0.09 | 0.11 | 0.09 | 0.05 | 12.4 | - | none (best 1.48) |
| 1 | -150 | 1.8 | 0.849 | 0.68 | 0.10 | 0.09 | 0.06 | 0.07 | 12.3 | - | none (best 1.46) |
| 2 | 70 | 1.67 | 1.5 | 0.53 | 0.17 | 0.19 | 0.09 | 0.02 | 3.1 | - | none (best 0.37) |
| 3 | -25 | 21.6 | 68.6 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0167 | + | constant |
| 4 | -65 | 1.71 | 1.11 | 0.61 | 0.13 | 0.14 | 0.08 | 0.05 | 6.83 | - | none (best 0.814) |
| 5 | -165 | 1.67 | 1.26 | 0.57 | 0.14 | 0.16 | 0.10 | 0.03 | 4.36 | - | none (best 0.518) |
| 6 | 151 | -1.83 | 0.862 | 0.06 | 0.06 | 0.09 | 0.10 | 0.69 | 7.9 | + | none (best 0.94) |
| 7 | 165 | 2.31 | 16.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0302 | + | constant |
| 8 | 48 | 1.75 | 0.81 | 0.70 | 0.09 | 0.09 | 0.05 | 0.08 | 11.9 | - | none (best 1.42) |
| 9 | 163 | 1.76 | 0.866 | 0.66 | 0.09 | 0.11 | 0.08 | 0.06 | 11.7 | - | none (best 1.39) |
| 10 | 10 | 1.78 | 0.864 | 0.66 | 0.09 | 0.11 | 0.09 | 0.05 | 11.8 | - | none (best 1.4) |
| 11 | 129 | 1.75 | 0.866 | 0.69 | 0.09 | 0.09 | 0.06 | 0.07 | 12.3 | - | none (best 1.46) |
| 12 | -25 | 1.76 | 1.1 | 0.62 | 0.13 | 0.13 | 0.07 | 0.05 | 6.63 | - | none (best 0.788) |
| 13 | -82 | 1.82 | 0.849 | 0.66 | 0.09 | 0.11 | 0.09 | 0.05 | 11.2 | - | none (best 1.33) |
| 14 | 170 | 3.39 | 37.9 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.023 | + | constant |
| 15 | 66 | -1.77 | 0.795 | 0.07 | 0.06 | 0.10 | 0.09 | 0.68 | 8.67 | + | none (best 1.03) |

## Groups

- negligible: units 3, 7, 14 (magnitude 0.0698 together; a quarter of the tolerance is 0.234)
- ring: 9 units (0, 1, 4, 8, 9, 10, 11, 12, 13) with gates at distance 1.71-1.82, switch width ~0.87, magnitude ~12 each, negative
- the rest: units 2, 5, 6, 15
