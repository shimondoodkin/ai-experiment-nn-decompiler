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
    return x0 * x0 + 0.9996 * x1 * x1 + 0.007839 * x0 * x1 + 0.009444 * x0 + 0.02533 * x1 < 4.029
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.38% (the dirt), faithful and the net 0.35% (what no conic follows), clean and the net 0.47%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x1 | 0 | 0.02533 | 1.90% |
| x0·x1 | 0 | 0.007839 | 1.76% |
| threshold | 4 | 4.029 | 0.72% |
| x0 | 0 | 0.009444 | 0.71% |
| x1² | 1 | 0.9996 | 0.09% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (1.06, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 126 | 1.7 | 0.768 | 0.69 | 0.08 | 0.08 | 0.06 | 0.09 | 14.2 | - | none (best 1.69) |
| 1 | -155 | 1.72 | 0.779 | 0.67 | 0.09 | 0.10 | 0.06 | 0.08 | 14 | - | none (best 1.66) |
| 2 | -120 | 865 | 487 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000608 | + | constant |
| 3 | -134 | 990 | 523 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000343 | + | constant |
| 4 | -126 | 813 | 459 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000663 | + | constant |
| 5 | 57 | -912 | 472 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.000329 | - | constant |
| 6 | -175 | -1.69 | 0.783 | 0.09 | 0.08 | 0.10 | 0.08 | 0.65 | 13.3 | + | none (best 1.58) |
| 7 | 59 | -819 | 471 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.000763 | - | constant |
| 8 | 66 | -1.72 | 0.793 | 0.08 | 0.06 | 0.10 | 0.09 | 0.67 | 14.3 | + | none (best 1.69) |
| 9 | 166 | 1.68 | 0.772 | 0.66 | 0.08 | 0.10 | 0.08 | 0.08 | 13.7 | - | none (best 1.62) |
| 10 | 45 | 1.66 | 0.783 | 0.69 | 0.08 | 0.08 | 0.06 | 0.09 | 14.7 | - | none (best 1.74) |
| 11 | 85 | 1.7 | 0.761 | 0.66 | 0.08 | 0.10 | 0.08 | 0.09 | 13.4 | - | none (best 1.6) |
| 12 | -125 | 788 | 443 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000634 | + | constant |
| 13 | -35 | 1.72 | 0.775 | 0.69 | 0.09 | 0.08 | 0.06 | 0.08 | 14.4 | - | none (best 1.7) |
| 14 | 61 | -925 | 492 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.000394 | - | constant |
| 15 | 104 | -1.69 | 0.759 | 0.08 | 0.08 | 0.10 | 0.08 | 0.66 | 13.4 | + | none (best 1.6) |

## Groups

- negligible: units 2, 3, 4, 5, 7, 12, 14 (magnitude 0.00373 together; a quarter of the tolerance is 0.266)
- ring: 6 units (0, 1, 9, 10, 11, 13) with gates at distance 1.66-1.72, switch width ~0.77, magnitude ~14 each, negative
- the rest: units 6, 8, 15
