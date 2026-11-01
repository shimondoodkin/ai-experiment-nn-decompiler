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
    return 0.9972 * x0 * x0 + x1 * x1 + 0.004528 * x0 * x1 - 0.01903 * x0 - 0.02855 * x1 < 3.983
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.34% (the dirt), faithful and the net 0.13% (what no conic follows), clean and the net 0.34%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x1 | 0 | -0.02855 | 2.14% |
| x0 | 0 | -0.01903 | 1.43% |
| x0·x1 | 0 | 0.004528 | 1.01% |
| x0² | 1 | 0.9972 | 0.63% |
| threshold | 4 | 3.983 | 0.42% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.905, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 178 | 1.68 | 0.753 | 0.66 | 0.08 | 0.09 | 0.08 | 0.09 | 8.58 | - | none (best 1.02) |
| 1 | -157 | 1.67 | 0.753 | 0.67 | 0.09 | 0.10 | 0.06 | 0.09 | 7.67 | - | none (best 0.915) |
| 2 | -48 | 1.69 | 0.762 | 0.69 | 0.08 | 0.08 | 0.05 | 0.09 | 9.06 | - | none (best 1.08) |
| 3 | 9 | 1.71 | 0.774 | 0.66 | 0.08 | 0.10 | 0.08 | 0.08 | 9.48 | - | none (best 1.13) |
| 4 | 67 | 1.7 | 0.766 | 0.67 | 0.09 | 0.10 | 0.06 | 0.08 | 9.1 | - | none (best 1.08) |
| 5 | 14 | 664 | 325 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000496 | + | constant |
| 6 | 31 | 620 | 296 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000546 | + | constant |
| 7 | 161 | -1.69 | 0.769 | 0.08 | 0.07 | 0.10 | 0.09 | 0.66 | 9.19 | + | none (best 1.09) |
| 8 | 151 | 1.7 | 0.753 | 0.68 | 0.09 | 0.09 | 0.06 | 0.09 | 8.78 | - | none (best 1.04) |
| 9 | 48 | -1.67 | 0.766 | 0.09 | 0.06 | 0.08 | 0.08 | 0.69 | 8.66 | + | none (best 1.03) |
| 10 | -57 | -1.7 | 0.751 | 0.09 | 0.05 | 0.08 | 0.08 | 0.69 | 8.87 | + | none (best 1.05) |
| 11 | -75 | 1.69 | 0.769 | 0.66 | 0.08 | 0.10 | 0.08 | 0.08 | 9.05 | - | none (best 1.07) |
| 12 | -142 | -1.71 | 0.773 | 0.08 | 0.06 | 0.08 | 0.08 | 0.69 | 9.41 | + | none (best 1.12) |
| 13 | 31 | 583 | 278 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000568 | + | constant |
| 14 | -105 | 1.66 | 0.768 | 0.65 | 0.08 | 0.10 | 0.08 | 0.09 | 9.37 | - | none (best 1.11) |
| 15 | 96 | 1.68 | 0.768 | 0.65 | 0.08 | 0.10 | 0.08 | 0.09 | 9.18 | - | none (best 1.09) |

## Groups

- negligible: units 5, 6, 13 (magnitude 0.00161 together; a quarter of the tolerance is 0.226)
- ring: 9 units (0, 1, 2, 3, 4, 8, 11, 14, 15) with gates at distance 1.66-1.71, switch width ~0.77, magnitude ~9.1 each, negative
- the rest: units 7, 9, 10, 12
