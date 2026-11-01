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
    return 0.9998 * x0 * x0 + x1 * x1 - 0.00111 * x0 * x1 - 0.01701 * x0 - 0.02395 * x1 < 3.992
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.31% (the dirt), faithful and the net 0.08% (what no conic follows), clean and the net 0.32%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x1 | 0 | -0.02395 | 1.80% |
| x0 | 0 | -0.01701 | 1.28% |
| x0·x1 | 0 | -0.00111 | 0.25% |
| threshold | 4 | 3.992 | 0.20% |
| x0² | 1 | 0.9998 | 0.04% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.988, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -156 | 1.71 | 0.902 | 0.65 | 0.10 | 0.11 | 0.07 | 0.07 | 10.6 | - | none (best 1.26) |
| 1 | 103 | -1.66 | 1.18 | 0.04 | 0.10 | 0.15 | 0.13 | 0.58 | 2.82 | + | none (best 0.335) |
| 2 | -33 | 1.73 | 0.887 | 0.67 | 0.10 | 0.10 | 0.06 | 0.07 | 11.3 | - | none (best 1.35) |
| 3 | -3 | 1.74 | 0.908 | 0.64 | 0.10 | 0.12 | 0.09 | 0.06 | 10.8 | - | none (best 1.28) |
| 4 | 53 | 1.73 | 0.94 | 0.67 | 0.10 | 0.10 | 0.06 | 0.07 | 10.6 | - | none (best 1.26) |
| 5 | 86 | 1.53 | 1.93 | 0.44 | 0.20 | 0.24 | 0.12 | 0.00 | 2.84 | - | none (best 0.264) |
| 6 | 125 | -1.71 | 0.96 | 0.07 | 0.06 | 0.11 | 0.11 | 0.65 | 4.41 | + | none (best 0.524) |
| 7 | 176 | 1.72 | 0.891 | 0.64 | 0.09 | 0.11 | 0.09 | 0.06 | 10.2 | - | none (best 1.22) |
| 8 | 148 | 1.76 | 0.926 | 0.66 | 0.11 | 0.10 | 0.06 | 0.07 | 10.3 | - | none (best 1.22) |
| 9 | 44 | -1.73 | 0.953 | 0.07 | 0.06 | 0.10 | 0.11 | 0.66 | 5.27 | + | none (best 0.625) |
| 10 | -149 | -1.75 | 0.926 | 0.06 | 0.06 | 0.10 | 0.11 | 0.66 | 4.54 | + | none (best 0.541) |
| 11 | -79 | 1.75 | 0.948 | 0.64 | 0.10 | 0.12 | 0.09 | 0.05 | 10.7 | - | none (best 1.27) |
| 12 | -158 | -1.76 | 0.97 | 0.05 | 0.07 | 0.12 | 0.11 | 0.64 | 4.82 | + | none (best 0.572) |
| 13 | 119 | 1.74 | 0.881 | 0.67 | 0.10 | 0.10 | 0.06 | 0.07 | 10.9 | - | none (best 1.3) |
| 14 | -112 | 1.69 | 0.926 | 0.63 | 0.11 | 0.12 | 0.07 | 0.07 | 10.9 | - | none (best 1.3) |
| 15 | 86 | 1.77 | 0.946 | 0.64 | 0.10 | 0.12 | 0.10 | 0.04 | 10.6 | - | none (best 1.27) |

## Groups

- ring: 10 units (0, 2, 3, 4, 7, 8, 11, 13, 14, 15) with gates at distance 1.69-1.77, switch width ~0.92, magnitude ~11 each, negative
- same line: units 10, 12 (gate angle -149 degrees, offset -1.75)
- the rest: units 1, 5, 6, 9
