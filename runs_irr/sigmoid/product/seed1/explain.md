# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (7 nodes):

```python
def net(x0, x1):
    return x0 * x1
```

## Faithful fit

The quadratic formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no quadratic formula follows).

Faithful:

```python
def net(x0, x1):
    return -0.005828 * x0 * x0 - 0.004721 * x1 * x1 + 0.989 * x0 * x1 - 0.002116 * x0 - 0.0003557 * x1 + 0.02724
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.8%, at the box edge (x0 = 2, x1 = 1.98)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.3 -0.3 -0.3 -0.3 -0.2 -0.2 -0.1 -0.1 +0.0 +0.1; largest: +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.6 +0.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.1 +0.1 -0.1 -0.3 -0.4 -0.3 -0.1 +0.1 -0.2 -0.3 -0.3 -0.2; largest: +0.4 +0.8 +0.4 +0.4 +0.4 +0.4 +0.4 +0.7 +0.4 +0.4 +0.3 +0.3
- ripple (faithful minus net): largest 0.7%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.3 +0.3 +0.3 +0.3 +0.2 +0.2 +0.1 +0.1 -0.0 -0.1; largest: +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.5 +0.6 +0.6 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 -0.1 +0.1 +0.3 +0.4 +0.3 +0.2 -0.1 +0.2 +0.3 +0.4 +0.3; largest: +0.4 +0.6 +0.4 +0.5 +0.7 +0.6 +0.4 +0.6 +0.4 +0.6 +0.6 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.989 | 0.57% |
| constant | 0 | 0.02724 | 0.35% |
| x0² | 0 | -0.005828 | 0.30% |
| x1² | 0 | -0.004721 | 0.25% |
| x0 | 0 | -0.002116 | 0.05% |
| x1 | 0 | -0.0003557 | 0.01% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.156, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -134 | -2.66 | 2.1 | 0.00 | 0.00 | 0.06 | 0.26 | 0.68 | 1.28 | - | none (best 0.127) |
| 1 | 136 | 2.85 | 2.65 | 0.57 | 0.37 | 0.06 | 0.00 | 0.00 | 0.985 | - | none (best 0.107) |
| 2 | 134 | 2.65 | 3.56 | 0.23 | 0.63 | 0.14 | 0.00 | 0.00 | 1.14 | - | none (best 0.109) |
| 3 | 134 | 2.71 | 3.45 | 0.27 | 0.60 | 0.13 | 0.00 | 0.00 | 0.879 | - | none (best 0.0821) |
| 4 | -46 | -2.28 | 3.87 | 0.00 | 0.00 | 0.25 | 0.66 | 0.10 | 0.627 | + | none (best 0.0513) |
| 5 | -48 | 2.71 | 2.55 | 0.55 | 0.37 | 0.07 | 0.00 | 0.00 | 1.02 | - | none (best 0.0981) |
| 6 | -45 | -2.17 | 3.87 | 0.00 | 0.00 | 0.28 | 0.64 | 0.08 | 0.924 | + | none (best 0.0718) |
| 7 | 46 | 2.61 | 2.27 | 0.61 | 0.32 | 0.07 | 0.00 | 0.00 | 1.45 | + | none (best 0.144) |
| 8 | 45 | -2.62 | 1.63 | 0.00 | 0.00 | 0.04 | 0.17 | 0.79 | 1.82 | - | none (best 0.182) |
| 9 | -45 | 2.65 | 2.82 | 0.44 | 0.46 | 0.10 | 0.00 | 0.00 | 1.07 | - | none (best 0.109) |
| 10 | 135 | -0.809 | 1.85 | 0.00 | 0.11 | 0.35 | 0.34 | 0.20 | 0.652 | + | none (best 0.0787) |
| 11 | -134 | 1.53 | 2.52 | 0.21 | 0.47 | 0.31 | 0.01 | 0.00 | 2.47 | + | none (best 0.16) |
| 12 | -134 | -1.03 | 1.96 | 0.00 | 0.07 | 0.33 | 0.37 | 0.23 | 0.792 | - | none (best 0.086) |
| 13 | 43 | 1.58 | 2.73 | 0.18 | 0.50 | 0.32 | 0.00 | 0.00 | 0.786 | + | none (best 0.054) |
| 14 | 136 | -2.57 | 3.09 | 0.00 | 0.00 | 0.13 | 0.54 | 0.33 | 0.903 | + | none (best 0.0841) |
| 15 | 136 | -2.56 | 3.11 | 0.00 | 0.00 | 0.13 | 0.55 | 0.32 | 0.961 | + | none (best 0.0911) |

## Groups

- ring: 5 units (1, 2, 3, 5, 9) with gates at distance 2.65-2.85, switch width ~2.8, magnitude ~1 each, negative
- same line: units 0, 7 (gate angle -134 degrees, offset -2.66)
- same line: units 4, 6 (gate angle -46 degrees, offset -2.28)
- same line: units 14, 15 (gate angle 136 degrees, offset -2.57)
- the rest: units 8, 10, 11, 12, 13
