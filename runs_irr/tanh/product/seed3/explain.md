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
    return 0.002186 * x0 * x0 + 0.00107 * x1 * x1 + 0.9927 * x0 * x1 - 0.001582 * x0 + 0.001055 * x1 - 0.007168
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.1 +0.1 +0.1 +0.1 +0.1 +0.0 +0.0 -0.0 -0.0 -0.1; largest: +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.3 +0.4 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 +0.2 +0.1 -0.0 -0.2 -0.1 +0.1 +0.2 +0.1 +0.0 -0.1 +0.0; largest: +0.2 +0.3 +0.2 +0.3 +0.5 +0.3 +0.2 +0.3 +0.2 +0.2 +0.4 +0.2
- ripple (faithful minus net): largest 0.4%, inside the box (x0 = 1.63, x1 = -1.75)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.0 +0.0; largest: +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.3 +0.4 +0.4 +0.3
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.1 -0.2 -0.2 -0.1 +0.1 -0.0 -0.1 -0.2 -0.1 -0.0 +0.1 -0.0; largest: +0.2 +0.3 +0.3 +0.2 +0.3 +0.2 +0.3 +0.4 +0.3 +0.3 +0.4 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9927 | 0.37% |
| x0² | 0 | 0.002186 | 0.11% |
| constant | 0 | -0.007168 | 0.09% |
| x1² | 0 | 0.00107 | 0.05% |
| x0 | 0 | -0.001582 | 0.04% |
| x1 | 0 | 0.001055 | 0.03% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.157, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -134 | 2.73 | 2.96 | 0.42 | 0.49 | 0.09 | 0.00 | 0.00 | 1.02 | + | none (best 0.104) |
| 1 | -134 | 2.86 | 2.41 | 0.64 | 0.31 | 0.05 | 0.00 | 0.00 | 1.2 | + | none (best 0.138) |
| 2 | -139 | 1.85 | 4 | 0.03 | 0.59 | 0.39 | 0.00 | 0.00 | 0.437 | + | rounded hinge |
| 3 | 45 | 2.88 | 1.95 | 0.78 | 0.20 | 0.03 | 0.00 | 0.00 | 1.16 | + | none (best 0.133) |
| 4 | 134 | 2.85 | 2.09 | 0.74 | 0.23 | 0.04 | 0.00 | 0.00 | 1.28 | - | none (best 0.136) |
| 5 | 45 | 1.56 | 3.21 | 0.09 | 0.53 | 0.38 | 0.00 | 0.00 | 1.04 | + | none (best 0.0689) |
| 6 | -135 | 2.65 | 3.14 | 0.34 | 0.54 | 0.11 | 0.00 | 0.00 | 1.08 | + | none (best 0.101) |
| 7 | 135 | -2.85 | 2.17 | 0.00 | 0.00 | 0.04 | 0.25 | 0.71 | 1.44 | + | none (best 0.167) |
| 8 | 135 | -1.57 | 3.23 | 0.00 | 0.00 | 0.38 | 0.53 | 0.09 | 2.13 | + | none (best 0.142) |
| 9 | 46 | 1.91 | 3.57 | 0.09 | 0.60 | 0.32 | 0.00 | 0.00 | 1.6 | + | none (best 0.114) |
| 10 | -46 | -0.966 | 2.33 | 0.00 | 0.06 | 0.40 | 0.40 | 0.13 | 0.747 | + | none (best 0.0624) |
| 11 | -46 | 2.77 | 3.28 | 0.33 | 0.56 | 0.10 | 0.00 | 0.00 | 1.1 | - | none (best 0.106) |
| 12 | -135 | -2.69 | 3.28 | 0.00 | 0.00 | 0.11 | 0.57 | 0.32 | 0.929 | - | none (best 0.0902) |
| 13 | 135 | 2.5 | 3.13 | 0.30 | 0.56 | 0.14 | 0.00 | 0.00 | 1.3 | - | none (best 0.117) |
| 14 | -134 | 0.988 | 2.51 | 0.10 | 0.41 | 0.43 | 0.05 | 0.00 | 0.989 | + | none (best 0.0707) |
| 15 | 135 | 2.19 | 3.42 | 0.16 | 0.61 | 0.23 | 0.00 | 0.00 | 1.36 | - | none (best 0.109) |

## Groups

- ring: 4 units (0, 1, 3, 6) with gates at distance 2.65-2.88, switch width ~2.7, magnitude ~1.1 each, positive
- same line: units 7, 11 (gate angle 135 degrees, offset -2.85)
- the rest: units 2, 4, 5, 8, 9, 10, 12, 13, 14, 15
