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
    return 0.006028 * x0 * x0 + 0.007586 * x1 * x1 + 0.9861 * x0 * x1 - 0.003025 * x0 + 0.002586 * x1 - 0.04155
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.5 +0.5 +0.5 +0.5 +0.4 +0.3 +0.2 +0.1 +0.0 -0.1; largest: +0.5 +0.5 +0.5 +0.6 +0.6 +0.6 +0.6 +0.6 +0.7 +1.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.5 +0.5 +0.4 +0.2 -0.1 +0.2 +0.4 +0.5 +0.5 +0.3 +0.1 +0.3; largest: +0.6 +0.6 +0.5 +0.5 +1.0 +0.5 +0.5 +0.5 +0.5 +0.5 +0.7 +0.5
- ripple (faithful minus net): largest 0.9%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.5 -0.5 -0.5 -0.4 -0.4 -0.3 -0.2 -0.1 -0.0 +0.0; largest: +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 +0.7 +0.8 +0.9
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.5 -0.5 -0.4 -0.2 +0.1 -0.2 -0.4 -0.5 -0.5 -0.3 +0.0 -0.3; largest: +0.6 +0.7 +0.6 +0.7 +0.9 +0.7 +0.6 +0.8 +0.6 +0.6 +0.7 +0.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9861 | 0.72% |
| constant | 0 | -0.04155 | 0.54% |
| x1² | 0 | 0.007586 | 0.40% |
| x0² | 0 | 0.006028 | 0.32% |
| x0 | 0 | -0.003025 | 0.08% |
| x1 | 0 | 0.002586 | 0.07% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.155, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -48 | 2.74 | 1.79 | 0.78 | 0.19 | 0.04 | 0.00 | 0.00 | 0.739 | - | none (best 0.0791) |
| 1 | 46 | -2.64 | 1.69 | 0.00 | 0.00 | 0.04 | 0.18 | 0.78 | 1.03 | - | none (best 0.108) |
| 2 | 46 | 0.886 | 1.77 | 0.24 | 0.34 | 0.32 | 0.10 | 0.00 | 0.619 | + | none (best 0.0748) |
| 3 | -134 | 2.56 | 2.41 | 0.55 | 0.36 | 0.08 | 0.00 | 0.00 | 1.45 | + | none (best 0.138) |
| 4 | -47 | -2.13 | 3.12 | 0.00 | 0.00 | 0.22 | 0.57 | 0.21 | 1.37 | + | none (best 0.109) |
| 5 | 45 | 2.49 | 2.7 | 0.43 | 0.46 | 0.11 | 0.00 | 0.00 | 0.804 | + | none (best 0.0713) |
| 6 | 134 | 2.08 | 3.1 | 0.21 | 0.57 | 0.23 | 0.00 | 0.00 | 1.57 | - | none (best 0.116) |
| 7 | 134 | 2.74 | 2.28 | 0.65 | 0.29 | 0.05 | 0.00 | 0.00 | 1.32 | - | none (best 0.146) |
| 8 | -47 | 2.47 | 2.28 | 0.56 | 0.34 | 0.09 | 0.00 | 0.00 | 1.41 | - | none (best 0.129) |
| 9 | 49 | 2.71 | 2.2 | 0.66 | 0.28 | 0.05 | 0.00 | 0.00 | 0.909 | + | none (best 0.0961) |
| 10 | -132 | -2.64 | 2.42 | 0.00 | 0.00 | 0.07 | 0.35 | 0.57 | 0.668 | - | none (best 0.0667) |
| 11 | 135 | -2.16 | 2.3 | 0.00 | 0.00 | 0.15 | 0.40 | 0.45 | 1.33 | + | none (best 0.0969) |
| 12 | 133 | -0.884 | 1.72 | 0.00 | 0.10 | 0.31 | 0.33 | 0.25 | 0.868 | + | none (best 0.105) |
| 13 | -137 | -2.21 | 2.87 | 0.00 | 0.00 | 0.18 | 0.53 | 0.30 | 0.876 | - | none (best 0.0704) |
| 14 | -135 | -2.46 | 2.72 | 0.00 | 0.00 | 0.12 | 0.47 | 0.41 | 0.427 | - | none (best 0.039) |
| 15 | -133 | 1.39 | 2.53 | 0.18 | 0.46 | 0.34 | 0.02 | 0.00 | 1.82 | + | none (best 0.114) |

## Groups

- ring: 3 units (3, 5, 9) with gates at distance 2.49-2.71, switch width ~2.4, magnitude ~0.91 each, positive
- same line: units 4, 6 (gate angle -47 degrees, offset -2.13)
- same line: units 10, 14 (gate angle -132 degrees, offset -2.64)
- the rest: units 0, 1, 2, 7, 8, 11, 12, 13, 15
