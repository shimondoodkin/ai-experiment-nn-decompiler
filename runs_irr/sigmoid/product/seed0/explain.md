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
    return 0.001452 * x0 * x0 + 0.00461 * x1 * x1 + 0.9893 * x0 * x1 - 0.001284 * x0 + 0.0004097 * x1 - 0.02035
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.6%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.3 +0.3 +0.2 +0.2 +0.2 +0.2 +0.1 +0.1 +0.0 -0.0; largest: +0.3 +0.3 +0.3 +0.3 +0.3 +0.4 +0.4 +0.5 +0.5 +0.6
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.3 +0.4 +0.2 +0.0 -0.1 +0.1 +0.3 +0.3 +0.2 +0.1 -0.1 +0.2; largest: +0.4 +0.5 +0.3 +0.3 +0.6 +0.3 +0.4 +0.5 +0.3 +0.3 +0.6 +0.3
- ripple (faithful minus net): largest 0.8%, at the box edge (x0 = 2, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.2 -0.2 -0.2 -0.2 -0.2 -0.1 -0.1 -0.0 +0.0 -0.0; largest: +0.3 +0.3 +0.3 +0.3 +0.4 +0.4 +0.5 +0.6 +0.6 +0.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.3 -0.3 -0.2 -0.0 +0.1 -0.1 -0.2 -0.3 -0.1 -0.0 +0.1 -0.2; largest: +0.6 +0.7 +0.5 +0.5 +0.7 +0.5 +0.4 +0.6 +0.3 +0.4 +0.8 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9893 | 0.55% |
| constant | 0 | -0.02035 | 0.26% |
| x1² | 0 | 0.00461 | 0.24% |
| x0² | 0 | 0.001452 | 0.08% |
| x0 | 0 | -0.001284 | 0.03% |
| x1 | 0 | 0.0004097 | 0.01% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.156, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 127 | 2.55 | 2.68 | 0.46 | 0.44 | 0.10 | 0.00 | 0.00 | 1.96 | - | none (best 0.194) |
| 1 | -135 | 2.63 | 3.09 | 0.35 | 0.54 | 0.11 | 0.00 | 0.00 | 0.939 | + | none (best 0.0888) |
| 2 | -135 | 2.7 | 2.66 | 0.52 | 0.41 | 0.08 | 0.00 | 0.00 | 0.96 | + | none (best 0.101) |
| 3 | 45 | 2.81 | 3.3 | 0.34 | 0.56 | 0.10 | 0.00 | 0.00 | 1.06 | + | none (best 0.108) |
| 4 | 45 | 2.86 | 3.16 | 0.40 | 0.52 | 0.08 | 0.00 | 0.00 | 0.846 | + | none (best 0.0846) |
| 5 | 132 | -2.71 | 1.81 | 0.00 | 0.00 | 0.04 | 0.19 | 0.77 | 1.15 | + | none (best 0.131) |
| 6 | -135 | 0.983 | 2.11 | 0.18 | 0.39 | 0.37 | 0.07 | 0.00 | 0.979 | + | none (best 0.0981) |
| 7 | -133 | -1.86 | 3.55 | 0.00 | 0.00 | 0.33 | 0.59 | 0.08 | 0.909 | - | none (best 0.0654) |
| 8 | 46 | 2.81 | 3.35 | 0.33 | 0.57 | 0.10 | 0.00 | 0.00 | 0.788 | + | none (best 0.0762) |
| 9 | 137 | 0.742 | 1.86 | 0.18 | 0.33 | 0.36 | 0.12 | 0.00 | 0.635 | - | none (best 0.0765) |
| 10 | 45 | -2.59 | 3.42 | 0.00 | 0.00 | 0.14 | 0.61 | 0.25 | 0.831 | - | none (best 0.0794) |
| 11 | 144 | 2.49 | 2.75 | 0.42 | 0.46 | 0.12 | 0.00 | 0.00 | 1.9 | - | none (best 0.174) |
| 12 | -136 | 2.77 | 2.25 | 0.67 | 0.28 | 0.05 | 0.00 | 0.00 | 1.02 | + | none (best 0.108) |
| 13 | -53 | 1.72 | 3.2 | 0.11 | 0.55 | 0.34 | 0.00 | 0.00 | 1.93 | - | none (best 0.132) |
| 14 | -134 | -2.7 | 3.55 | 0.00 | 0.00 | 0.13 | 0.62 | 0.25 | 0.859 | - | none (best 0.0834) |
| 15 | 143 | -2.33 | 3.02 | 0.00 | 0.00 | 0.17 | 0.54 | 0.29 | 1.38 | + | none (best 0.122) |

## Groups

- ring: 6 units (1, 2, 3, 4, 8, 12) with gates at distance 2.63-2.86, switch width ~3.1, magnitude ~0.95 each, positive
- the rest: units 0, 5, 6, 7, 9, 10, 11, 13, 14, 15
