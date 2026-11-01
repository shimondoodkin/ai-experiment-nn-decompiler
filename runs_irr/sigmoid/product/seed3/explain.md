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
    return -0.004527 * x0 * x0 - 0.004511 * x1 * x1 + 0.9896 * x0 * x1 + 0.001169 * x0 + 0.001243 * x1 + 0.02897
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, at the box edge (x0 = -1.99, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.4 -0.4 -0.3 -0.3 -0.3 -0.2 -0.2 -0.1 -0.0 +0.0; largest: +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.5 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.0 -0.2 -0.3 -0.4 -0.3 -0.1 +0.1 -0.1 -0.3 -0.4 -0.3; largest: +0.4 +0.6 +0.4 +0.4 +0.4 +0.4 +0.4 +0.7 +0.4 +0.4 +0.4 +0.4
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = -1.29, x1 = -2)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.3 +0.3 +0.3 +0.3 +0.3 +0.2 +0.2 +0.1 +0.0 +0.0; largest: +0.4 +0.4 +0.4 +0.4 +0.5 +0.5 +0.4 +0.5 +0.6 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.2 -0.0 +0.1 +0.3 +0.4 +0.3 +0.1 -0.1 +0.1 +0.3 +0.4 +0.3; largest: +0.4 +0.5 +0.4 +0.5 +0.6 +0.5 +0.5 +0.6 +0.5 +0.5 +0.6 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9896 | 0.53% |
| constant | 0 | 0.02897 | 0.37% |
| x0² | 0 | -0.004527 | 0.23% |
| x1² | 0 | -0.004511 | 0.23% |
| x1 | 0 | 0.001243 | 0.03% |
| x0 | 0 | 0.001169 | 0.03% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.156, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -135 | 2.65 | 3.5 | 0.25 | 0.62 | 0.14 | 0.00 | 0.00 | 0.987 | + | none (best 0.0954) |
| 1 | -137 | 2.55 | 3.55 | 0.21 | 0.63 | 0.16 | 0.00 | 0.00 | 1.23 | + | none (best 0.112) |
| 2 | -135 | 2.9 | 2.99 | 0.47 | 0.47 | 0.07 | 0.00 | 0.00 | 1.01 | + | none (best 0.102) |
| 3 | -46 | 2.74 | 1.82 | 0.77 | 0.19 | 0.04 | 0.00 | 0.00 | 1.34 | - | none (best 0.149) |
| 4 | 133 | -1.12 | 1.78 | 0.00 | 0.07 | 0.28 | 0.35 | 0.29 | 0.627 | + | none (best 0.0694) |
| 5 | 46 | 2.64 | 1.92 | 0.72 | 0.23 | 0.05 | 0.00 | 0.00 | 1.77 | + | none (best 0.188) |
| 6 | -135 | 2.52 | 3.58 | 0.19 | 0.64 | 0.17 | 0.00 | 0.00 | 1.14 | + | none (best 0.106) |
| 7 | 134 | 0.906 | 1.76 | 0.24 | 0.34 | 0.32 | 0.10 | 0.00 | 0.546 | - | none (best 0.0659) |
| 8 | 132 | 2.54 | 3.06 | 0.33 | 0.54 | 0.13 | 0.00 | 0.00 | 1.09 | - | none (best 0.101) |
| 9 | 45 | 2.34 | 2.84 | 0.34 | 0.51 | 0.15 | 0.00 | 0.00 | 1.56 | + | none (best 0.139) |
| 10 | -43 | -2.73 | 2.59 | 0.00 | 0.00 | 0.07 | 0.38 | 0.55 | 0.466 | + | none (best 0.0459) |
| 11 | -43 | 2.26 | 2.51 | 0.41 | 0.44 | 0.14 | 0.00 | 0.00 | 1.89 | - | none (best 0.146) |
| 12 | -30 | 0.863 | 3.55 | 0.00 | 0.36 | 0.62 | 0.02 | 0.00 | 0.574 | - | rounded hinge |
| 13 | 138 | 2.74 | 2.5 | 0.58 | 0.35 | 0.07 | 0.00 | 0.00 | 1.09 | - | none (best 0.119) |
| 14 | -135 | -1.03 | 2.17 | 0.00 | 0.06 | 0.36 | 0.40 | 0.18 | 1.31 | - | none (best 0.116) |
| 15 | 132 | 2.52 | 3.09 | 0.32 | 0.55 | 0.14 | 0.00 | 0.00 | 1.26 | - | none (best 0.12) |

## Groups

- ring: 6 units (0, 1, 2, 5, 6, 9) with gates at distance 2.34-2.90, switch width ~3.2, magnitude ~1.2 each, positive
- ring: 5 units (3, 8, 11, 13, 15) with gates at distance 2.26-2.74, switch width ~2.5, magnitude ~1.3 each, negative
- the rest: units 4, 7, 10, 12, 14
