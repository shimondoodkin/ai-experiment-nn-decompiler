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
    return 0.005807 * x0 * x0 + 0.004815 * x1 * x1 + 0.9906 * x0 * x1 - 0.0006697 * x0 - 0.0004986 * x1 - 0.02361
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.3 +0.3 +0.3 +0.2 +0.2 +0.1 +0.1 -0.0 -0.1 -0.2; largest: +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.4 +0.5 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.2 +0.3 +0.2 +0.1 -0.1 +0.1 +0.2 +0.3 +0.2 +0.1 -0.1 +0.1; largest: +0.3 +0.3 +0.3 +0.3 +0.7 +0.4 +0.3 +0.3 +0.3 +0.3 +0.7 +0.3
- ripple (faithful minus net): largest 0.7%, at the box edge (x0 = 2, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.3 -0.3 -0.3 -0.3 -0.3 -0.2 -0.1 -0.1 -0.0 -0.0; largest: +0.3 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.3 -0.3 -0.3 -0.2 -0.0 -0.2 -0.3 -0.3 -0.3 -0.2 -0.0 -0.2; largest: +0.3 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.3 +0.4 +0.7 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9906 | 0.48% |
| constant | 0 | -0.02361 | 0.30% |
| x0² | 0 | 0.005807 | 0.30% |
| x1² | 0 | 0.004815 | 0.25% |
| x0 | 0 | -0.0006697 | 0.02% |
| x1 | 0 | -0.0004986 | 0.01% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.156, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -45 | -2.77 | 3.21 | 0.00 | 0.00 | 0.10 | 0.54 | 0.36 | 1.05 | + | none (best 0.109) |
| 1 | 43 | -2.84 | 1.94 | 0.00 | 0.00 | 0.03 | 0.20 | 0.77 | 1.31 | - | none (best 0.154) |
| 2 | 56 | 1.12 | 3.27 | 0.03 | 0.44 | 0.52 | 0.01 | 0.00 | 1.05 | + | none (best 0.0652) |
| 3 | -134 | 2.41 | 2.76 | 0.38 | 0.49 | 0.13 | 0.00 | 0.00 | 1.65 | + | none (best 0.143) |
| 4 | -45 | -2.76 | 3.58 | 0.00 | 0.00 | 0.13 | 0.62 | 0.25 | 1.12 | + | none (best 0.108) |
| 5 | 46 | 2.86 | 3.74 | 0.24 | 0.65 | 0.12 | 0.00 | 0.00 | 0.971 | + | none (best 0.0969) |
| 6 | 134 | 1.26 | 2.54 | 0.15 | 0.45 | 0.37 | 0.02 | 0.00 | 1.68 | - | none (best 0.101) |
| 7 | 42 | 2.99 | 3.07 | 0.47 | 0.47 | 0.06 | 0.00 | 0.00 | 0.803 | + | none (best 0.0831) |
| 8 | -51 | 2.73 | 3.41 | 0.29 | 0.59 | 0.12 | 0.00 | 0.00 | 1.17 | - | none (best 0.117) |
| 9 | 133 | 2.68 | 1.69 | 0.79 | 0.17 | 0.04 | 0.00 | 0.00 | 1.13 | - | none (best 0.114) |
| 10 | -129 | -3.03 | 2.25 | 0.00 | 0.00 | 0.02 | 0.24 | 0.74 | 0.473 | - | none (best 0.053) |
| 11 | 136 | -2.51 | 3.86 | 0.00 | 0.00 | 0.20 | 0.67 | 0.14 | 1.63 | + | none (best 0.145) |
| 12 | 138 | -2.82 | 3.27 | 0.00 | 0.00 | 0.10 | 0.55 | 0.35 | 1.58 | + | none (best 0.166) |
| 13 | -135 | -2.84 | 3.55 | 0.00 | 0.00 | 0.11 | 0.61 | 0.28 | 1.07 | - | none (best 0.11) |
| 14 | -124 | -2.7 | 4.73 | 0.00 | 0.00 | 0.23 | 0.73 | 0.04 | 0.535 | - | none (best 0.0461) |
| 15 | -140 | 1.14 | 2.96 | 0.06 | 0.45 | 0.47 | 0.02 | 0.00 | 1.92 | + | none (best 0.118) |

## Groups

- same line: units 0, 4, 9 (gate angle -45 degrees, offset -2.77)
- same line: units 5, 7, 10, 13, 14 (gate angle 46 degrees, offset 2.86)
- same line: units 8, 12 (gate angle -51 degrees, offset 2.73)
- the rest: units 1, 2, 3, 6, 11, 15
