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
    return -0.003304 * x0 * x0 - 0.003707 * x1 * x1 + 0.9942 * x0 * x1 - 0.0002265 * x0 + 0.003499 * x1 + 0.01487
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = -1.99, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.2 -0.2 -0.2 -0.1 -0.1 -0.1 -0.0 +0.0 +0.1 +0.1; largest: +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.3 +0.4 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.1 +0.0 -0.1 -0.2 -0.2 -0.2 -0.0 +0.1 +0.0 -0.1 -0.1 -0.1; largest: +0.2 +0.4 +0.2 +0.2 +0.2 +0.2 +0.3 +0.5 +0.3 +0.2 +0.2 +0.2
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = 2, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.2 +0.2 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1 +0.0 -0.0; largest: +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3 +0.6
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 -0.0 +0.1 +0.2 +0.2 +0.2 +0.1 -0.0 +0.1 +0.2 +0.2 +0.1; largest: +0.2 +0.4 +0.2 +0.2 +0.4 +0.3 +0.2 +0.3 +0.2 +0.2 +0.6 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9942 | 0.30% |
| constant | 0 | 0.01487 | 0.19% |
| x1² | 0 | -0.003707 | 0.19% |
| x0² | 0 | -0.003304 | 0.17% |
| x1 | 0 | 0.003499 | 0.09% |
| x0 | 0 | -0.0002265 | 0.01% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.157, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -135 | -1.76 | 3.65 | 0.00 | 0.00 | 0.37 | 0.57 | 0.05 | 0.853 | - | none (best 0.0598) |
| 1 | 133 | 2.96 | 2.34 | 0.70 | 0.27 | 0.04 | 0.00 | 0.00 | 0.663 | - | none (best 0.0742) |
| 2 | 135 | 2.83 | 3.17 | 0.39 | 0.52 | 0.09 | 0.00 | 0.00 | 0.999 | - | none (best 0.0992) |
| 3 | 136 | 2.67 | 3.54 | 0.24 | 0.62 | 0.14 | 0.00 | 0.00 | 0.761 | - | none (best 0.0741) |
| 4 | -45 | -2.86 | 3.12 | 0.00 | 0.00 | 0.08 | 0.51 | 0.41 | 0.712 | + | none (best 0.0738) |
| 5 | -44 | 2.8 | 2.09 | 0.72 | 0.24 | 0.04 | 0.00 | 0.00 | 1.79 | - | none (best 0.199) |
| 6 | -45 | -2.29 | 4.06 | 0.00 | 0.00 | 0.26 | 0.67 | 0.07 | 1.03 | + | none (best 0.0819) |
| 7 | -135 | 2.38 | 2.84 | 0.35 | 0.51 | 0.14 | 0.00 | 0.00 | 1.96 | + | none (best 0.175) |
| 8 | 44 | -2.83 | 1.93 | 0.00 | 0.00 | 0.03 | 0.20 | 0.77 | 1.25 | - | none (best 0.146) |
| 9 | -47 | 2.24 | 3.18 | 0.22 | 0.58 | 0.20 | 0.00 | 0.00 | 1.35 | - | none (best 0.106) |
| 10 | -133 | -2.88 | 2.32 | 0.00 | 0.00 | 0.04 | 0.28 | 0.68 | 1.04 | - | none (best 0.11) |
| 11 | -136 | -2.85 | 2.81 | 0.00 | 0.00 | 0.07 | 0.42 | 0.51 | 1.4 | - | none (best 0.151) |
| 12 | 134 | 0.957 | 2.45 | 0.11 | 0.40 | 0.43 | 0.06 | 0.00 | 0.595 | - | none (best 0.0478) |
| 13 | 44 | -1.08 | 2.8 | 0.00 | 0.03 | 0.46 | 0.43 | 0.07 | 1.54 | - | none (best 0.0929) |
| 14 | 136 | -1.25 | 2.89 | 0.00 | 0.01 | 0.43 | 0.47 | 0.09 | 1.56 | + | none (best 0.0993) |
| 15 | 46 | 1.41 | 3.45 | 0.04 | 0.50 | 0.46 | 0.00 | 0.00 | 1.49 | + | none (best 0.0975) |

## Groups

- same line: units 1, 2, 3, 4 (gate angle 133 degrees, offset 2.96)
- same line: units 10, 11 (gate angle -133 degrees, offset -2.88)
- the rest: units 0, 5, 6, 7, 8, 9, 12, 13, 14, 15
