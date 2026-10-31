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
    return 0.006977 * x0 * x0 + 0.008035 * x1 * x1 + 0.9899 * x0 * x1 - 0.001643 * x0 + 0.0007739 * x1 - 0.02122
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 1.1%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.3 +0.3 +0.2 +0.2 +0.1 +0.0 -0.1 -0.2 -0.3 -0.4; largest: +0.3 +0.3 +0.3 +0.3 +0.2 +0.2 +0.4 +0.6 +0.8 +1.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.2 +0.2 +0.1 -0.1 -0.3 -0.0 +0.1 +0.1 +0.1 -0.0 -0.2 +0.0; largest: +0.3 +0.3 +0.3 +0.6 +1.1 +0.6 +0.3 +0.3 +0.3 +0.5 +0.9 +0.5
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = -2, x1 = 1.39)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.5 -0.4 -0.4 -0.3 -0.3 -0.3 -0.2 -0.1 +0.0 +0.0; largest: +0.5 +0.5 +0.5 +0.5 +0.5 +0.4 +0.4 +0.4 +0.5 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.3 -0.4 -0.3 -0.2 +0.0 -0.2 -0.3 -0.4 -0.3 -0.2 -0.0 -0.2; largest: +0.5 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9899 | 0.51% |
| x1² | 0 | 0.008035 | 0.41% |
| x0² | 0 | 0.006977 | 0.36% |
| constant | 0 | -0.02122 | 0.27% |
| x0 | 0 | -0.001643 | 0.04% |
| x1 | 0 | 0.0007739 | 0.02% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.156, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 133 | 2.06 | 3.07 | 0.21 | 0.56 | 0.23 | 0.00 | 0.00 | 2.47 | - | none (best 0.18) |
| 1 | -132 | 2.63 | 3.15 | 0.33 | 0.55 | 0.12 | 0.00 | 0.00 | 0.986 | + | none (best 0.0926) |
| 2 | -44 | 0.932 | 2.22 | 0.15 | 0.39 | 0.39 | 0.07 | 0.00 | 1.44 | - | none (best 0.136) |
| 3 | 44 | 2.82 | 2.88 | 0.48 | 0.45 | 0.07 | 0.00 | 0.00 | 1.05 | + | none (best 0.107) |
| 4 | 42 | 1.49 | 3.33 | 0.06 | 0.52 | 0.42 | 0.00 | 0.00 | 0.524 | + | rounded hinge |
| 5 | -136 | -2.9 | 2.41 | 0.00 | 0.00 | 0.04 | 0.30 | 0.66 | 0.873 | - | none (best 0.0979) |
| 6 | -155 | -0.948 | 3.48 | 0.00 | 0.01 | 0.59 | 0.40 | 0.00 | 0.0307 | + | constant |
| 7 | -135 | -2.53 | 3.5 | 0.00 | 0.00 | 0.16 | 0.62 | 0.22 | 0.742 | - | none (best 0.0667) |
| 8 | 46 | 0.975 | 2.27 | 0.15 | 0.40 | 0.39 | 0.06 | 0.00 | 0.713 | + | none (best 0.0635) |
| 9 | -75 | -0.16 | 4.23 | 0.00 | 0.06 | 0.81 | 0.13 | 0.00 | 0.75 | + | none (best 0.0418) |
| 10 | 46 | -1.54 | 2.73 | 0.00 | 0.00 | 0.33 | 0.50 | 0.17 | 1.85 | - | none (best 0.121) |
| 11 | 133 | 2.86 | 2.25 | 0.69 | 0.27 | 0.04 | 0.00 | 0.00 | 1.52 | - | none (best 0.163) |
| 12 | -134 | 2.81 | 1.88 | 0.77 | 0.19 | 0.03 | 0.00 | 0.00 | 1.42 | + | none (best 0.159) |
| 13 | -42 | 2.49 | 2.49 | 0.50 | 0.40 | 0.10 | 0.00 | 0.00 | 1.7 | - | none (best 0.152) |
| 14 | -135 | -2.71 | 3.2 | 0.00 | 0.00 | 0.11 | 0.55 | 0.35 | 1 | - | none (best 0.0976) |
| 15 | 136 | -2.7 | 2.3 | 0.00 | 0.00 | 0.06 | 0.31 | 0.63 | 1.66 | + | none (best 0.162) |

## Groups

- negligible: units 6 (magnitude 0.0307 together; a quarter of the tolerance is 0.0391)
- same line: units 1, 12 (gate angle -132 degrees, offset 2.63)
- same line: units 3, 5, 7, 14 (gate angle 44 degrees, offset 2.82)
- the rest: units 0, 2, 4, 8, 9, 10, 11, 13, 15
