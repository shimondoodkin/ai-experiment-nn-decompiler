# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (reciprocal, 2 correction terms, 88 nodes), max error 2.7% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = 0.01051 * x0 * x0 + 0.0105 * x1 * x1 - 0.0004079 * x0 * x1 + 0.0006249 * x0 - 0.0005818 * x1 + 0.1572
    return -1 / u + 14 * relu(0.15 - u) - 3.5 * relu(u - 0.31) + 7
```

Where it is off (formula minus net, % of the output range):
- largest error 2.7%, at the box edge (x0 = -3, x1 = -2.18)
- by distance from (-0.0292, 0.0271), rings 0.427 wide, mean: +1.3 +1.4 +1.4 +1.0 +0.4 -0.2 -0.4 -0.5 +0.3 +0.3; largest: +1.9 +2.4 +2.4 +2.3 +2.4 +2.5 +2.5 +2.6 +2.7 +2.4
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.6 -0.7 -0.5 -0.2 -0.4 -0.2 -0.5 -0.6 +0.8 +1.4 +1.6 +1.4; largest: +2.4 +2.5 +2.6 +2.3 +2.4 +2.3 +2.5 +2.7 +2.4 +2.4 +2.1 +2.3

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0, x1):
    return -1 / (0.01475 * x0 * x0 + 0.01462 * x1 * x1 - 0.0002664 * x0 * x1 + 0.001049 * x0 - 0.0007783 * x1 + 0.172) + 6.321
```

Clean:

```python
def net(x0, x1):
    return -1 / (0.01475 * x0 * x0 + 0.01462 * x1 * x1 - 0.0002664 * x0 * x1 + 0.001049 * x0 - 0.0007783 * x1 + 0.172) + 6.321
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.0353, 0.0263), rings 0.428 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 3.0%, at the box edge (x0 = -2.99, x1 = 2.99)
  - by distance from (-0.0353, 0.0263), rings 0.428 wide, mean: -2.3 -1.5 -0.5 +0.3 +0.4 +0.2 -0.1 -0.6 -0.6 -0.0; largest: +2.7 +2.5 +1.5 +2.3 +2.7 +2.7 +2.7 +2.8 +2.8 +3.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.7 -0.6 -0.7 -0.8 -1.2 -0.7 -0.7 -0.8 +0.4 +0.8 +1.0 +1.2; largest: +2.7 +2.7 +2.8 +2.5 +3.0 +2.5 +2.8 +2.8 +2.7 +2.7 +2.7 +2.7

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0693, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -38 | -9.76 | 21.2 | 0.00 | 0.00 | 0.17 | 0.83 | 0.00 | 0.256 | + | rounded hinge |
| 1 | -37 | -17.2 | 29.3 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.101 | + | rounded hinge |
| 2 | -38 | -5.22 | 14 | 0.00 | 0.00 | 0.51 | 0.49 | 0.00 | 0.681 | + | none (best 0.0464) |
| 3 | -151 | 1.68 | 3.75 | 0.13 | 0.42 | 0.42 | 0.03 | 0.00 | 3.06 | + | none (best 0.18) |
| 4 | -37 | -11.6 | 23.6 | 0.00 | 0.00 | 0.06 | 0.94 | 0.00 | 0.258 | + | rounded hinge |
| 5 | -10 | -1.61 | 3.3 | 0.00 | 0.03 | 0.40 | 0.35 | 0.21 | 2.67 | - | none (best 0.168) |
| 6 | 143 | 12 | 24.1 | 0.00 | 0.96 | 0.04 | 0.00 | 0.00 | 0.247 | - | rounded hinge |
| 7 | 143 | 15.9 | 28.1 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.121 | - | rounded hinge |
| 8 | 143 | 13 | 25.2 | 0.00 | 0.99 | 0.01 | 0.00 | 0.00 | 0.207 | - | rounded hinge |
| 9 | 100 | 1.63 | 3.36 | 0.21 | 0.36 | 0.41 | 0.03 | 0.00 | 2.89 | + | none (best 0.185) |
| 10 | -37 | -16.6 | 28.8 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.146 | + | rounded hinge |
| 11 | -38 | -9.91 | 21.5 | 0.00 | 0.00 | 0.16 | 0.84 | 0.00 | 0.335 | + | none (best 0.0208) |
| 12 | -38 | -5.82 | 15.1 | 0.00 | 0.00 | 0.47 | 0.53 | 0.00 | 0.506 | + | none (best 0.0341) |
| 13 | -119 | -1.59 | 3.4 | 0.00 | 0.05 | 0.39 | 0.39 | 0.17 | 2.41 | - | none (best 0.181) |
| 14 | -37 | -13.7 | 25.9 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.197 | + | rounded hinge |
| 15 | 137 | -2.03 | 5.16 | 0.00 | 0.00 | 0.48 | 0.49 | 0.03 | 4.6 | - | none (best 0.299) |

## Groups

- same line: units 0, 11 (gate angle -38 degrees, offset -9.76)
- the rest: units 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15
