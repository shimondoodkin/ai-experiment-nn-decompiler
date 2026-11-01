# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 3 correction terms, 82 nodes), max error 2.9% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = -0.25 * x0 * x0 - 0.25 * x1 * x1
    return -3 * exp(u) + 0.4 * relu(u + 1) + 0.13 * relu(-0.5 - u) + 0.25 * relu(-2 - u) + 3
```

Where it is off (formula minus net, % of the output range):
- largest error 2.9%, at the box edge (x0 = 3, x1 = -0.582)
- by distance from (0, 0), rings 0.423 wide, mean: -1.3 -0.6 +0.1 +0.2 -0.6 -0.4 -0.9 -0.8 -1.1 -1.0; largest: +1.7 +1.7 +1.3 +1.4 +2.5 +2.1 +2.7 +2.9 +2.2 +2.9
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.7 -1.0 -0.9 -0.6 -0.5 +0.1 -0.6 -0.5 -0.6 -0.5 -0.6 -0.2; largest: +2.5 +2.1 +2.5 +2.3 +2.7 +2.1 +1.8 +2.1 +2.2 +1.8 +2.9 +2.9

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return -4.355 * exp(-0.08895 * x0 * x0 - 0.0928 * x1 * x1 - 0.003587 * x0 * x1 - 0.00473 * x0 + 0.002651 * x1) + 4.938
```

Clean:

```python
def net(x0, x1):
    return -4.355 * exp(-0.08895 * x0 * x0 - 0.0928 * x1 * x1 - 0.003587 * x0 * x1 - 0.00473 * x0 + 0.002651 * x1) + 4.938
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.0269, 0.0148), rings 0.426 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 3.8%, inside the box (x0 = 1.97, x1 = -0.029)
  - by distance from (-0.0269, 0.0148), rings 0.426 wide, mean: +3.0 +1.6 -0.3 -2.0 -2.5 -1.9 -0.4 +0.4 +0.1 -0.5; largest: +3.7 +3.5 +2.3 +3.5 +3.8 +3.7 +3.7 +3.7 +2.3 +3.6
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.9 -0.2 -0.8 -1.4 -1.9 -2.0 -1.9 -0.1 -0.1 -0.2 -0.4 -0.7; largest: +3.8 +3.5 +3.6 +3.5 +3.7 +3.6 +3.7 +3.3 +3.3 +3.4 +3.6 +3.8

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0752, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -122 | 36.3 | 56 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0539 | - | rounded hinge |
| 1 | -106 | 0.721 | 2.65 | 0.16 | 0.29 | 0.35 | 0.20 | 0.01 | 2.08 | + | none (best 0.25) |
| 2 | -101 | 2.7 | 3.24 | 0.41 | 0.35 | 0.25 | 0.00 | 0.00 | 1.76 | + | none (best 0.133) |
| 3 | -122 | 26 | 50.3 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0998 | - | rounded hinge |
| 4 | -122 | 25.9 | 50.2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.108 | - | rounded hinge |
| 5 | 39 | 0.885 | 1.83 | 0.31 | 0.24 | 0.26 | 0.13 | 0.06 | 0.918 | + | none (best 0.109) |
| 6 | 137 | -1 | 2.12 | 0.03 | 0.13 | 0.29 | 0.28 | 0.27 | 1.25 | - | none (best 0.149) |
| 7 | 58 | -27.8 | 51.5 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.118 | + | rounded hinge |
| 8 | 104 | 1.32 | 4.04 | 0.05 | 0.42 | 0.49 | 0.04 | 0.00 | 3.2 | + | none (best 0.19) |
| 9 | -1 | -1.45 | 4.07 | 0.00 | 0.00 | 0.51 | 0.42 | 0.06 | 2.85 | - | none (best 0.185) |
| 10 | 3 | 1.79 | 4.09 | 0.12 | 0.43 | 0.46 | 0.00 | 0.00 | 2.41 | + | none (best 0.164) |
| 11 | -122 | 32.1 | 54 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0751 | - | rounded hinge |
| 12 | 58 | -25.9 | 50.2 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.114 | + | rounded hinge |
| 13 | -123 | 23.1 | 48.1 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.115 | - | rounded hinge |
| 14 | -122 | 30.2 | 53 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.088 | - | rounded hinge |
| 15 | -123 | 21.3 | 46.7 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.144 | - | rounded hinge |

## Groups

- same line: units 3, 4, 12 (gate angle -122 degrees, offset 26)
- the rest: units 0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15
