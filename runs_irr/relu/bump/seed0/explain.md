# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 4 correction terms, 81 nodes), max error 12.5% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = -5 * x0 * x0 - 5 * x1 * x1 + 0.25 * x0 * x1 - 0.2 * x0
    return 0.25 * exp(u) + 0.05 * relu(u + 15) - 0.002831 * relu(u + 45) + 0.05
```

Where it is off (formula minus net, % of the output range):
- largest error 12.5%, inside the box (x0 = -2.13, x1 = 0.012)
- by distance from (-0.02, -0.0005), rings 0.425 wide, mean: -1.5 -0.8 +2.8 +1.2 -3.9 +0.2 +2.0 +4.7 +5.9 +6.5; largest: +6.8 +6.4 +9.0 +9.2 +12.5 +12.0 +8.5 +9.4 +8.6 +8.6
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.2 +1.8 +0.3 +0.6 +2.8 +1.4 +1.7 +1.9 +1.0 +0.4 +2.1 +0.7; largest: +7.9 +8.7 +9.7 +9.1 +8.6 +12.5 +12.1 +10.5 +8.7 +9.8 +7.5 +11.6

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.026 * exp(-1.117 * x0 * x0 - 0.9127 * x1 * x1 - 0.09593 * x0 * x1 - 0.01951 * x0 - 0.06914 * x1) - 0.005116
```

Clean:

```python
def net(x0, x1):
    return 1.026 * exp(-1.117 * x0 * x0 - 0.9127 * x1 * x1 - 0.09593 * x0 * x1 - 0.01951 * x0 - 0.06914 * x1) - 0.005116
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.00712, -0.0375), rings 0.425 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 13.9%, inside the box (x0 = -2.13, x1 = 0.012)
  - by distance from (-0.00712, -0.0375), rings 0.425 wide, mean: +8.8 +3.7 -3.8 -1.7 +0.1 +0.1 -1.1 -0.1 +1.1 +1.8; largest: +12.3 +11.7 +13.0 +12.4 +13.9 +13.4 +11.6 +12.6 +3.8 +3.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -2.0 -1.0 -0.8 -0.1 +0.5 -0.5 -0.4 -0.6 +0.7 +0.9 +0.5 -0.7; largest: +13.0 +7.6 +12.6 +12.4 +11.8 +13.9 +12.5 +12.3 +12.2 +11.6 +10.3 +12.1

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0229, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 99 | -0.0233 | 0 | 0.50 | 0.50 | 1.38 | - | hinge |
| 1 | -132 | -0.0198 | 0 | 0.49 | 0.51 | 0.226 | - | hinge |
| 2 | 154 | 4.31 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 134 | 1.49 | 0 | 0.79 | 0.21 | 1.56 | + | hinge |
| 4 | 61 | 4.63 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 150 | -1.47 | 0 | 0.22 | 0.78 | 1.42 | + | hinge |
| 6 | -136 | -1.48 | 0 | 0.21 | 0.79 | 3.42 | + | hinge |
| 7 | 163 | -1.98e-05 | 0 | 0.50 | 0.50 | 0.933 | - | hinge |
| 8 | 128 | -1.46 | 0 | 0.22 | 0.78 | 2.27 | + | hinge |
| 9 | -178 | -0.0814 | 0 | 0.48 | 0.52 | 1.09 | - | hinge |
| 10 | 78 | -0.0177 | 0 | 0.50 | 0.50 | 1.49 | - | hinge |
| 11 | 131 | -0.087 | 0 | 0.48 | 0.52 | 0.566 | - | hinge |
| 12 | 165 | 10.2 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | -71 | 3 | 0 | 0.97 | 0.03 | 0.0196 | + | hinge |
| 14 | -164 | -0.0438 | 0 | 0.49 | 0.51 | 1.06 | - | hinge |
| 15 | 47 | -1.45 | 0 | 0.22 | 0.78 | 3.24 | + | hinge |

## Groups

- negligible: units 2, 4, 12 (magnitude 0 together; a quarter of the tolerance is 0.00572)
- the rest: units 0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15
