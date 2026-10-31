# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (reciprocal, 2 correction terms, 61 nodes), max error 12.7% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = 0.4 * x0 * x0 + 0.4 * x1 * x1 + 0.75
    return 1 / u + 0.2 * relu(u - 2.5) - 0.2 * relu(u - 4) - 0.45
```

Where it is off (formula minus net, % of the output range):
- largest error 12.7%, at the box edge (x0 = -2.78, x1 = -0.636)
- by distance from (0, 0), rings 0.423 wide, mean: -1.0 +1.6 +1.0 +2.9 -1.6 -1.6 +5.1 +5.7 +3.4 +1.5; largest: +9.4 +5.3 +6.6 +9.8 +10.6 +10.0 +12.7 +11.7 +7.4 +4.0
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +1.4 +2.0 +1.5 +1.7 +2.9 +2.4 +2.5 +1.8 +1.9 +1.7 +2.4 +2.1; largest: +10.9 +10.4 +12.0 +11.0 +9.2 +10.4 +12.7 +10.6 +11.0 +10.2 +9.8 +10.0

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0, x1):
    return 1 / (1.551 * x0 * x0 + 1.222 * x1 * x1 + 0.1431 * x0 * x1 - 0.02352 * x0 - 0.08732 * x1 + 0.8554) - 0.1137
```

Clean:

```python
def net(x0, x1):
    return 1 / (1.551 * x0 * x0 + 1.222 * x1 * x1 + 0.1431 * x0 * x1 - 0.02352 * x0 - 0.08732 * x1 + 0.8554) - 0.1137
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.00595, 0.0354), rings 0.425 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 14.8%, inside the box (x0 = -2.13, x1 = 0.012)
  - by distance from (0.00595, 0.0354), rings 0.425 wide, mean: +7.1 -2.9 -5.6 +1.4 +3.0 +0.6 -3.0 -3.6 -3.7 -4.0; largest: +13.3 +13.1 +14.1 +11.5 +14.7 +14.8 +14.5 +13.9 +6.5 +5.9
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -2.3 -2.2 +0.4 +1.1 -0.5 -1.1 -1.7 -3.2 -0.5 -0.0 -1.5 -1.6; largest: +13.3 +10.2 +14.7 +14.4 +13.3 +13.9 +14.8 +8.7 +13.8 +13.9 +8.3 +12.9

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
