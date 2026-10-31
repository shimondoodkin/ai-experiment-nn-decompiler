# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (reciprocal, 0 correction terms, 39 nodes), max error 6.1% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return -1 / (0.01328 * x0 * x0 + 0.01397 * x1 * x1 - 0.001886 * x0 + 0.17) + 6.5
```

Where it is off (formula minus net, % of the output range):
- largest error 6.1%, inside the box (x0 = -0.944, x1 = 1.88)
- by distance from (0.071, 0), rings 0.428 wide, mean: -0.7 +0.1 +1.2 +2.1 +2.4 +2.2 +1.7 +1.1 +0.5 -0.3; largest: +0.9 +1.4 +3.6 +5.4 +6.1 +6.1 +5.5 +5.7 +5.6 +5.9
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.6 -0.4 +2.1 +3.9 +3.1 +2.9 +2.6 +2.4 +3.3 +1.5 -0.8 -0.8; largest: +3.2 +5.7 +4.6 +6.1 +6.1 +5.7 +4.9 +5.9 +5.5 +3.9 +5.5 +3.4

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0, x1):
    return -1 / (0.01344 * x0 * x0 + 0.0143 * x1 * x1 + 9.832e-05 * x0 * x1 - 0.00278 * x0 - 0.0007883 * x1 + 0.1658) + 6.492
```

Clean:

```python
def net(x0, x1):
    return -1 / (0.01344 * x0 * x0 + 0.0143 * x1 * x1 + 9.832e-05 * x0 * x1 - 0.00278 * x0 - 0.0007883 * x1 + 0.1658) + 6.492
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.103, 0.0272), rings 0.431 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 5.3%, inside the box (x0 = 0.173, x1 = 0.0799)
  - by distance from (0.103, 0.0272), rings 0.431 wide, mean: -5.0 -3.9 -2.3 -0.8 +0.2 +0.6 +0.6 +0.6 +0.3 -0.7; largest: +5.3 +5.0 +4.3 +3.8 +5.1 +5.2 +5.3 +5.2 +5.2 +5.2
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -3.8 -3.0 -0.7 +1.6 +1.5 +1.4 +1.5 +2.0 +2.4 -0.0 -2.6 -3.7; largest: +5.3 +5.3 +5.3 +5.3 +5.3 +5.3 +5.3 +5.3 +5.3 +5.3 +5.3 +5.3

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0731, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -60 | 1.87 | 3.85 | 0.14 | 0.44 | 0.40 | 0.02 | 0.00 | 4.8 | + | none (best 0.3) |
| 1 | 12 | 54.7 | 81 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0227 | - | constant |
| 2 | 12 | 48.2 | 77.1 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0418 | - | rounded hinge |
| 3 | 14 | 33.6 | 66.7 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0666 | - | rounded hinge |
| 4 | -168 | -44.9 | 75.5 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0351 | + | constant |
| 5 | 18 | 22.1 | 54.1 | 0.00 | 0.83 | 0.17 | 0.00 | 0.00 | 0.11 | - | rounded hinge |
| 6 | -164 | -26 | 58.9 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.109 | + | rounded hinge |
| 7 | -180 | -1.2 | 2.84 | 0.00 | 0.12 | 0.36 | 0.30 | 0.23 | 0.933 | - | none (best 0.059) |
| 8 | -168 | -57.9 | 82.6 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0169 | + | constant |
| 9 | 14 | 31.6 | 64.8 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0748 | - | rounded hinge |
| 10 | -167 | -43.1 | 74.2 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0453 | + | rounded hinge |
| 11 | 60 | 1.89 | 3.93 | 0.13 | 0.45 | 0.40 | 0.02 | 0.00 | 4.99 | + | none (best 0.318) |
| 12 | 15 | 30.4 | 63.6 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0712 | - | rounded hinge |
| 13 | -167 | -35.9 | 68.7 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.065 | + | rounded hinge |
| 14 | -180 | 1.65 | 4.73 | 0.00 | 0.48 | 0.52 | 0.00 | 0.00 | 4.26 | + | none (best 0.285) |
| 15 | -158 | -17.8 | 47.5 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.147 | + | rounded hinge |

## Groups

- negligible: units 8 (magnitude 0.0169 together; a quarter of the tolerance is 0.0183)
- lines: units 1, 4 (each a constant or a line on the box; together one linear term)
- ring: 3 units (0, 11, 14) with gates at distance 1.65-1.89, switch width ~3.9, magnitude ~4.8 each, positive
- the rest: units 2, 3, 5, 6, 7, 9, 10, 12, 13, 15
