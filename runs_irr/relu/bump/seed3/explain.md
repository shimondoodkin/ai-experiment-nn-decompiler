# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 8.4% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 8.3%, inside the box (x0 = 1.21, x1 = 2.1)
- by distance from (0, 0), rings 0.423 wide, mean: +1.4 +1.7 -0.1 +0.3 -0.3 +0.5 +1.1 +1.0 +0.9 +1.2; largest: +5.9 +4.4 +5.4 +8.1 +6.2 +8.3 +6.9 +5.7 +6.2 +5.7
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.3 +0.2 +0.3 +0.4 +0.7 +1.1 +0.9 +0.8 +0.6 +0.3 +1.2 +0.6; largest: +8.0 +8.3 +8.0 +7.4 +7.7 +8.1 +5.6 +5.6 +7.8 +7.5 +6.2 +7.2

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.009 * exp(-1.024 * x0 * x0 - 1.001 * x1 * x1 - 0.04011 * x0 * x1 + 0.04107 * x0 + 0.01561 * x1) - 0.004153
```

Clean:

```python
def net(x0, x1):
    return 1.009 * exp(-1.024 * x0 * x0 - 1.001 * x1 * x1 - 0.04011 * x0 * x1 + 0.04107 * x0 + 0.01561 * x1) - 0.004153
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.0199, 0.0074), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 7.9%, inside the box (x0 = 1.21, x1 = 2.1)
  - by distance from (0.0199, 0.0074), rings 0.424 wide, mean: +1.7 +1.6 -0.6 -0.3 -0.7 +0.1 +0.7 +0.6 +0.5 +0.9; largest: +5.4 +5.4 +5.6 +7.6 +6.4 +7.9 +6.4 +5.5 +6.5 +4.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 -0.0 +0.1 +0.3 +0.3 +0.4 +0.1 +0.1 +0.1 -0.0 +1.0 +0.4; largest: +7.6 +7.8 +7.9 +7.3 +7.2 +7.3 +5.1 +5.2 +7.2 +7.0 +7.0 +7.5

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0232, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -150 | 0.0399 | 0 | 0.51 | 0.49 | 1.96 | - | hinge |
| 1 | -103 | 1.36 | 0 | 0.73 | 0.27 | 0.341 | + | hinge |
| 2 | -58 | 1.19 | 0 | 0.73 | 0.27 | 1.14 | + | hinge |
| 3 | 46 | 0.108 | 0 | 0.53 | 0.47 | 0.722 | - | hinge |
| 4 | 98 | -0.0541 | 0 | 0.49 | 0.51 | 0.863 | - | hinge |
| 5 | 119 | 1.24 | 0 | 0.74 | 0.26 | 1.1 | + | hinge |
| 6 | -175 | 1.21 | 0 | 0.70 | 0.30 | 0.662 | + | hinge |
| 7 | 180 | -1.21 | 0 | 0.30 | 0.70 | 1.76 | + | hinge |
| 8 | 114 | -0.0309 | 0 | 0.50 | 0.50 | 0.268 | - | hinge |
| 9 | 55 | -1.21 | 0 | 0.26 | 0.74 | 1.62 | + | hinge |
| 10 | -20 | -1.31 | 0 | 0.27 | 0.73 | 0.676 | + | hinge |
| 11 | -39 | -0.0177 | 0 | 0.49 | 0.51 | 1.41 | - | hinge |
| 12 | -121 | -1.21 | 0 | 0.27 | 0.73 | 2.29 | + | hinge |
| 13 | -167 | -0.0494 | 0 | 0.49 | 0.51 | 0.575 | - | hinge |
| 14 | -95 | -0.0375 | 0 | 0.49 | 0.51 | 1.43 | - | hinge |
| 15 | 157 | -0.0272 | 0 | 0.50 | 0.50 | 1.56 | - | hinge |

## Groups

- ring: 3 units (2, 5, 6) with gates at distance 1.19-1.24, switch width ~0, magnitude ~1.1 each, positive
- the rest: units 0, 1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15
