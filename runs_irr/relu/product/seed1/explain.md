# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (41 nodes):

```python
def net(x0, x1):
    u = -0.02157 * x0 * x1
    return -46 * exp(u) + 5 * relu(u - 0.05) + 46.04
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return -16570 * exp(1.619e-07 * x0 * x0 + 9.045e-07 * x1 * x1 - 5.883e-05 * x0 * x1 + 7.752e-07 * x0 - 4.284e-07 * x1) + 16570
```

Clean:

```python
def net(x0, x1):
    return -16570 * exp(1.619e-07 * x0 * x0 + 9.045e-07 * x1 * x1 - 5.883e-05 * x0 * x1 + 7.752e-07 * x0 - 4.284e-07 * x1) + 16570
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.548, x1 = 0.428)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.6%, inside the box (x0 = 1.61, x1 = 1.36)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.6 -0.0 -0.1 -0.5 -0.3 -0.2 -0.6 -0.9 -0.6 -0.3; largest: +1.2 +1.1 +1.5 +1.5 +1.5 +1.7 +2.6 +2.6 +2.5 +2.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.4 -1.1 -0.7 -0.4 -0.2 +0.1 -0.2 -1.0 -0.6 -0.5 -0.1 -0.1; largest: +1.8 +2.6 +2.2 +1.5 +1.2 +1.3 +1.9 +2.6 +2.5 +1.5 +2.5 +1.4

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.153, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -47 | 0.0907 | 0 | 0.53 | 0.47 | 2 | - | hinge |
| 1 | 130 | 0.509 | 0 | 0.66 | 0.34 | 1.21 | - | hinge |
| 2 | 131 | 1.03 | 0 | 0.80 | 0.20 | 0.933 | - | hinge |
| 3 | 122 | -0.804 | 0 | 0.26 | 0.74 | 1.18 | - | hinge |
| 4 | 122 | -2.69 | 0 | 0.00 | 1.00 | 0.0837 | + | line |
| 5 | -41 | 0.975 | 0 | 0.79 | 0.21 | 1.16 | - | hinge |
| 6 | -78 | -2.26 | 0 | 0.00 | 1.00 | 0.0501 | - | constant |
| 7 | -129 | 4.01 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | -47 | -2.17 | 0 | 0.03 | 0.97 | 4.27 | - | hinge |
| 9 | -9 | 0.814 | 0 | 0.71 | 0.29 | 0.153 | + | hinge |
| 10 | -131 | -0.531 | 0 | 0.33 | 0.67 | 3.02 | + | hinge |
| 11 | -130 | -1.47 | 0 | 0.11 | 0.89 | 4.18 | + | hinge |
| 12 | 130 | -1.73 | 0 | 0.08 | 0.92 | 3.58 | - | hinge |
| 13 | 50 | -0.387 | 0 | 0.38 | 0.62 | 3.02 | + | hinge |
| 14 | 131 | 1.52 | 0 | 0.89 | 0.11 | 0.644 | - | hinge |
| 15 | 49 | -1.4 | 0 | 0.13 | 0.87 | 4.65 | + | hinge |

## Groups

- negligible: units 7 (magnitude 0 together; a quarter of the tolerance is 0.0382)
- lines: units 4, 6 (each a constant or a line on the box; together one linear term)
- the rest: units 0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15
