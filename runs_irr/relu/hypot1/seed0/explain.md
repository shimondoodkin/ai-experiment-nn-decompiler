# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

No formula passes the check. Near formula (sqrt, 0 correction terms, 24 nodes), max error 2.5% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 1) - 0.003859
```

Where it is off (formula minus net, % of the output range):
- largest error 2.5%, inside the box (x0 = 0.501, x1 = 0.189)
- by distance from (0, 0), rings 0.282 wide, mean: -0.1 -0.3 -0.4 -0.2 -0.4 -0.5 -0.4 -0.2 -0.2 -0.3; largest: +1.3 +2.5 +2.4 +1.9 +1.7 +1.8 +1.9 +2.5 +2.1 +1.8
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.1 -0.1 -0.3 +0.1 -0.4 -0.3 -0.4 -0.5 -0.6 -0.4 -0.6 -0.3; largest: +2.5 +2.1 +1.8 +2.5 +1.9 +1.8 +1.8 +1.9 +2.2 +1.9 +2.3 +1.4

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(0.9419 * x0 * x0 + 0.948 * x1 * x1 + 0.0106 * x0 * x1 - 0.029 * x0 - 0.02219 * x1 + 0.8115) + 0.08492
```

Clean:

```python
def net(x0, x1):
    return sqrt(0.9419 * x0 * x0 + 0.948 * x1 * x1 + 0.0106 * x0 * x1 - 0.029 * x0 - 0.02219 * x1 + 0.8115) + 0.08492
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.548, x1 = 0.428)
  - by distance from (0.0153, 0.0116), rings 0.283 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.1%, inside the box (x0 = -0.151, x1 = -0.522)
  - by distance from (0.0153, 0.0116), rings 0.283 wide, mean: -0.5 -0.6 -0.6 -0.2 -0.4 -0.6 -0.5 -0.5 -0.7 -0.9; largest: +1.3 +2.1 +2.1 +2.0 +2.0 +2.0 +2.0 +2.0 +2.0 +1.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.9 -0.8 -0.9 -0.3 -0.8 -0.1 +0.2 +0.2 -0.0 -0.2 -1.1 -1.0; largest: +2.0 +2.0 +1.8 +2.0 +2.0 +2.0 +1.7 +1.7 +2.1 +1.7 +2.0 +1.9

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0406, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 88 | 2.68 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | -128 | 0.123 | 0 | 0.54 | 0.46 | 0.664 | + | hinge |
| 2 | 162 | 0.269 | 0 | 0.57 | 0.43 | 0.548 | + | hinge |
| 3 | 98 | 0.116 | 0 | 0.53 | 0.47 | 0.503 | + | hinge |
| 4 | 117 | -0.0687 | 0 | 0.48 | 0.52 | 0.424 | + | hinge |
| 5 | -10 | -0.976 | 0 | 0.25 | 0.75 | 0.682 | + | hinge |
| 6 | -126 | -0.49 | 0 | 0.35 | 0.65 | 0.692 | + | hinge |
| 7 | -75 | -0.758 | 0 | 0.30 | 0.70 | 0.797 | + | hinge |
| 8 | 18 | 0.118 | 0 | 0.53 | 0.47 | 0.702 | + | hinge |
| 9 | -139 | 0.791 | 0 | 0.74 | 0.26 | 0.357 | + | hinge |
| 10 | 98 | -0.325 | 0 | 0.42 | 0.58 | 0.442 | + | hinge |
| 11 | 88 | 2.29 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 147 | 3.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | -20 | 0.401 | 0 | 0.61 | 0.39 | 0.726 | + | hinge |
| 14 | -165 | -0.856 | 0 | 0.28 | 0.72 | 0.338 | + | hinge |
| 15 | 103 | -0.901 | 0 | 0.27 | 0.73 | 0.625 | + | hinge |

## Groups

- negligible: units 0, 11, 12 (magnitude 0 together; a quarter of the tolerance is 0.0102)
- ring: 3 units (1, 3, 8) with gates at distance 0.12-0.12, switch width ~0, magnitude ~0.66 each, positive
- the rest: units 2, 4, 5, 6, 7, 9, 10, 13, 14, 15
