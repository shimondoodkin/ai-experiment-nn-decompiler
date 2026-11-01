# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (21 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 0.02)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.001 * x0 * x0 + x1 * x1 - 0.0009318 * x0 * x1 - 0.001279 * x0 - 0.003239 * x1 + 0.01242) - 0.01022
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

% of the output range:
- distortion (clean minus faithful): largest 2.2%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0, 0), rings 0.423 wide, mean: -0.3 +0.0 +0.1 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2; largest: +2.2 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.2 +0.2 +0.2 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1; largest: +1.8 +2.0 +1.5 +2.2 +1.7 +1.6 +1.7 +2.1 +1.4 +1.4 +1.4 +2.0
- ripple (faithful minus net): largest 0.7%, at the box edge (x0 = -0.141, x1 = 2.87)
  - by distance from (0.00064, 0.00162), rings 0.423 wide, mean: +0.0 -0.0 -0.1 -0.1 -0.1 -0.2 -0.2 -0.2 -0.2 -0.1; largest: +0.7 +0.6 +0.4 +0.5 +0.5 +0.6 +0.7 +0.7 +0.7 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.1 -0.2 -0.1 -0.1 -0.1 -0.2 -0.2 -0.2 -0.1 -0.2 -0.1; largest: +0.7 +0.6 +0.6 +0.7 +0.7 +0.6 +0.7 +0.6 +0.7 +0.7 +0.7 +0.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.01242 | 2.15% |
| offset | 0 | -0.01022 | 0.25% |
| x1 inside sqrt | 0 | -0.003239 | 0.04% |
| x0² inside sqrt | 1 | 1.001 | 0.04% |
| x0·x1 inside sqrt | 0 | -0.0009318 | 0.02% |
| x0 inside sqrt | 0 | -0.001279 | 0.02% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0827, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -133 | 0.54 | 0 | 0.62 | 0.38 | 0.277 | + | hinge |
| 1 | -115 | 0.187 | 0 | 0.53 | 0.47 | 0.868 | + | hinge |
| 2 | -75 | 0.0582 | 0 | 0.51 | 0.49 | 1.17 | + | hinge |
| 3 | 3 | 0.00169 | 0 | 0.50 | 0.50 | 1.23 | + | hinge |
| 4 | 67 | 0.147 | 0 | 0.53 | 0.47 | 0.72 | + | hinge |
| 5 | 105 | 0.263 | 0 | 0.55 | 0.45 | 0.27 | + | hinge |
| 6 | -156 | -0.0013 | 0 | 0.50 | 0.50 | 1.3 | + | hinge |
| 7 | 163 | -0.00158 | 0 | 0.50 | 0.50 | 1.16 | + | hinge |
| 8 | 85 | -0.00335 | 0 | 0.50 | 0.50 | 0.929 | + | hinge |
| 9 | 44 | -0.0739 | 0 | 0.48 | 0.52 | 1.18 | + | hinge |
| 10 | -52 | -0.104 | 0 | 0.48 | 0.52 | 1.12 | + | hinge |
| 11 | -52 | 0.248 | 0 | 0.55 | 0.45 | 0.466 | + | hinge |
| 12 | -137 | -0.267 | 0 | 0.44 | 0.56 | 0.413 | + | hinge |
| 13 | 146 | 0.00605 | 0 | 0.50 | 0.50 | 1.15 | + | hinge |
| 14 | -127 | 0.587 | 0 | 0.62 | 0.38 | 0.324 | - | hinge |
| 15 | 130 | 0.551 | 0 | 0.62 | 0.38 | 0.00753 | + | constant |

## Groups

- negligible: units 15 (magnitude 0.00753 together; a quarter of the tolerance is 0.0207)
- same line: units 0, 14 (gate angle -133 degrees, offset 0.54)
- the rest: units 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
