# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (21 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1) + 0.02
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.004 * x0 * x0 + 0.9983 * x1 * x1 - 0.002061 * x0 * x1 + 0.004282 * x0 + 0.0002795 * x1 + 0.01292) - 0.009803
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1) + 0.02
```

% of the output range:
- distortion (clean minus faithful): largest 1.8%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0, 0), rings 0.423 wide, mean: +0.1 +0.5 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6; largest: +1.8 +0.6 +0.6 +0.7 +0.7 +0.7 +0.7 +0.8 +0.8 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.5 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 +0.7 +0.7 +0.6 +0.5 +0.5; largest: +1.4 +1.6 +1.1 +1.8 +1.3 +1.1 +1.2 +1.7 +1.0 +1.0 +1.0 +1.6
- ripple (faithful minus net): largest 0.9%, at the box edge (x0 = -1.17, x1 = 2.96)
  - by distance from (-0.00213, -0.000142), rings 0.423 wide, mean: -0.0 +0.0 -0.1 -0.1 -0.1 -0.1 -0.2 -0.1 -0.2 +0.0; largest: +0.8 +0.6 +0.6 +0.5 +0.6 +0.7 +0.8 +0.9 +0.7 +0.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.1 -0.1 -0.1 -0.2 -0.1 -0.1 -0.2 -0.1 -0.1 -0.2 -0.1; largest: +0.7 +0.8 +0.8 +0.9 +0.7 +0.7 +0.8 +0.7 +0.8 +0.9 +0.7 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.01292 | 2.23% |
| offset | 0.02 | -0.009803 | 0.73% |
| x0² inside sqrt | 1 | 1.004 | 0.15% |
| x1² inside sqrt | 1 | 0.9983 | 0.06% |
| x0·x1 inside sqrt | 0 | -0.002061 | 0.05% |
| x0 inside sqrt | 0 | 0.004282 | 0.05% |
| x1 inside sqrt | 0 | 0.0002795 | 0.00% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0828, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -2 | -0.00508 | 0 | 0.50 | 0.50 | 1.22 | + | hinge |
| 1 | 110 | 0.00606 | 0 | 0.50 | 0.50 | 1.4 | + | hinge |
| 2 | 159 | 1.78 | 0 | 0.81 | 0.19 | 0.0231 | - | constant |
| 3 | 84 | 0.995 | 0 | 0.67 | 0.33 | 0.0503 | - | hinge |
| 4 | -93 | -2.95 | 0 | 0.01 | 0.99 | 0.252 | + | line |
| 5 | -49 | 0.216 | 0 | 0.55 | 0.45 | 0.592 | + | hinge |
| 6 | -49 | -0.131 | 0 | 0.47 | 0.53 | 1.05 | + | hinge |
| 7 | -159 | 0.00021 | 0 | 0.50 | 0.50 | 1.59 | + | hinge |
| 8 | -39 | -2.02 | 0 | 0.14 | 0.86 | 0.0499 | + | line |
| 9 | -26 | 0.254 | 0 | 0.55 | 0.45 | 0.454 | + | hinge |
| 10 | -115 | -0.234 | 0 | 0.46 | 0.54 | 0.976 | + | hinge |
| 11 | -116 | 0.181 | 0 | 0.53 | 0.47 | 0.98 | + | hinge |
| 12 | -163 | 6.32 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 13 | 42 | 0.00232 | 0 | 0.50 | 0.50 | 1.33 | + | hinge |
| 14 | 155 | 0.0951 | 0 | 0.52 | 0.48 | 1.21 | + | hinge |
| 15 | 89 | 0.0166 | 0 | 0.51 | 0.49 | 1.14 | + | hinge |

## Groups

- negligible: units 12 (magnitude 0 together; a quarter of the tolerance is 0.0207)
- lines: units 2, 4, 8 (each a constant or a line on the box; together one linear term)
- the rest: units 0, 1, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15
