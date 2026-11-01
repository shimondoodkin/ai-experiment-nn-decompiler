# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (21 nodes):

```python
def net(x0, x1):
    return 1 / (-1.104e-05 * x0 * x1 + 0.003356) - 298
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0, x1):
    return 1 / (2.845e-06 * x0 * x0 + 3.957e-06 * x1 * x1 - 0.0001511 * x0 * x1 + 1.439e-06 * x0 + 4.054e-07 * x1 + 0.01238) - 80.72
```

Clean:

```python
def net(x0, x1):
    return 1 / (-1.104e-05 * x0 * x1 + 0.003356) - 298
```

% of the output range:
- distortion (clean minus faithful): largest 1.1%, at the box edge (x0 = -1.99, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -1.1 -1.0 -0.9 -0.8 -0.6 -0.4 -0.3 -0.2 -0.3 -0.4; largest: +1.1 +1.1 +1.1 +1.0 +0.9 +0.9 +0.8 +0.8 +0.9 +1.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.4 -0.4 -0.3 -0.4 -0.5 -0.7 -0.8 -0.8 -0.5 -0.4 -0.3 -0.4; largest: +1.1 +1.1 +1.1 +1.1 +1.1 +1.1 +1.1 +1.1 +1.1 +1.1 +1.1 +1.1
- ripple (faithful minus net): largest 1.9%, inside the box (x0 = 1.64, x1 = -1.21)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.8 +0.7 +0.6 +0.7 +0.3 +0.1 +0.2 +0.3 +0.4 +0.2; largest: +1.6 +1.7 +1.8 +1.7 +1.5 +1.6 +1.9 +1.9 +1.7 +1.9
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.3 -0.0 +0.1 +0.3 +0.8 +0.5 +0.3 +0.1 -0.1 +0.3 +0.8 +0.4; largest: +1.3 +1.9 +1.5 +1.6 +1.8 +1.4 +1.7 +1.7 +1.2 +1.2 +1.9 +1.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside reciprocal | 0.003356 | 0.01238 | 2868.36% |
| offset | -298 | -80.72 | 2821.58% |
| x0·x1 inside reciprocal | -1.104e-05 | -0.0001511 | 798.35% |
| x1² inside reciprocal | 0 | 3.957e-06 | 18.65% |
| x0² inside reciprocal | 0 | 2.845e-06 | 13.43% |
| x0 inside reciprocal | 0 | 1.439e-06 | 3.41% |
| x1 inside reciprocal | 0 | 4.054e-07 | 0.96% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.153, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -139 | 0.392 | 0 | 0.63 | 0.37 | 0.825 | + | hinge |
| 1 | -142 | 1.62 | 0 | 0.91 | 0.09 | 0.576 | + | hinge |
| 2 | -140 | 2.19 | 0 | 0.97 | 0.03 | 0.406 | + | hinge |
| 3 | 40 | 1.59 | 0 | 0.91 | 0.09 | 0.949 | + | hinge |
| 4 | 49 | -0.923 | 0 | 0.23 | 0.77 | 1.97 | + | hinge |
| 5 | 79 | 2.87 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | -138 | -0.961 | 0 | 0.22 | 0.78 | 1.62 | + | hinge |
| 7 | -172 | -1.65 | 0 | 0.08 | 0.92 | 0.818 | + | hinge |
| 8 | 139 | 0.659 | 0 | 0.71 | 0.29 | 1.9 | - | hinge |
| 9 | 42 | 0.0691 | 0 | 0.53 | 0.47 | 1.69 | + | hinge |
| 10 | -49 | 1.19 | 0 | 0.83 | 0.17 | 1.4 | - | hinge |
| 11 | -40 | 0.219 | 0 | 0.57 | 0.43 | 2.45 | - | hinge |
| 12 | -143 | -0.584 | 0 | 0.32 | 0.68 | 1.41 | + | hinge |
| 13 | 164 | -1.46 | 0 | 0.12 | 0.88 | 2.08 | - | hinge |
| 14 | -154 | 1.01 | 0 | 0.78 | 0.22 | 0.49 | + | hinge |
| 15 | 140 | 1.56 | 0 | 0.90 | 0.10 | 1.21 | - | hinge |

## Groups

- negligible: units 5 (magnitude 0 together; a quarter of the tolerance is 0.0383)
- the rest: units 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
