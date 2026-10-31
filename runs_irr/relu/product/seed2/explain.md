# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (19 nodes):

```python
def net(x0, x1):
    return 1188 * exp(0.0008335 * x0 * x1) - 1188
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1188 * exp(-2.14e-06 * x0 * x0 + 1.003e-06 * x1 * x1 + 0.0008342 * x0 * x1 - 7.317e-07 * x0 - 3.249e-08 * x1) - 1188
```

Clean:

```python
def net(x0, x1):
    return 1188 * exp(0.0008335 * x0 * x1) - 1188
```

% of the output range:
- distortion (clean minus faithful): largest 0.2%, at the box edge (x0 = 2, x1 = -0.68)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.1; largest: +0.0 +0.0 +0.0 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2 +0.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 +0.0 -0.0 -0.0 +0.0 +0.1 +0.0 -0.0 -0.0 -0.0 +0.1 +0.1; largest: +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2
- ripple (faithful minus net): largest 1.5%, at the box edge (x0 = -2, x1 = 1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.0 -0.0 -0.0 +0.0 +0.0 -0.1 -0.0 +0.1 +0.0 -0.0; largest: +0.6 +0.8 +0.9 +0.8 +1.0 +1.1 +1.1 +1.2 +1.2 +1.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.2 -0.0 +0.1 +0.3 +0.1 -0.1 -0.1 -0.1 +0.1 +0.1 -0.1; largest: +1.2 +1.3 +0.8 +0.9 +1.5 +1.1 +1.0 +1.0 +1.0 +0.7 +1.4 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0² inside exp | 0 | -2.14e-06 | 0.13% |
| x1² inside exp | 0 | 1.003e-06 | 0.06% |
| x0·x1 inside exp | 0.0008335 | 0.0008342 | 0.04% |
| x0 inside exp | 0 | -7.317e-07 | 0.02% |
| x1 inside exp | 0 | -3.249e-08 | 0.00% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.154, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -40 | 1.83 | 0 | 0.94 | 0.06 | 0.73 | - | hinge |
| 1 | -42 | -0.835 | 0 | 0.25 | 0.75 | 2.53 | - | hinge |
| 2 | 41 | -0.293 | 0 | 0.41 | 0.59 | 1.52 | + | hinge |
| 3 | -138 | 1.88 | 0 | 0.94 | 0.06 | 0.681 | + | hinge |
| 4 | -112 | -1.82 | 0 | 0.05 | 0.95 | 1.31 | + | hinge |
| 5 | 41 | -1.23 | 0 | 0.16 | 0.84 | 2.24 | + | hinge |
| 6 | 139 | -0.463 | 0 | 0.35 | 0.65 | 2.21 | - | hinge |
| 7 | 35 | 1.72 | 0 | 0.93 | 0.07 | 0.671 | + | hinge |
| 8 | -42 | -0.177 | 0 | 0.44 | 0.56 | 1.93 | - | hinge |
| 9 | 129 | 1.6 | 0 | 0.90 | 0.10 | 0.759 | - | hinge |
| 10 | 139 | -1.13 | 0 | 0.18 | 0.82 | 2.72 | - | hinge |
| 11 | 41 | -0.748 | 0 | 0.27 | 0.73 | 1.6 | + | hinge |
| 12 | 29 | 0.993 | 0 | 0.78 | 0.22 | 0.52 | + | hinge |
| 13 | -21 | -1.69 | 0 | 0.07 | 0.93 | 1.76 | - | hinge |
| 14 | -132 | -0.915 | 0 | 0.23 | 0.77 | 1.98 | + | hinge |
| 15 | -138 | -0.226 | 0 | 0.42 | 0.58 | 1.78 | + | hinge |

## Groups

- the rest: units 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
