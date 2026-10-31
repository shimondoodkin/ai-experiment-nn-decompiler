# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (18 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.005 * x0 * x0 + 1.001 * x1 * x1 - 0.001227 * x0 * x1 - 0.005697 * x0 - 0.01118 * x1 + 0.009579) - 0.01381
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

% of the output range:
- distortion (clean minus faithful): largest 1.8%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0, 0), rings 0.423 wide, mean: -0.1 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2; largest: +1.8 +0.3 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4 +0.4
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.3 +0.3 +0.4 +0.3 +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2; largest: +1.4 +1.6 +1.1 +1.8 +1.3 +1.2 +1.3 +1.7 +1.1 +1.1 +1.0 +1.6
- ripple (faithful minus net): largest 0.6%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0.00284, 0.00559), rings 0.424 wide, mean: -0.0 -0.1 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.1; largest: +0.6 +0.4 +0.5 +0.5 +0.5 +0.5 +0.6 +0.6 +0.6 +0.6
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.3 -0.3 -0.2 -0.2 -0.1 -0.1 -0.2 -0.2 -0.1 -0.2 -0.1; largest: +0.5 +0.5 +0.5 +0.6 +0.6 +0.4 +0.5 +0.6 +0.6 +0.5 +0.6 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.009579 | 1.82% |
| offset | 0 | -0.01381 | 0.33% |
| x0² inside sqrt | 1 | 1.005 | 0.18% |
| x1 inside sqrt | 0 | -0.01118 | 0.15% |
| x0 inside sqrt | 0 | -0.005697 | 0.07% |
| x1² inside sqrt | 1 | 1.001 | 0.04% |
| x0·x1 inside sqrt | 0 | -0.001227 | 0.03% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0834, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -35 | -0.00404 | 0 | 0.50 | 0.50 | 0.973 | + | hinge |
| 1 | -2 | -0.0738 | 0 | 0.49 | 0.51 | 0.234 | + | hinge |
| 2 | 8 | 0.00412 | 0 | 0.50 | 0.50 | 0.648 | + | hinge |
| 3 | -85 | 0.0878 | 0 | 0.51 | 0.49 | 0.782 | + | hinge |
| 4 | -66 | 0.00342 | 0 | 0.50 | 0.50 | 1.09 | + | hinge |
| 5 | -2 | 0.0414 | 0 | 0.51 | 0.49 | 0.449 | + | hinge |
| 6 | 61 | 0.259 | 0 | 0.55 | 0.45 | 0.324 | + | hinge |
| 7 | 77 | 0.0149 | 0 | 0.50 | 0.50 | 0.95 | + | hinge |
| 8 | -50 | 0.256 | 0 | 0.55 | 0.45 | 0.233 | + | hinge |
| 9 | 95 | 0.218 | 0 | 0.54 | 0.46 | 0.407 | + | hinge |
| 10 | 162 | 0.00069 | 0 | 0.50 | 0.50 | 1.28 | + | hinge |
| 11 | 60 | -0.0612 | 0 | 0.49 | 0.51 | 1.03 | + | hinge |
| 12 | 130 | 0.0577 | 0 | 0.51 | 0.49 | 0.979 | + | hinge |
| 13 | -103 | -0.771 | 0 | 0.37 | 0.63 | 0.101 | - | hinge |
| 14 | -137 | 0.000611 | 0 | 0.50 | 0.50 | 1.24 | + | hinge |
| 15 | -156 | -0.000257 | 0 | 0.50 | 0.50 | 1.32 | + | hinge |

## Groups

- same line: units 1, 5 (gate angle -2 degrees, offset -0.0738)
- the rest: units 0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
