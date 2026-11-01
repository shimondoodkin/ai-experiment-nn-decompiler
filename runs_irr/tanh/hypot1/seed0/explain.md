# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (34 nodes):

```python
def net(x0, x1):
    return -2.95 * exp(-0.1316 * x0 * x0 - 0.1282 * x1 * x1) + 4
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return -2.878 * exp(-0.136 * x0 * x0 - 0.1322 * x1 * x1 - 0.0016 * x0 * x1 - 0.00148 * x0 - 0.001783 * x1) + 3.93
```

Clean:

```python
def net(x0, x1):
    return -2.95 * exp(-0.1316 * x0 * x0 - 0.1282 * x1 * x1) + 4
```

% of the output range:
- distortion (clean minus faithful): largest 0.9%, at the box edge (x0 = 2, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.1 -0.1 -0.2 -0.2 -0.2 -0.2 -0.1 -0.0 +0.2 +0.4; largest: +0.2 +0.4 +0.5 +0.7 +0.8 +0.8 +0.8 +0.8 +0.7 +0.9
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.6 -0.6 -0.5 -0.2 +0.1 +0.0 +0.0 +0.1 +0.1 +0.2 +0.2 -0.2; largest: +0.8 +0.8 +0.8 +0.5 +0.9 +0.4 +0.2 +0.6 +0.3 +0.5 +0.9 +0.5
- ripple (faithful minus net): largest 1.5%, inside the box (x0 = -0.554, x1 = -1.59)
  - by distance from (-0.0054, -0.00671), rings 0.282 wide, mean: +1.1 +0.9 +0.6 +0.3 +0.1 +0.0 +0.1 -0.0 -0.1 +0.2; largest: +1.5 +1.5 +1.4 +1.0 +1.1 +1.5 +1.5 +1.5 +1.5 +1.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.6 +0.3 +0.3 +0.2 -0.2 +0.4 +0.5 +0.0 -0.4 +0.1 -0.2 +0.7; largest: +1.4 +1.5 +1.5 +1.4 +1.2 +1.5 +1.4 +1.5 +1.5 +1.5 +1.5 +1.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | -2.95 | -2.878 | 3.74% |
| offset | 4 | 3.93 | 3.64% |
| x0² inside exp | -0.1316 | -0.136 | 1.58% |
| x1² inside exp | -0.1282 | -0.1322 | 1.46% |
| x0·x1 inside exp | 0 | -0.0016 | 0.35% |
| x1 inside exp | 0 | -0.001783 | 0.33% |
| x0 inside exp | 0 | -0.00148 | 0.27% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0384, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -47 | 15.7 | 31.1 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0907 | - | rounded hinge |
| 1 | -47 | 17.9 | 32.2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0794 | - | rounded hinge |
| 2 | 147 | 0.927 | 2.73 | 0.06 | 0.41 | 0.48 | 0.05 | 0.00 | 1.44 | + | none (best 0.092) |
| 3 | 127 | 2.22 | 3.06 | 0.25 | 0.56 | 0.19 | 0.00 | 0.00 | 1.22 | + | none (best 0.102) |
| 4 | -50 | 7.76 | 24.7 | 0.00 | 0.11 | 0.89 | 0.00 | 0.00 | 0.0916 | - | rounded hinge |
| 5 | 134 | -20.4 | 33.4 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0496 | + | rounded hinge |
| 6 | 132 | -13.4 | 29.6 | 0.00 | 0.00 | 0.02 | 0.98 | 0.00 | 0.135 | + | rounded hinge |
| 7 | -165 | -0.413 | 2.02 | 0.00 | 0.20 | 0.40 | 0.32 | 0.09 | 1.04 | - | none (best 0.123) |
| 8 | 132 | -13.3 | 29.5 | 0.00 | 0.00 | 0.02 | 0.98 | 0.00 | 0.174 | + | rounded hinge |
| 9 | -48 | 13.4 | 29.7 | 0.00 | 0.98 | 0.02 | 0.00 | 0.00 | 0.138 | - | rounded hinge |
| 10 | 66 | -1.28 | 2.92 | 0.00 | 0.01 | 0.45 | 0.47 | 0.08 | 1.59 | - | none (best 0.102) |
| 11 | -46 | 19.6 | 33.1 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0654 | - | rounded hinge |
| 12 | -49 | 8.52 | 25.5 | 0.00 | 0.20 | 0.80 | 0.00 | 0.00 | 0.0792 | - | rounded hinge |
| 13 | -77 | 1.36 | 2.76 | 0.14 | 0.44 | 0.42 | 0.00 | 0.00 | 1.48 | + | none (best 0.0985) |
| 14 | -161 | -1.87 | 2.45 | 0.00 | 0.00 | 0.25 | 0.41 | 0.35 | 1.84 | - | none (best 0.139) |
| 15 | 133 | -16.5 | 31.5 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0888 | + | rounded hinge |

## Groups

- same line: units 6, 8, 9 (gate angle 132 degrees, offset -13.4)
- the rest: units 0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15
