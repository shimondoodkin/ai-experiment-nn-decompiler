# Explain

Inputs: x0 in [-1, 3]; output: regression.

## Formula

A family program passes the check (12 nodes):

```python
def net(x0):
    return 17.4 / (x0 + 3.5) - 2
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 18.56 / (x0 + 3.597) - 2.178
```

Clean:

```python
def net(x0):
    return 17.4 / (x0 + 3.5) - 2
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, at the box edge (x0 = 3)
  - mean error in 10 equal slices of x0: -0.3 -0.3 -0.2 -0.1 +0.1 +0.3 +0.4 +0.6 +0.8 +0.9
  - largest error per slice: +0.4 +0.4 +0.3 +0.2 +0.2 +0.3 +0.5 +0.7 +0.8 +1.0
- ripple (faithful minus net): largest 1.1%, at the box edge (x0 = -1)
  - mean error in 10 equal slices of x0: -0.3 -0.9 -0.3 +0.2 +0.7 +1.0 +1.1 +0.9 +0.4 -0.5
  - largest error per slice: +1.1 +1.0 +0.6 +0.5 +0.8 +1.1 +1.1 +1.0 +0.7 +1.0

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 17.4 | 18.56 | 10.94% |
| constant inside reciprocal | 3.5 | 3.597 | 6.13% |
| offset | -2 | -2.178 | 4.20% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0848, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.484 | 4.47 | 0.00 | 0.21 | 0.79 | 0.00 | 0.00 | 0.62 | + | none (best 0.0288) |
| 1 | 180 | -0.947 | 4.88 | 0.00 | 0.06 | 0.92 | 0.03 | 0.00 | 0.552 | + | rounded hinge |
| 2 | 180 | 1.02 | 1.93 | 0.52 | 0.30 | 0.18 | 0.00 | 0.00 | 0.341 | + | none (best 0.0347) |
| 3 | 180 | 1.05 | 1.46 | 0.65 | 0.22 | 0.13 | 0.00 | 0.00 | 0.393 | + | none (best 0.0442) |
| 4 | 180 | 0.878 | 2.38 | 0.37 | 0.37 | 0.25 | 0.00 | 0.00 | 0.649 | + | none (best 0.0601) |
| 5 | 180 | 0.0991 | 4.52 | 0.00 | 0.35 | 0.65 | 0.00 | 0.00 | 0.665 | + | none (best 0.0406) |
| 6 | 180 | 0.556 | 3.06 | 0.12 | 0.48 | 0.40 | 0.00 | 0.00 | 0.499 | + | none (best 0.0397) |
| 7 | 180 | -0.727 | 5.11 | 0.00 | 0.09 | 0.91 | 0.00 | 0.00 | 0.521 | + | rounded hinge |

## Groups

- same line: units 2, 3, 4 (gate angle 180 degrees, offset 1.02)
- the rest: units 0, 1, 5, 6, 7
