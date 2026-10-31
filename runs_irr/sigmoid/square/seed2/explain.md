# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (7 nodes):

```python
def net(x0):
    return x0 * x0
```

## Faithful fit

The square formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no square formula follows).

Faithful:

```python
def net(x0):
    return 0.991 * (x0 + 0.0004321) * (x0 + 0.0004321) + 0.02014
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, inside the box (x0 = 0.048)
  - mean error in 10 equal slices of x0: +0.3 -0.0 -0.3 -0.4 -0.5 -0.5 -0.4 -0.3 -0.1 +0.2
  - largest error per slice: +0.4 +0.2 +0.3 +0.5 +0.5 +0.5 +0.5 +0.4 +0.2 +0.4
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: -0.3 +0.2 +0.2 +0.5 +0.5 +0.4 +0.5 +0.2 +0.2 -0.3
  - largest error per slice: +0.6 +0.3 +0.3 +0.6 +0.6 +0.5 +0.6 +0.4 +0.2 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.991 | 0.91% |
| offset | 0 | 0.02014 | 0.51% |
| constant inside square | 0 | 0.0004321 | 0.04% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0793, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.177 | 4.97 | 0.00 | 0.08 | 0.92 | 0.00 | 0.00 | 0.312 | + | rounded hinge |
| 1 | 180 | 1.87 | 1.48 | 0.60 | 0.23 | 0.17 | 0.00 | 0.00 | 2.89 | + | none (best 0.289) |
| 2 | 0 | 1.9 | 1.11 | 0.70 | 0.17 | 0.13 | 0.00 | 0.00 | 2.07 | + | none (best 0.233) |
| 3 | 0 | 0.697 | 1.81 | 0.22 | 0.28 | 0.34 | 0.15 | 0.00 | 0.89 | + | none (best 0.0637) |
| 4 | 0 | -0.818 | 1.21 | 0.00 | 0.18 | 0.22 | 0.19 | 0.40 | 0.984 | - | none (best 0.116) |
| 5 | 0 | -0.745 | 2.16 | 0.00 | 0.11 | 0.40 | 0.34 | 0.15 | 0.144 | - | clamp |
| 6 | 0 | 1.1 | 1.37 | 0.43 | 0.21 | 0.26 | 0.10 | 0.00 | 1.67 | + | none (best 0.12) |
| 7 | 180 | -0.329 | 7.19 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.292 | + | rounded hinge |

## Groups

- same line: units 4, 5 (gate angle 0 degrees, offset -0.818)
- the rest: units 0, 1, 2, 3, 6, 7
