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
    return 0.9913 * (x0 + 0.0006945) * (x0 + 0.0006945) + 0.02011
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, inside the box (x0 = 0.08)
  - mean error in 10 equal slices of x0: +0.3 -0.0 -0.2 -0.4 -0.5 -0.5 -0.4 -0.3 -0.1 +0.1
  - largest error per slice: +0.4 +0.1 +0.3 +0.5 +0.5 +0.5 +0.5 +0.4 +0.2 +0.3
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: -0.4 +0.1 +0.2 +0.5 +0.5 +0.5 +0.5 +0.2 +0.2 -0.3
  - largest error per slice: +0.6 +0.2 +0.3 +0.6 +0.6 +0.6 +0.6 +0.3 +0.3 +0.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.9913 | 0.88% |
| offset | 0 | 0.02011 | 0.51% |
| constant inside square | 0 | 0.0006945 | 0.07% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0794, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.825 | 1.33 | 0.37 | 0.21 | 0.25 | 0.17 | 0.00 | 0.804 | + | none (best 0.0811) |
| 1 | 0 | -1.84 | 9.91 | 0.00 | 0.00 | 0.97 | 0.03 | 0.00 | 0.235 | + | rounded hinge |
| 2 | 180 | 0.975 | 1.75 | 0.31 | 0.28 | 0.33 | 0.09 | 0.00 | 1.05 | + | none (best 0.0644) |
| 3 | 0 | 0.174 | 4.11 | 0.00 | 0.16 | 0.77 | 0.07 | 0.00 | 0.532 | + | none (best 0.0212) |
| 4 | 180 | 1.9 | 1.57 | 0.58 | 0.24 | 0.18 | 0.00 | 0.00 | 1.49 | + | none (best 0.156) |
| 5 | 180 | 1.96 | 1.39 | 0.64 | 0.21 | 0.14 | 0.00 | 0.00 | 1.3 | + | none (best 0.13) |
| 6 | 0 | 1.87 | 1.39 | 0.62 | 0.22 | 0.16 | 0.00 | 0.00 | 2.72 | + | none (best 0.284) |
| 7 | 180 | -0.871 | 1.22 | 0.00 | 0.17 | 0.23 | 0.19 | 0.41 | 1.14 | - | none (best 0.124) |

## Groups

- same line: units 0, 2 (gate angle 180 degrees, offset 0.825)
- same line: units 1, 4, 5 (gate angle 0 degrees, offset -1.84)
- the rest: units 3, 6, 7
