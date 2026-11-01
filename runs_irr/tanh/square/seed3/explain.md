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
    return 0.9886 * (x0 + 0.0006941) * (x0 + 0.0006941) + 0.02127
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.5 +0.1 -0.2 -0.4 -0.5 -0.5 -0.4 -0.3 -0.0 +0.3
  - largest error per slice: +0.7 +0.3 +0.3 +0.5 +0.5 +0.5 +0.5 +0.4 +0.2 +0.5
- ripple (faithful minus net): largest 0.8%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: -0.4 -0.1 +0.6 +0.4 +0.3 +0.7 +0.5 +0.1 +0.2 -0.5
  - largest error per slice: +0.8 +0.7 +0.7 +0.6 +0.5 +0.8 +0.8 +0.2 +0.3 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.9886 | 1.15% |
| offset | 0 | 0.02127 | 0.54% |
| constant inside square | 0 | 0.0006941 | 0.07% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0793, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -1.89 | 1.47 | 0.00 | 0.00 | 0.16 | 0.23 | 0.61 | 3.16 | - | none (best 0.305) |
| 1 | 180 | -12.2 | 16.7 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.137 | + | rounded hinge |
| 2 | 180 | 1.9 | 2.21 | 0.42 | 0.35 | 0.23 | 0.00 | 0.00 | 3.98 | + | none (best 0.394) |
| 3 | 180 | -11.8 | 16 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.167 | + | rounded hinge |
| 4 | 0 | 11.3 | 15.9 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0729 | - | rounded hinge |
| 5 | 0 | 9.9 | 15.2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.198 | - | rounded hinge |
| 6 | 0 | 0.783 | 1.5 | 0.32 | 0.24 | 0.28 | 0.16 | 0.00 | 1.98 | + | none (best 0.17) |
| 7 | 0 | 7.12 | 13.6 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.557 | - | none (best 0.033) |

## Groups

- the rest: units 0, 1, 2, 3, 4, 5, 6, 7
