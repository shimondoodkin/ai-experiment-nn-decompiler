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
    return 0.9987 * (x0 + 0.001832) * (x0 + 0.001832) - 0.01703
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.8%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.7 +0.6 +0.6 +0.5 +0.5 +0.4 +0.4 +0.4 +0.4 +0.4
  - largest error per slice: +0.8 +0.7 +0.6 +0.5 +0.5 +0.4 +0.4 +0.4 +0.4 +0.4
- ripple (faithful minus net): largest 1.1%, inside the box (x0 = 0.836)
  - mean error in 10 equal slices of x0: -0.5 -0.1 -0.4 -0.7 -0.2 -0.1 -0.6 -0.4 -0.1 -0.5
  - largest error per slice: +1.1 +1.1 +1.1 +1.1 +0.9 +1.0 +1.0 +1.1 +1.1 +1.1

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| offset | 0 | -0.01703 | 0.43% |
| constant inside square | 0 | 0.001832 | 0.19% |
| scale | 1 | 0.9987 | 0.13% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0785, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.836 | 0 | 0.71 | 0.29 | 1.36 | + | hinge |
| 1 | 180 | 3.72 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 0 | 1.43 | 0 | 0.86 | 0.14 | 0.666 | + | hinge |
| 3 | 180 | 0.843 | 0 | 0.71 | 0.29 | 1.33 | + | hinge |
| 4 | 0 | 3.02 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 5 | 0 | -0.276 | 0 | 0.43 | 0.57 | 2.5 | + | hinge |
| 6 | 180 | 1.42 | 0 | 0.85 | 0.15 | 0.669 | + | hinge |
| 7 | 180 | -0.27 | 0 | 0.43 | 0.57 | 2.51 | + | hinge |

## Groups

- negligible: units 1, 4 (magnitude 0 together; a quarter of the tolerance is 0.0196)
- the rest: units 0, 2, 3, 5, 6, 7
