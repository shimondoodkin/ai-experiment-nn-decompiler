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
    return 0.9993 * (x0 - 0.000408) * (x0 - 0.000408) - 0.007699
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.3%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3
  - largest error per slice: +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.3 +0.3
- ripple (faithful minus net): largest 0.7%, inside the box (x0 = -1.1)
  - mean error in 10 equal slices of x0: -0.3 -0.2 -0.1 -0.1 -0.2 -0.1 -0.2 -0.2 -0.2 -0.3
  - largest error per slice: +0.7 +0.7 +0.7 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| offset | 0 | -0.007699 | 0.19% |
| scale | 1 | 0.9993 | 0.07% |
| constant inside square | 0 | -0.000408 | 0.04% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.079, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.221 | 0 | 0.56 | 0.44 | 1.58 | + | hinge |
| 1 | 180 | 1.1 | 0 | 0.78 | 0.22 | 0.808 | + | hinge |
| 2 | 180 | 0.218 | 0 | 0.56 | 0.44 | 1.56 | + | hinge |
| 3 | 0 | 0.669 | 0 | 0.67 | 0.33 | 1.19 | + | hinge |
| 4 | 180 | 0.659 | 0 | 0.67 | 0.33 | 1.19 | + | hinge |
| 5 | 0 | 1.11 | 0 | 0.78 | 0.22 | 0.785 | + | hinge |
| 6 | 180 | 1.55 | 0 | 0.89 | 0.11 | 0.399 | + | hinge |
| 7 | 0 | 1.56 | 0 | 0.89 | 0.11 | 0.393 | + | hinge |

## Groups

- the rest: units 0, 1, 2, 3, 4, 5, 6, 7
