# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (13 nodes):

```python
def net(x0):
    return 1.002 * x0 * x0 - 0.01697
```

## Faithful fit

The square formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no square formula follows).

Faithful:

```python
def net(x0):
    return 1.002 * (x0 - 0.001989) * (x0 - 0.001989) - 0.01697
```

Clean:

```python
def net(x0):
    return 1.002 * x0 * x0 - 0.01697
```

% of the output range:
- distortion (clean minus faithful): largest 0.2%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: -0.2 -0.1 -0.1 -0.1 -0.0 +0.0 +0.1 +0.1 +0.1 +0.2
  - largest error per slice: +0.2 +0.2 +0.1 +0.1 +0.0 +0.0 +0.1 +0.1 +0.2 +0.2
- ripple (faithful minus net): largest 1.8%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: -1.1 -0.4 -1.2 -0.4 -0.3 -1.0 -0.1 -0.8 -0.3 -0.6
  - largest error per slice: +1.8 +1.6 +1.5 +1.4 +1.4 +1.4 +1.2 +1.4 +1.8 +1.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside square | 0 | -0.001989 | 0.20% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0805, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -1.24 | 0 | 0.19 | 0.81 | 5.8 | - | hinge |
| 1 | 180 | -0.567 | 0 | 0.36 | 0.64 | 3.36 | + | hinge |
| 2 | 180 | 0.714 | 0 | 0.68 | 0.32 | 1.64 | + | hinge |
| 3 | 180 | 0.0769 | 0 | 0.52 | 0.48 | 2.47 | + | hinge |
| 4 | 180 | 12.3 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | 1.35 | 0 | 0.84 | 0.16 | 0.836 | + | hinge |
| 6 | 0 | 1.24 | 0 | 0.81 | 0.19 | 2.46 | + | hinge |
| 7 | 0 | 2.58 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 4, 7 (magnitude 0 together; a quarter of the tolerance is 0.0201)
- same line: units 0, 6 (gate angle 180 degrees, offset -1.24)
- the rest: units 1, 2, 3, 5
