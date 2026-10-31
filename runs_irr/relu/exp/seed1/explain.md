# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return exp(x0)
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0):
    return 1.011 * exp(0.9951 * x0) - 0.03264
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.4 +0.4 +0.4 +0.3 +0.3 +0.3 +0.3 +0.2 +0.2 +0.3
  - largest error per slice: +0.4 +0.4 +0.4 +0.4 +0.3 +0.3 +0.3 +0.2 +0.2 +0.3
- ripple (faithful minus net): largest 0.7%, at the box edge (x0 = 1.83)
  - mean error in 10 equal slices of x0: -0.4 -0.3 -0.5 -0.1 -0.4 -0.1 -0.2 -0.3 -0.2 -0.2
  - largest error per slice: +0.7 +0.5 +0.5 +0.3 +0.5 +0.5 +0.5 +0.6 +0.6 +0.7

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.011 | 1.13% |
| x0 inside exp | 1 | 0.9951 | 1.01% |
| offset | 0 | -0.03264 | 0.46% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.143, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.158 | 0 | 0.54 | 0.46 | 1.39 | + | hinge |
| 1 | 180 | -2.02 | 0 | 0.00 | 1.00 | 0.194 | - | line |
| 2 | 0 | -0.571 | 0 | 0.36 | 0.64 | 1.22 | + | hinge |
| 3 | 0 | 0.738 | 0 | 0.69 | 0.31 | 1.4 | + | hinge |
| 4 | 0 | -1.58 | 0 | 0.11 | 0.89 | 1.08 | + | hinge |
| 5 | 0 | 1.64 | 0 | 0.91 | 0.09 | 0.705 | + | hinge |
| 6 | 180 | 4.49 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 7 | 0 | 1.23 | 0 | 0.81 | 0.19 | 1.18 | + | hinge |

## Groups

- negligible: units 6 (magnitude 0 together; a quarter of the tolerance is 0.0358)
- lines: units 1 (each a constant or a line on the box; together one linear term)
- the rest: units 0, 2, 3, 4, 5, 7
