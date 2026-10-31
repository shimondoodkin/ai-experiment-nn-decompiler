# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (10 nodes):

```python
def net(x0):
    return x0 * x0 - 0.04
```

## Faithful fit

The square formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no square formula follows).

Faithful:

```python
def net(x0):
    return 0.9815 * (x0 - 0.002783) * (x0 - 0.002783) + 0.02745
```

Clean:

```python
def net(x0):
    return x0 * x0 - 0.04
```

% of the output range:
- distortion (clean minus faithful): largest 1.7%, inside the box (x0 = -0.148)
  - mean error in 10 equal slices of x0: -0.4 -1.0 -1.4 -1.6 -1.7 -1.7 -1.5 -1.1 -0.6 +0.1
  - largest error per slice: +0.7 +1.2 +1.5 +1.7 +1.7 +1.7 +1.6 +1.3 +0.9 +0.4
- ripple (faithful minus net): largest 1.2%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: -0.6 -0.4 +1.0 +1.0 +0.4 +0.4 +1.0 +1.0 -0.4 -0.6
  - largest error per slice: +1.2 +1.1 +1.2 +1.2 +0.7 +0.6 +1.2 +1.2 +1.1 +1.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.9815 | 1.89% |
| offset | -0.04 | 0.02745 | 1.73% |
| constant inside square | 0 | -0.002783 | 0.28% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0782, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -42.6 | 64.2 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0403 | + | rounded hinge |
| 1 | 180 | 58.1 | 69.2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0151 | - | constant |
| 2 | 180 | 18.8 | 47.4 | 0.00 | 0.76 | 0.24 | 0.00 | 0.00 | 0.0726 | - | rounded hinge |
| 3 | 0 | -30.4 | 57.4 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.106 | + | rounded hinge |
| 4 | 180 | 40.7 | 63.1 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0676 | - | rounded hinge |
| 5 | 0 | -69 | 71.3 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00746 | + | constant |
| 6 | 0 | -1.84 | 2.42 | 0.00 | 0.00 | 0.27 | 0.38 | 0.36 | 4.74 | - | none (best 0.449) |
| 7 | 180 | -1.82 | 2.31 | 0.00 | 0.00 | 0.26 | 0.36 | 0.37 | 4.4 | - | none (best 0.393) |

## Groups

- negligible: units 1, 5 (magnitude 0.0225 together; a quarter of the tolerance is 0.0195)
- the rest: units 0, 2, 3, 4, 6, 7
