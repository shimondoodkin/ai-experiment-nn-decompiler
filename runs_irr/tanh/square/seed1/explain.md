# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (10 nodes):

```python
def net(x0):
    return x0 * x0 - 0.02
```

## Faithful fit

The square formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no square formula follows).

Faithful:

```python
def net(x0):
    return 0.9826 * (x0 - 0.0009228) * (x0 - 0.0009228) + 0.03436
```

Clean:

```python
def net(x0):
    return x0 * x0 - 0.02
```

% of the output range:
- distortion (clean minus faithful): largest 1.4%, inside the box (x0 = -0.052)
  - mean error in 10 equal slices of x0: -0.0 -0.6 -1.0 -1.3 -1.4 -1.4 -1.2 -0.9 -0.4 +0.1
  - largest error per slice: +0.3 +0.8 +1.1 +1.3 +1.4 +1.4 +1.3 +1.1 +0.7 +0.5
- ripple (faithful minus net): largest 1.2%, inside the box (x0 = 0.816)
  - mean error in 10 equal slices of x0: -0.6 -0.4 +1.0 +1.0 +0.5 +0.5 +1.0 +1.0 -0.4 -0.6
  - largest error per slice: +1.2 +1.1 +1.2 +1.2 +0.7 +0.7 +1.2 +1.2 +1.1 +1.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.9826 | 1.78% |
| offset | -0.02 | 0.03436 | 1.39% |
| constant inside square | 0 | -0.0009228 | 0.09% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0782, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -42.5 | 65 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0449 | + | rounded hinge |
| 1 | 180 | 44.3 | 65.9 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0408 | - | rounded hinge |
| 2 | 0 | -28.1 | 56.6 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0762 | + | rounded hinge |
| 3 | 180 | 1.84 | 2.43 | 0.35 | 0.38 | 0.27 | 0.00 | 0.00 | 4.79 | + | none (best 0.456) |
| 4 | 180 | -1.81 | 2.29 | 0.00 | 0.00 | 0.26 | 0.36 | 0.38 | 4.38 | - | none (best 0.389) |
| 5 | 180 | 40.8 | 63.9 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0451 | - | rounded hinge |
| 6 | 180 | 29.4 | 57.6 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0795 | - | rounded hinge |
| 7 | 180 | 25.9 | 54.8 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.105 | - | rounded hinge |

## Groups

- the rest: units 0, 1, 2, 3, 4, 5, 6, 7
