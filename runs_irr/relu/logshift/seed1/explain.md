# Explain

Inputs: x0 in [-1.5, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return log(x0 + 2)
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.013 * log(x0 + 2.014) - 0.009834
```

Clean:

```python
def net(x0):
    return log(x0 + 2)
```

% of the output range:
- distortion (clean minus faithful): largest 0.6%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: -0.2 -0.2 -0.2 -0.3 -0.4 -0.4 -0.5 -0.5 -0.6 -0.6
  - largest error per slice: +0.4 +0.2 +0.3 +0.3 +0.4 +0.5 +0.5 +0.6 +0.6 +0.6
- ripple (faithful minus net): largest 0.8%, inside the box (x0 = -1.11)
  - mean error in 10 equal slices of x0: +0.1 +0.2 +0.4 +0.1 +0.4 +0.5 +0.3 +0.5 +0.8 +0.6
  - largest error per slice: +0.8 +0.7 +0.5 +0.4 +0.5 +0.6 +0.5 +0.7 +0.8 +0.7

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 2 | 2.014 | 1.12% |
| scale | 1 | 1.013 | 0.95% |
| offset | 0 | -0.009834 | 0.40% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0493, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -0.489 | 0 | 0.19 | 0.81 | 0.768 | + | hinge |
| 1 | 180 | 0.497 | 0 | 0.82 | 0.18 | 0.693 | - | hinge |
| 2 | 180 | -27.2 | 0 | 0.00 | 1.00 | 0.0819 | + | line |
| 3 | 0 | -0.489 | 0 | 0.19 | 0.81 | 0.879 | + | hinge |
| 4 | 180 | -0.326 | 0 | 0.67 | 0.33 | 0.32 | - | hinge |
| 5 | 0 | 2.24 | 0 | 0.68 | 0.32 | 0.145 | - | hinge |
| 6 | 180 | 1.11 | 0 | 0.93 | 0.07 | 0.239 | - | hinge |
| 7 | 0 | 1.19 | 0 | 0.49 | 0.51 | 0.208 | - | hinge |

## Groups

- lines: units 2 (each a constant or a line on the box; together one linear term)
- same line: units 0, 1, 3 (gate angle 0 degrees, offset -0.489)
- the rest: units 4, 5, 6, 7
