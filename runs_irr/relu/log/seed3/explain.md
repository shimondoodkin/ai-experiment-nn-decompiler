# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (12 nodes):

```python
def net(x0):
    return 0.98 * log(x0) - 0.004005
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.03 * log(x0 + 0.01193) + 0.007058
```

Clean:

```python
def net(x0):
    return 0.98 * log(x0) - 0.004005
```

% of the output range:
- distortion (clean minus faithful): largest 2.5%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: +0.1 -0.4 -0.9 -1.2 -1.5 -1.7 -1.9 -2.1 -2.3 -2.4
  - largest error per slice: +0.3 +0.7 +1.1 +1.4 +1.6 +1.9 +2.0 +2.2 +2.3 +2.5
- ripple (faithful minus net): largest 1.7%, at the box edge (x0 = 0.181)
  - mean error in 10 equal slices of x0: +0.6 +0.3 +0.6 +0.7 +1.3 +0.7 +1.2 +1.6 +1.6 +1.2
  - largest error per slice: +1.7 +1.2 +1.2 +1.3 +1.4 +1.1 +1.5 +1.7 +1.7 +1.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 0.98 | 1.03 | 3.02% |
| constant inside log | 0 | 0.01193 | 2.89% |
| offset | -0.004005 | 0.007058 | 0.29% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0764, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -1.54 | 0 | 0.71 | 0.29 | 0.654 | - | hinge |
| 1 | 180 | -0.724 | 0 | 0.87 | 0.13 | 0.703 | - | hinge |
| 2 | 180 | 0.229 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | -0.294 | 0 | 0.96 | 0.04 | 0.732 | - | hinge |
| 4 | 180 | 12.3 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | 0.224 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | 0 | 0.3 | 0 | 0.04 | 0.96 | 2.15 | + | hinge |
| 7 | 0 | 2.89 | 0 | 0.57 | 0.43 | 0.422 | - | hinge |

## Groups

- negligible: units 2, 4, 5 (magnitude 0 together; a quarter of the tolerance is 0.0191)
- same line: units 3, 6 (gate angle 180 degrees, offset -0.294)
- the rest: units 0, 1, 7
