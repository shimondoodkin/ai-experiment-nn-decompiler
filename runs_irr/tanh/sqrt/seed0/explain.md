# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return sqrt(x0 + 0.01)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.021 * sqrt(x0 + 0.01782) - 0.03748
```

Clean:

```python
def net(x0):
    return sqrt(x0 + 0.01)
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, at the box edge (x0 = 0.174)
  - mean error in 10 equal slices of x0: +0.9 +0.8 +0.7 +0.5 +0.3 +0.2 +0.1 -0.1 -0.2 -0.3
  - largest error per slice: +1.0 +0.9 +0.8 +0.6 +0.4 +0.3 +0.1 +0.1 +0.2 +0.3
- ripple (faithful minus net): largest 0.7%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: -0.2 -0.5 -0.3 -0.2 -0.1 +0.1 +0.2 +0.3 +0.4 +0.6
  - largest error per slice: +0.7 +0.7 +0.4 +0.2 +0.1 +0.2 +0.3 +0.4 +0.5 +0.7

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.021 | 2.23% |
| offset | 0 | -0.03748 | 1.99% |
| constant inside sqrt | 0.01 | 0.01782 | 1.78% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0377, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.385 | 13.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.118 | + | rounded hinge |
| 1 | 0 | -0.0373 | 3.24 | 0.00 | 0.00 | 0.29 | 0.51 | 0.20 | 0.334 | + | none (best 0.0316) |
| 2 | 180 | 0.143 | 0.531 | 0.90 | 0.09 | 0.01 | 0.00 | 0.00 | 0.25 | - | none (best 0.0324) |
| 3 | 180 | 0.637 | 3.15 | 0.37 | 0.49 | 0.14 | 0.00 | 0.00 | 0.156 | - | none (best 0.0164) |
| 4 | 180 | 4.7 | 11.9 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00319 | + | constant |
| 5 | 0 | 2.69 | 7.8 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.796 | + | none (best 0.0244) |
| 6 | 180 | 2.3 | 11.6 | 0.00 | 0.49 | 0.51 | 0.00 | 0.00 | 0.0794 | - | rounded hinge |
| 7 | 0 | -0.651 | 2.06 | 0.00 | 0.00 | 0.03 | 0.32 | 0.65 | 0.154 | + | none (best 0.0183) |

## Groups

- negligible: units 4 (magnitude 0.00319 together; a quarter of the tolerance is 0.00942)
- same line: units 1, 2 (gate angle 0 degrees, offset -0.0373)
- same line: units 3, 7 (gate angle 180 degrees, offset 0.637)
- the rest: units 0, 5, 6
