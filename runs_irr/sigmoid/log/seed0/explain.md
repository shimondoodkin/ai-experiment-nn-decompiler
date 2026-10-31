# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return log(x0) + 0.02
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.029 * log(x0 + 0.01509) - 0.0272
```

Clean:

```python
def net(x0):
    return log(x0) + 0.02
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, inside the box (x0 = 0.521)
  - mean error in 10 equal slices of x0: +0.7 +0.9 +0.7 +0.6 +0.4 +0.3 +0.2 +0.1 +0.0 -0.0
  - largest error per slice: +1.0 +1.0 +0.8 +0.6 +0.5 +0.4 +0.3 +0.2 +0.1 +0.1
- ripple (faithful minus net): largest 0.7%, at the box edge (x0 = 0.169)
  - mean error in 10 equal slices of x0: -0.2 -0.3 -0.2 -0.1 +0.1 +0.2 +0.3 +0.3 +0.4 +0.6
  - largest error per slice: +0.7 +0.5 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3 +0.4 +0.7

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.01509 | 3.67% |
| scale | 1 | 1.029 | 1.74% |
| offset | 0.02 | -0.0272 | 1.23% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0766, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 2.09 | 8.39 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.214 | + | line |
| 1 | 0 | 0.46 | 2.89 | 0.00 | 0.00 | 0.29 | 0.37 | 0.34 | 0.363 | + | none (best 0.0318) |
| 2 | 180 | 0.0277 | 0.557 | 0.91 | 0.07 | 0.02 | 0.00 | 0.00 | 0.5 | - | none (best 0.0848) |
| 3 | 180 | 0.0179 | 0.526 | 0.92 | 0.07 | 0.02 | 0.00 | 0.00 | 0.491 | - | none (best 0.0946) |
| 4 | 180 | -0.349 | 1.92 | 0.56 | 0.24 | 0.20 | 0.00 | 0.00 | 0.779 | - | none (best 0.0771) |
| 5 | 0 | 3.07 | 7.67 | 0.00 | 0.02 | 0.98 | 0.00 | 0.00 | 0.488 | + | rounded hinge |
| 6 | 180 | -1.48 | 5.94 | 0.00 | 0.26 | 0.74 | 0.00 | 0.00 | 0.692 | - | none (best 0.0374) |
| 7 | 0 | 0.0159 | 1.27 | 0.00 | 0.00 | 0.08 | 0.16 | 0.76 | 0.303 | + | none (best 0.0359) |

## Groups

- lines: units 0 (each a constant or a line on the box; together one linear term)
- same line: units 1, 4 (gate angle 0 degrees, offset 0.46)
- same line: units 2, 3, 7 (gate angle 180 degrees, offset 0.0277)
- the rest: units 5, 6
