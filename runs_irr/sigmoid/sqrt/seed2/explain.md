# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (12 nodes):

```python
def net(x0):
    return sqrt(x0 + 0.02) - 0.01
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.029 * sqrt(x0 + 0.03745) - 0.05841
```

Clean:

```python
def net(x0):
    return sqrt(x0 + 0.02) - 0.01
```

% of the output range:
- distortion (clean minus faithful): largest 0.9%, inside the box (x0 = 0.276)
  - mean error in 10 equal slices of x0: +0.7 +0.8 +0.6 +0.3 +0.1 -0.0 -0.2 -0.4 -0.6 -0.7
  - largest error per slice: +0.9 +0.9 +0.7 +0.5 +0.2 +0.1 +0.3 +0.5 +0.6 +0.8
- ripple (faithful minus net): largest 1.2%, inside the box (x0 = 0.714)
  - mean error in 10 equal slices of x0: +0.5 -1.0 -0.9 -0.2 +0.1 +0.0 -0.1 -0.1 +0.1 +0.7
  - largest error per slice: +1.2 +1.2 +1.2 +0.5 +0.1 +0.1 +0.2 +0.2 +0.3 +1.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.029 | 3.18% |
| constant inside sqrt | 0.02 | 0.03745 | 2.85% |
| offset | -0.01 | -0.05841 | 2.65% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0365, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1.01 | 6.47 | 0.00 | 0.00 | 0.86 | 0.14 | 0.00 | 0.288 | + | none (best 0.013) |
| 1 | 180 | 0.299 | 1.27 | 0.76 | 0.20 | 0.04 | 0.00 | 0.00 | 0.388 | - | none (best 0.051) |
| 2 | 0 | 1.58 | 7.94 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.116 | + | line |
| 3 | 0 | 0.603 | 7.13 | 0.00 | 0.00 | 0.82 | 0.18 | 0.00 | 0.335 | + | none (best 0.0161) |
| 4 | 180 | 1.89 | 21 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0327 | - | line |
| 5 | 0 | -0.253 | 1.31 | 0.00 | 0.00 | 0.06 | 0.20 | 0.74 | 0.126 | + | none (best 0.0165) |
| 6 | 0 | 1.57 | 5.39 | 0.00 | 0.00 | 0.90 | 0.10 | 0.00 | 0.312 | + | none (best 0.0117) |
| 7 | 180 | -1.69 | 9.2 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.228 | - | rounded hinge |

## Groups

- lines: units 2, 4 (each a constant or a line on the box; together one linear term)
- same line: units 1, 5 (gate angle 180 degrees, offset 0.299)
- same line: units 6, 7 (gate angle 0 degrees, offset 1.57)
- the rest: units 0, 3
