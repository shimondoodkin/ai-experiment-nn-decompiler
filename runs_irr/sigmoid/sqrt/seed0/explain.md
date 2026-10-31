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
    return 1.029 * sqrt(x0 + 0.02666) - 0.04945
```

Clean:

```python
def net(x0):
    return sqrt(x0 + 0.01)
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: +0.8 +0.9 +0.6 +0.4 +0.2 +0.0 -0.1 -0.3 -0.5 -0.6
  - largest error per slice: +1.0 +0.9 +0.7 +0.5 +0.3 +0.1 +0.2 +0.4 +0.5 +0.7
- ripple (faithful minus net): largest 1.0%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: +0.1 -0.8 -0.4 -0.1 +0.0 +0.1 +0.2 +0.3 +0.5 +0.8
  - largest error per slice: +0.9 +0.9 +0.6 +0.2 +0.1 +0.1 +0.2 +0.4 +0.6 +1.0

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.01 | 0.02666 | 3.40% |
| scale | 1 | 1.029 | 3.12% |
| offset | 0 | -0.04945 | 2.66% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0372, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.684 | 10.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.149 | + | rounded hinge |
| 1 | 0 | 0.0245 | 5.03 | 0.00 | 0.00 | 0.48 | 0.52 | 0.00 | 0.215 | + | none (best 0.0159) |
| 2 | 180 | 0.248 | 0.872 | 0.84 | 0.14 | 0.02 | 0.00 | 0.00 | 0.373 | - | none (best 0.0484) |
| 3 | 180 | -1.84 | 10.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.304 | - | line |
| 4 | 0 | 2.91 | 9.57 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.193 | + | rounded hinge |
| 5 | 0 | 2.08 | 9.36 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.266 | + | line |
| 6 | 0 | 2.21 | 10.6 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0585 | + | line |
| 7 | 0 | 0.166 | 2.87 | 0.00 | 0.00 | 0.31 | 0.45 | 0.24 | 0.303 | + | none (best 0.0265) |

## Groups

- lines: units 3, 5, 6 (each a constant or a line on the box; together one linear term)
- same line: units 1, 7 (gate angle 0 degrees, offset 0.0245)
- the rest: units 0, 2, 4
