# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return log(x0) + 0.04
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.039 * log(x0 + 0.02188) - 0.03928
```

Clean:

```python
def net(x0):
    return log(x0) + 0.04
```

% of the output range:
- distortion (clean minus faithful): largest 1.6%, inside the box (x0 = 0.561)
  - mean error in 10 equal slices of x0: +1.2 +1.6 +1.4 +1.2 +1.0 +0.8 +0.7 +0.6 +0.5 +0.4
  - largest error per slice: +1.6 +1.6 +1.5 +1.2 +1.1 +0.9 +0.8 +0.6 +0.5 +0.4
- ripple (faithful minus net): largest 0.9%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: -0.2 -0.4 -0.2 -0.1 +0.1 +0.3 +0.4 +0.5 +0.6 +0.8
  - largest error per slice: +0.9 +0.8 +0.2 +0.2 +0.3 +0.4 +0.4 +0.5 +0.6 +0.9

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.02188 | 5.22% |
| scale | 1 | 1.039 | 2.37% |
| offset | 0.04 | -0.03928 | 2.09% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0758, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 2.19 | 5.76 | 0.00 | 0.00 | 0.87 | 0.13 | 0.00 | 0.435 | + | rounded hinge |
| 1 | 180 | 0.0093 | 0.601 | 0.90 | 0.08 | 0.02 | 0.00 | 0.00 | 1.02 | - | none (best 0.125) |
| 2 | 0 | 1.83 | 8.77 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0318 | + | constant |
| 3 | 0 | 2.31 | 7.08 | 0.00 | 0.00 | 0.99 | 0.01 | 0.00 | 0.468 | + | rounded hinge |
| 4 | 180 | -2.18 | 8.23 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.382 | - | rounded hinge |
| 5 | 0 | 0.97 | 3.27 | 0.00 | 0.00 | 0.43 | 0.42 | 0.15 | 0.206 | + | rounded hinge |
| 6 | 0 | 0.831 | 1.79 | 0.00 | 0.01 | 0.27 | 0.23 | 0.48 | 0.626 | + | none (best 0.0437) |
| 7 | 180 | 0.038 | 1.39 | 0.74 | 0.18 | 0.08 | 0.00 | 0.00 | 0.627 | - | none (best 0.0761) |

## Groups

- lines: units 2 (each a constant or a line on the box; together one linear term)
- same line: units 0, 3, 4 (gate angle 0 degrees, offset 2.19)
- same line: units 1, 7 (gate angle 180 degrees, offset 0.0093)
- same line: units 5, 6 (gate angle 0 degrees, offset 0.97)
