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
    return 1.017 * log(x0 + 2.024) - 0.02908
```

Clean:

```python
def net(x0):
    return log(x0 + 2)
```

% of the output range:
- distortion (clean minus faithful): largest 0.3%, at the box edge (x0 = -1.5)
  - mean error in 10 equal slices of x0: +0.1 +0.2 +0.2 +0.2 +0.1 +0.0 -0.0 -0.1 -0.1 -0.2
  - largest error per slice: +0.3 +0.2 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2
- ripple (faithful minus net): largest 0.3%, at the box edge (x0 = -1.5)
  - mean error in 10 equal slices of x0: +0.0 -0.2 -0.1 -0.1 -0.0 +0.0 +0.1 +0.2 +0.3 +0.3
  - largest error per slice: +0.3 +0.3 +0.2 +0.1 +0.1 +0.1 +0.2 +0.2 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 2 | 2.024 | 1.90% |
| scale | 1 | 1.017 | 1.23% |
| offset | 0 | -0.02908 | 1.18% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0494, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.922 | 14.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.165 | + | line |
| 1 | 0 | -0.871 | 4.11 | 0.00 | 0.00 | 0.39 | 0.47 | 0.14 | 0.359 | + | none (best 0.0281) |
| 2 | 180 | 1.66 | 1.92 | 0.68 | 0.21 | 0.10 | 0.00 | 0.00 | 0.411 | - | none (best 0.0458) |
| 3 | 180 | 0.963 | 3.02 | 0.35 | 0.34 | 0.30 | 0.00 | 0.00 | 0.281 | - | none (best 0.0234) |
| 4 | 180 | 1.62 | 0.928 | 0.85 | 0.11 | 0.04 | 0.00 | 0.00 | 0.284 | - | none (best 0.0381) |
| 5 | 0 | 4.12 | 10.7 | 0.00 | 0.29 | 0.71 | 0.00 | 0.00 | 0.399 | + | none (best 0.0212) |
| 6 | 180 | 1.02 | 6.99 | 0.00 | 0.43 | 0.57 | 0.00 | 0.00 | 0.155 | - | rounded hinge |
| 7 | 0 | -0.0523 | 6.78 | 0.00 | 0.00 | 0.73 | 0.27 | 0.00 | 0.415 | + | none (best 0.0229) |

## Groups

- lines: units 0 (each a constant or a line on the box; together one linear term)
- same line: units 1, 3, 6 (gate angle 0 degrees, offset -0.871)
- same line: units 2, 4 (gate angle 180 degrees, offset 1.66)
- the rest: units 5, 7
