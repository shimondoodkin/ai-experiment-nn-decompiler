# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (7 nodes):

```python
def net(x0):
    return x0 * x0
```

## Faithful fit

The square formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no square formula follows).

Faithful:

```python
def net(x0):
    return 0.9901 * (x0 + 0.0005455) * (x0 + 0.0005455) + 0.02295
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.6%, inside the box (x0 = 0.054)
  - mean error in 10 equal slices of x0: +0.3 -0.0 -0.3 -0.5 -0.6 -0.6 -0.5 -0.4 -0.1 +0.2
  - largest error per slice: +0.5 +0.2 +0.4 +0.5 +0.6 +0.6 +0.6 +0.4 +0.3 +0.4
- ripple (faithful minus net): largest 0.7%, inside the box (x0 = -1.79)
  - mean error in 10 equal slices of x0: -0.4 +0.3 +0.1 +0.4 +0.6 +0.3 +0.4 +0.4 -0.1 -0.5
  - largest error per slice: +0.7 +0.4 +0.3 +0.7 +0.7 +0.5 +0.5 +0.5 +0.5 +0.7

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.9901 | 1.00% |
| offset | 0 | 0.02295 | 0.58% |
| constant inside square | 0 | 0.0005455 | 0.06% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0792, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.779 | 4.73 | 0.00 | 0.00 | 0.75 | 0.25 | 0.00 | 0.168 | - | rounded hinge |
| 1 | 180 | 0.768 | 1.36 | 0.35 | 0.21 | 0.25 | 0.18 | 0.00 | 1.54 | + | none (best 0.159) |
| 2 | 0 | 0.415 | 4.89 | 0.00 | 0.15 | 0.85 | 0.00 | 0.00 | 0.202 | + | rounded hinge |
| 3 | 180 | 0.598 | 13 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0304 | - | constant |
| 4 | 180 | 1.91 | 1.38 | 0.63 | 0.21 | 0.15 | 0.00 | 0.00 | 1.42 | + | none (best 0.147) |
| 5 | 0 | 1.27 | 1.86 | 0.35 | 0.29 | 0.35 | 0.01 | 0.00 | 2.51 | + | none (best 0.175) |
| 6 | 180 | 1.76 | 1.39 | 0.59 | 0.21 | 0.19 | 0.00 | 0.00 | 1.52 | + | none (best 0.141) |
| 7 | 0 | 1.99 | 1.28 | 0.68 | 0.20 | 0.12 | 0.00 | 0.00 | 1.58 | + | none (best 0.162) |

## Groups

- lines: units 3 (each a constant or a line on the box; together one linear term)
- same line: units 4, 6 (gate angle 180 degrees, offset 1.91)
- the rest: units 0, 1, 2, 5, 7
