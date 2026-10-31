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
    return 0.9941 * (x0 - 0.0008044) * (x0 - 0.0008044) + 0.0126
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.1 -0.1 -0.2 -0.3 -0.3 -0.3 -0.2 -0.1 +0.0 +0.2
  - largest error per slice: +0.2 +0.2 +0.3 +0.3 +0.3 +0.3 +0.3 +0.2 +0.1 +0.4
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = -1.82)
  - mean error in 10 equal slices of x0: -0.2 +0.1 +0.1 +0.3 +0.3 +0.3 +0.2 +0.2 +0.1 -0.2
  - largest error per slice: +0.4 +0.2 +0.1 +0.3 +0.3 +0.3 +0.2 +0.2 +0.1 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.9941 | 0.59% |
| offset | 0 | 0.0126 | 0.32% |
| constant inside square | 0 | -0.0008044 | 0.08% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0796, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1.19 | 2.09 | 0.28 | 0.33 | 0.39 | 0.01 | 0.00 | 2.91 | + | none (best 0.196) |
| 1 | 180 | -91.4 | 122 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0258 | + | constant |
| 2 | 0 | 80.7 | 96.4 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.021 | - | constant |
| 3 | 180 | 1.99 | 1.4 | 0.65 | 0.21 | 0.14 | 0.00 | 0.00 | 2.2 | + | none (best 0.237) |
| 4 | 0 | 187 | 214 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00399 | - | constant |
| 5 | 0 | 2.03 | 1.33 | 0.67 | 0.21 | 0.12 | 0.00 | 0.00 | 1.71 | + | none (best 0.183) |
| 6 | 0 | 64.4 | 86.8 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0198 | - | constant |
| 7 | 0 | -1 | 1.86 | 0.00 | 0.08 | 0.35 | 0.29 | 0.28 | 2.36 | - | none (best 0.152) |

## Groups

- negligible: units 4, 6 (magnitude 0.0238 together; a quarter of the tolerance is 0.0199)
- lines: units 1, 2 (each a constant or a line on the box; together one linear term)
- the rest: units 0, 3, 5, 7
