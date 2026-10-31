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
    return 0.9926 * (x0 + 0.0002791) * (x0 + 0.0002791) + 0.01816
```

Clean:

```python
def net(x0):
    return x0 * x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, inside the box (x0 = 0.038)
  - mean error in 10 equal slices of x0: +0.2 -0.1 -0.3 -0.4 -0.4 -0.4 -0.4 -0.3 -0.1 +0.1
  - largest error per slice: +0.3 +0.2 +0.3 +0.4 +0.5 +0.5 +0.4 +0.3 +0.2 +0.3
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: -0.3 +0.1 +0.2 +0.4 +0.4 +0.4 +0.4 +0.2 +0.1 -0.2
  - largest error per slice: +0.4 +0.2 +0.3 +0.4 +0.4 +0.5 +0.4 +0.3 +0.3 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 0.9926 | 0.74% |
| offset | 0 | 0.01816 | 0.46% |
| constant inside square | 0 | 0.0002791 | 0.03% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0795, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.821 | 1.41 | 0.35 | 0.22 | 0.27 | 0.16 | 0.00 | 0.913 | + | none (best 0.0814) |
| 1 | 0 | 0.664 | 7.68 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.084 | + | line |
| 2 | 180 | -0.963 | 2.65 | 0.00 | 0.01 | 0.49 | 0.41 | 0.08 | 0.314 | - | none (best 0.0205) |
| 3 | 180 | 0.898 | 1.65 | 0.31 | 0.26 | 0.31 | 0.12 | 0.00 | 1.65 | + | none (best 0.114) |
| 4 | 180 | 1.92 | 1.75 | 0.54 | 0.27 | 0.18 | 0.00 | 0.00 | 1.64 | + | none (best 0.177) |
| 5 | 0 | 1.98 | 1.48 | 0.62 | 0.23 | 0.14 | 0.00 | 0.00 | 1.94 | + | none (best 0.216) |
| 6 | 180 | 2.01 | 1.43 | 0.65 | 0.22 | 0.13 | 0.00 | 0.00 | 1.18 | + | none (best 0.126) |
| 7 | 0 | 1.66 | 1.99 | 0.42 | 0.31 | 0.27 | 0.00 | 0.00 | 1.22 | + | none (best 0.108) |

## Groups

- lines: units 1 (each a constant or a line on the box; together one linear term)
- same line: units 0, 2 (gate angle 0 degrees, offset 0.821)
- same line: units 4, 6 (gate angle 180 degrees, offset 1.92)
- the rest: units 3, 5, 7
