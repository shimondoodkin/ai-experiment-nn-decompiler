# Explain

Inputs: x0 in [0.5, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return 1 / x0 - 0.02
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 1.065 / (x0 + 0.03161) - 0.02445
```

Clean:

```python
def net(x0):
    return 1 / x0 - 0.02
```

% of the output range:
- distortion (clean minus faithful): largest 1.6%, inside the box (x0 = 0.988)
  - mean error in 10 equal slices of x0: -1.1 -1.6 -1.5 -1.3 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6
  - largest error per slice: +1.6 +1.6 +1.6 +1.4 +1.2 +1.0 +0.9 +0.8 +0.7 +0.6
- ripple (faithful minus net): largest 0.9%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: -0.2 +0.7 +0.3 -0.1 -0.1 -0.1 -0.2 -0.3 -0.5 -0.7
  - largest error per slice: +0.9 +0.8 +0.6 +0.1 +0.2 +0.1 +0.2 +0.4 +0.6 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.065 | 7.60% |
| constant inside reciprocal | 0 | 0.03161 | 6.95% |
| offset | -0.02 | -0.02445 | 0.26% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0342, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.663 | 1.15 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0254 | - | rounded hinge |
| 1 | 180 | -0.347 | 1.12 | 0.72 | 0.20 | 0.08 | 0.00 | 0.00 | 0.53 | + | none (best 0.0633) |
| 2 | 180 | -2.25 | 9.15 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.16 | + | line |
| 3 | 180 | -0.343 | 0.725 | 0.84 | 0.13 | 0.03 | 0.00 | 0.00 | 0.433 | + | none (best 0.0566) |
| 4 | 180 | 8.88 | 12.5 | 0.11 | 0.89 | 0.00 | 0.00 | 0.00 | 0.0323 | - | rounded hinge |
| 5 | 180 | 0.938 | 1.27 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0158 | - | constant |
| 6 | 0 | 0.764 | 2.61 | 0.00 | 0.00 | 0.35 | 0.47 | 0.18 | 0.241 | - | none (best 0.0202) |
| 7 | 0 | 0.816 | 3.12 | 0.00 | 0.00 | 0.42 | 0.56 | 0.02 | 0.419 | - | none (best 0.0323) |

## Groups

- lines: units 2, 5 (each a constant or a line on the box; together one linear term)
- same line: units 1, 3 (gate angle 180 degrees, offset -0.347)
- same line: units 6, 7 (gate angle 0 degrees, offset 0.764)
- the rest: units 0, 4
