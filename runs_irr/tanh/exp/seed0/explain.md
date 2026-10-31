# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return exp(x0)
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0):
    return 1.035 * exp(0.9835 * x0) - 0.03391
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.3%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.3 +0.3 +0.2 +0.1 +0.0 -0.1 -0.2 -0.2 -0.2 +0.1
  - largest error per slice: +0.3 +0.3 +0.2 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2 +0.3
- ripple (faithful minus net): largest 0.3%, at the box edge (x0 = 1.85)
  - mean error in 10 equal slices of x0: -0.3 -0.3 -0.2 -0.1 +0.0 +0.1 +0.1 +0.2 +0.2 -0.2
  - largest error per slice: +0.3 +0.3 +0.3 +0.2 +0.1 +0.1 +0.1 +0.2 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.035 | 3.58% |
| x0 inside exp | 1 | 0.9835 | 3.32% |
| offset | 0 | -0.03391 | 0.47% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 2.08 | 1.19 | 0.72 | 0.19 | 0.09 | 0.00 | 0.00 | 2.53 | + | none (best 0.3) |
| 1 | 0 | -23.2 | 35.1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0764 | + | rounded hinge |
| 2 | 180 | 25 | 35.4 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0651 | - | constant |
| 3 | 180 | 23.9 | 35.5 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0693 | - | constant |
| 4 | 180 | 35.4 | 38.4 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0157 | - | constant |
| 5 | 0 | 0.977 | 3.13 | 0.00 | 0.45 | 0.55 | 0.00 | 0.00 | 2.07 | + | none (best 0.136) |
| 6 | 180 | 26.2 | 36.1 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0529 | - | constant |
| 7 | 0 | 1.45 | 2.03 | 0.35 | 0.32 | 0.33 | 0.00 | 0.00 | 2.34 | + | none (best 0.172) |

## Groups

- negligible: units 4 (magnitude 0.0157 together; a quarter of the tolerance is 0.0361)
- lines: units 2, 3, 6 (each a constant or a line on the box; together one linear term)
- the rest: units 0, 1, 5, 7
