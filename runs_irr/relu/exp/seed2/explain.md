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
    return 1.023 * exp(0.9913 * x0) - 0.05997
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.8%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.7 +0.7 +0.7 +0.6 +0.5 +0.5 +0.4 +0.3 +0.2 +0.2
  - largest error per slice: +0.8 +0.7 +0.7 +0.6 +0.6 +0.5 +0.4 +0.3 +0.3 +0.3
- ripple (faithful minus net): largest 1.0%, inside the box (x0 = 1.56)
  - mean error in 10 equal slices of x0: -0.4 -0.9 -0.8 -0.1 -0.8 -0.3 -0.6 -0.1 -0.3 -0.4
  - largest error per slice: +0.7 +1.0 +1.0 +0.5 +0.9 +0.9 +0.9 +0.8 +1.0 +1.0

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.023 | 2.36% |
| x0 inside exp | 1 | 0.9913 | 1.77% |
| offset | 0 | -0.05997 | 0.83% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.351 | 0 | 0.59 | 0.41 | 1.8 | + | hinge |
| 1 | 180 | 3.72 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 0 | 1.56 | 0 | 0.89 | 0.11 | 1.01 | + | hinge |
| 3 | 0 | 1.03 | 0 | 0.76 | 0.24 | 1.61 | + | hinge |
| 4 | 0 | 3.03 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 5 | 0 | -0.56 | 0 | 0.36 | 0.64 | 1.61 | + | hinge |
| 6 | 180 | -2 | 0 | 0.00 | 1.00 | 0.395 | - | line |
| 7 | 0 | -2.08 | 0 | 0.00 | 1.00 | 0.785 | + | line |

## Groups

- negligible: units 1, 4 (magnitude 0 together; a quarter of the tolerance is 0.0361)
- lines: units 6, 7 (each a constant or a line on the box; together one linear term)
- the rest: units 0, 2, 3, 5
