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
    return 1.043 * exp(0.9803 * x0) - 0.04412
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.4 +0.4 +0.3 +0.2 +0.1 -0.0 -0.2 -0.2 -0.2 +0.1
  - largest error per slice: +0.5 +0.4 +0.3 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2 +0.3
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = 1.81)
  - mean error in 10 equal slices of x0: -0.3 -0.4 -0.3 -0.1 +0.0 +0.1 +0.1 +0.3 +0.3 -0.2
  - largest error per slice: +0.4 +0.4 +0.4 +0.2 +0.1 +0.1 +0.2 +0.4 +0.4 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.043 | 4.40% |
| x0 inside exp | 1 | 0.9803 | 3.96% |
| offset | 0 | -0.04412 | 0.61% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -1.85 | 2.03 | 0.00 | 0.00 | 0.23 | 0.32 | 0.46 | 1.47 | - | none (best 0.136) |
| 1 | 0 | -17.2 | 19.8 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.103 | + | rounded hinge |
| 2 | 180 | -0.636 | 2.33 | 0.00 | 0.12 | 0.44 | 0.36 | 0.08 | 1.81 | - | none (best 0.104) |
| 3 | 180 | -1.91 | 2 | 0.00 | 0.00 | 0.21 | 0.31 | 0.48 | 1.51 | - | none (best 0.15) |
| 4 | 180 | 18 | 20.3 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0494 | - | constant |
| 5 | 180 | 16.3 | 20.4 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0755 | - | rounded hinge |
| 6 | 0 | 2.11 | 1.32 | 0.70 | 0.21 | 0.09 | 0.00 | 0.00 | 1.99 | + | none (best 0.25) |
| 7 | 180 | 11.3 | 17.6 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.213 | - | rounded hinge |

## Groups

- lines: units 4 (each a constant or a line on the box; together one linear term)
- same line: units 0, 3 (gate angle 180 degrees, offset -1.85)
- the rest: units 1, 2, 5, 6, 7
