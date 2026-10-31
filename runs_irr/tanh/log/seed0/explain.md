# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return log(x0)
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.026 * log(x0 + 0.0137) - 0.03
```

Clean:

```python
def net(x0):
    return log(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 1.1%, at the box edge (x0 = 0.1)
  - mean error in 10 equal slices of x0: +0.3 +0.5 +0.3 +0.2 +0.1 -0.0 -0.1 -0.2 -0.3 -0.4
  - largest error per slice: +1.1 +0.5 +0.4 +0.2 +0.1 +0.1 +0.2 +0.3 +0.3 +0.4
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = 0.166)
  - mean error in 10 equal slices of x0: -0.2 -0.2 -0.2 -0.1 +0.1 +0.2 +0.3 +0.3 +0.3 +0.5
  - largest error per slice: +0.6 +0.5 +0.2 +0.1 +0.2 +0.3 +0.3 +0.3 +0.4 +0.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.0137 | 3.35% |
| scale | 1 | 1.026 | 1.56% |
| offset | 0 | -0.03 | 0.78% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0767, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 2.8 | 6.18 | 0.00 | 0.08 | 0.92 | 0.00 | 0.00 | 1.18 | + | none (best 0.038) |
| 1 | 0 | 0.335 | 1.35 | 0.00 | 0.00 | 0.15 | 0.17 | 0.68 | 0.603 | + | none (best 0.0584) |
| 2 | 180 | 0.0467 | 0.522 | 0.92 | 0.07 | 0.01 | 0.00 | 0.00 | 0.385 | - | none (best 0.0717) |
| 3 | 180 | 0.0356 | 0.497 | 0.93 | 0.06 | 0.01 | 0.00 | 0.00 | 0.391 | - | none (best 0.067) |
| 4 | 180 | 10.2 | 10.1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.025 | + | constant |
| 5 | 0 | 0.7 | 2.65 | 0.00 | 0.00 | 0.32 | 0.34 | 0.34 | 1 | + | none (best 0.0775) |
| 6 | 180 | 6.25 | 10.2 | 0.21 | 0.79 | 0.00 | 0.00 | 0.00 | 0.0876 | + | rounded hinge |
| 7 | 0 | -0.0972 | 0.823 | 0.00 | 0.00 | 0.02 | 0.11 | 0.87 | 0.388 | + | none (best 0.051) |

## Groups

- lines: units 4 (each a constant or a line on the box; together one linear term)
- same line: units 2, 3, 7 (gate angle 180 degrees, offset 0.0467)
- the rest: units 0, 1, 5, 6
