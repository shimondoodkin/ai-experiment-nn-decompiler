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
    return 1.034 * log(x0 + 0.01801) - 0.03267
```

Clean:

```python
def net(x0):
    return log(x0) + 0.04
```

% of the output range:
- distortion (clean minus faithful): largest 1.6%, inside the box (x0 = 0.529)
  - mean error in 10 equal slices of x0: +1.2 +1.5 +1.3 +1.1 +1.0 +0.8 +0.7 +0.6 +0.5 +0.4
  - largest error per slice: +1.6 +1.6 +1.4 +1.2 +1.0 +0.9 +0.8 +0.6 +0.5 +0.5
- ripple (faithful minus net): largest 0.9%, at the box edge (x0 = 0.171)
  - mean error in 10 equal slices of x0: -0.3 -0.3 -0.3 -0.1 +0.1 +0.3 +0.4 +0.4 +0.4 +0.7
  - largest error per slice: +0.9 +0.6 +0.3 +0.3 +0.2 +0.4 +0.4 +0.4 +0.5 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.01801 | 4.34% |
| scale | 1 | 1.034 | 2.05% |
| offset | 0.04 | -0.03267 | 1.91% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0763, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.252 | 1.37 | 0.00 | 0.00 | 0.14 | 0.17 | 0.69 | 0.37 | + | none (best 0.0366) |
| 1 | 180 | -2.66 | 6.36 | 0.00 | 0.00 | 0.97 | 0.03 | 0.00 | 0.879 | - | none (best 0.0267) |
| 2 | 0 | 0.3 | 8.31 | 0.00 | 0.00 | 0.68 | 0.32 | 0.00 | 0.209 | + | rounded hinge |
| 3 | 0 | 0.346 | 2.15 | 0.00 | 0.00 | 0.22 | 0.28 | 0.51 | 0.363 | + | none (best 0.0347) |
| 4 | 180 | -0.0497 | 0.5 | 0.91 | 0.07 | 0.03 | 0.00 | 0.00 | 1.02 | - | none (best 0.172) |
| 5 | 0 | 1.61 | 6.54 | 0.00 | 0.00 | 0.81 | 0.19 | 0.00 | 0.119 | + | rounded hinge |
| 6 | 180 | -0.35 | 2.32 | 0.47 | 0.30 | 0.23 | 0.00 | 0.00 | 0.678 | - | none (best 0.0662) |
| 7 | 0 | 1.41 | 5.13 | 0.00 | 0.00 | 0.66 | 0.34 | 0.00 | 0.173 | + | rounded hinge |

## Groups

- same line: units 0, 2, 3, 4, 6 (gate angle 0 degrees, offset 0.252)
- same line: units 5, 7 (gate angle 0 degrees, offset 1.61)
- the rest: units 1
