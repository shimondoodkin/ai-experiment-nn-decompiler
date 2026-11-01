# Explain

Inputs: x0 in [0.5, 4]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return 1 / x0
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 1.044 / (x0 + 0.02162) - 0.01633
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.9%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.1 -0.3 -0.2 -0.1 +0.0 +0.1 +0.2 +0.3 +0.3 +0.4
  - largest error per slice: +0.9 +0.3 +0.3 +0.2 +0.1 +0.2 +0.2 +0.3 +0.3 +0.4
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: -0.1 +0.5 +0.3 -0.0 -0.1 -0.1 -0.1 -0.1 -0.3 -0.5
  - largest error per slice: +0.6 +0.6 +0.5 +0.1 +0.1 +0.1 +0.1 +0.2 +0.4 +0.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.044 | 5.11% |
| constant inside reciprocal | 0 | 0.02162 | 4.82% |
| offset | 0 | -0.01633 | 0.95% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0344, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.181 | 0.941 | 0.82 | 0.17 | 0.01 | 0.00 | 0.00 | 0.231 | + | none (best 0.0309) |
| 1 | 180 | -0.28 | 2.39 | 0.38 | 0.43 | 0.19 | 0.00 | 0.00 | 0.377 | + | none (best 0.0404) |
| 2 | 180 | -0.557 | 4.5 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 | 0.406 | + | none (best 0.0286) |
| 3 | 180 | -0.122 | 1.28 | 0.74 | 0.23 | 0.03 | 0.00 | 0.00 | 0.289 | + | none (best 0.0359) |
| 4 | 180 | 0.224 | 8.67 | 0.00 | 0.28 | 0.72 | 0.00 | 0.00 | 0.147 | + | rounded hinge |
| 5 | 180 | -0.26 | 0.762 | 0.85 | 0.14 | 0.01 | 0.00 | 0.00 | 0.314 | + | none (best 0.0435) |
| 6 | 0 | 0.814 | 7.29 | 0.00 | 0.00 | 0.87 | 0.13 | 0.00 | 0.138 | + | rounded hinge |
| 7 | 0 | 0.739 | 6.97 | 0.00 | 0.00 | 0.82 | 0.18 | 0.00 | 0.0954 | - | rounded hinge |

## Groups

- same line: units 0, 1, 3, 5 (gate angle 180 degrees, offset -0.181)
- same line: units 6, 7 (gate angle 0 degrees, offset 0.814)
- the rest: units 2, 4
