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
    return 1.046 / (x0 + 0.02129) - 0.01908
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.0 -0.3 -0.2 -0.0 +0.1 +0.2 +0.3 +0.4 +0.4 +0.5
  - largest error per slice: +0.7 +0.3 +0.2 +0.1 +0.2 +0.3 +0.3 +0.4 +0.5 +0.5
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: -0.0 +0.4 -0.0 +0.0 +0.0 -0.1 -0.4 -0.6 -0.6 -0.4
  - largest error per slice: +0.6 +0.6 +0.1 +0.1 +0.1 +0.3 +0.5 +0.6 +0.6 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.046 | 5.31% |
| constant inside reciprocal | 0 | 0.02129 | 4.72% |
| offset | 0 | -0.01908 | 1.10% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0346, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -1.25 | 1.89 | 0.00 | 0.00 | 0.00 | 0.04 | 0.96 | 0.0165 | + | constant |
| 1 | 180 | 1.38 | 1.88 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0111 | - | constant |
| 2 | 180 | -7.12 | 21.6 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.246 | + | rounded hinge |
| 3 | 0 | -1.28 | 2.37 | 0.00 | 0.00 | 0.00 | 0.17 | 0.83 | 0.0323 | + | rounded hinge |
| 4 | 180 | 0.485 | 1.23 | 0.93 | 0.07 | 0.00 | 0.00 | 0.00 | 0.0225 | - | rounded hinge |
| 5 | 0 | 0.367 | 0.745 | 0.00 | 0.00 | 0.04 | 0.14 | 0.82 | 0.787 | - | none (best 0.0994) |
| 6 | 180 | 123 | 289 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00412 | - | constant |
| 7 | 0 | 0.751 | 2.13 | 0.00 | 0.00 | 0.30 | 0.38 | 0.32 | 0.785 | - | none (best 0.0704) |

## Groups

- negligible: units 6 (magnitude 0.00412 together; a quarter of the tolerance is 0.00866)
- lines: units 0, 1 (each a constant or a line on the box; together one linear term)
- the rest: units 2, 3, 4, 5, 7
