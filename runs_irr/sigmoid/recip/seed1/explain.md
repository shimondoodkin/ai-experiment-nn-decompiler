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
    return 1.04 / (x0 + 0.01977) - 0.01421
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.8%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.0 -0.3 -0.2 -0.1 -0.0 +0.1 +0.1 +0.2 +0.3 +0.3
  - largest error per slice: +0.8 +0.3 +0.3 +0.2 +0.1 +0.1 +0.2 +0.2 +0.3 +0.3
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 0.591)
  - mean error in 10 equal slices of x0: -0.1 +0.4 +0.2 +0.0 -0.1 -0.1 -0.1 -0.1 -0.2 -0.4
  - largest error per slice: +0.5 +0.4 +0.3 +0.1 +0.1 +0.1 +0.1 +0.2 +0.3 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.04 | 4.63% |
| constant inside reciprocal | 0 | 0.01977 | 4.41% |
| offset | 0 | -0.01421 | 0.82% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0345, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -20.8 | 85.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00767 | + | constant |
| 1 | 180 | -0.259 | 1.48 | 0.65 | 0.26 | 0.09 | 0.00 | 0.00 | 0.354 | + | none (best 0.0411) |
| 2 | 180 | -0.729 | 4.84 | 0.00 | 0.42 | 0.58 | 0.00 | 0.00 | 0.396 | + | none (best 0.0255) |
| 3 | 180 | -50.5 | 274 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00309 | + | constant |
| 4 | 180 | -0.356 | 0.639 | 0.86 | 0.12 | 0.03 | 0.00 | 0.00 | 0.467 | + | none (best 0.066) |
| 5 | 0 | 0.642 | 3.55 | 0.00 | 0.00 | 0.42 | 0.58 | 0.00 | 0.126 | - | none (best 0.0101) |
| 6 | 180 | -0.384 | 1.74 | 0.54 | 0.31 | 0.15 | 0.00 | 0.00 | 0.335 | + | none (best 0.0382) |
| 7 | 0 | 0.581 | 5.69 | 0.00 | 0.00 | 0.63 | 0.37 | 0.00 | 0.0381 | - | rounded hinge |

## Groups

- negligible: units 0, 3 (magnitude 0.0108 together; a quarter of the tolerance is 0.00863)
- same line: units 1, 4, 6 (gate angle 180 degrees, offset -0.259)
- same line: units 2, 5, 7 (gate angle 180 degrees, offset -0.729)
