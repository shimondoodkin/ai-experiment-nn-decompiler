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
    return 1.053 / (x0 + 0.0235) - 0.02749
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.9%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.2 -0.1 +0.1 +0.3 +0.4 +0.6 +0.7 +0.8 +0.8 +0.9
  - largest error per slice: +0.9 +0.1 +0.2 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +0.9
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = 0.591)
  - mean error in 10 equal slices of x0: -0.1 +0.4 -0.0 -0.1 -0.1 -0.2 -0.4 -0.5 -0.6 -0.4
  - largest error per slice: +0.6 +0.6 +0.1 +0.1 +0.1 +0.3 +0.5 +0.6 +0.6 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.053 | 6.11% |
| constant inside reciprocal | 0 | 0.0235 | 5.18% |
| offset | 0 | -0.02749 | 1.58% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0347, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -3.41 | 6.64 | 0.00 | 0.00 | 0.00 | 0.78 | 0.22 | 0.0689 | + | rounded hinge |
| 1 | 180 | 4.34 | 5.17 | 0.90 | 0.10 | 0.00 | 0.00 | 0.00 | 0.0255 | - | rounded hinge |
| 2 | 0 | 1.2 | 6.56 | 0.00 | 0.00 | 0.90 | 0.10 | 0.00 | 0.168 | - | rounded hinge |
| 3 | 180 | -8.53 | 29.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.18 | + | rounded hinge |
| 4 | 0 | 0.613 | 2.18 | 0.00 | 0.00 | 0.27 | 0.39 | 0.34 | 0.808 | - | none (best 0.0789) |
| 5 | 0 | -159 | 169 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00128 | + | constant |
| 6 | 180 | 8.14 | 37.4 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0212 | - | rounded hinge |
| 7 | 180 | -0.383 | 0.706 | 0.83 | 0.13 | 0.04 | 0.00 | 0.00 | 0.695 | + | none (best 0.086) |

## Groups

- negligible: units 5 (magnitude 0.00128 together; a quarter of the tolerance is 0.00867)
- the rest: units 0, 1, 2, 3, 4, 6, 7
